"""
Train Mistral 7B on OpenWebMath dataset using LoRA and streaming to avoid downloading the full dataset.
Integrates Weights & Biases (wandb) for tracking.
"""

import os
import torch
import wandb
import sys
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer, 
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
import bitsandbytes as bnb

parent_dir = os.path.abspath(os.path.join(os.getcwd(), '../..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from utils.helper import get_device, WandbModelLogger, MemoryManagementCallback  # Import custom logger


def main():
    # Parse command-line arguments for testing mode
    import argparse
    parser = argparse.ArgumentParser(description='Train Mistral 7B with LoRA on OpenWebMath')
    parser.add_argument('--test', action='store_true', help='Run in testing mode with limited data')
    parser.add_argument('--samples', type=int, default=1000, help='Number of samples to use in testing mode')
    args = parser.parse_args()

    # create wandb config to log parameter
    config = {
            "model_name": "mistralai/Mistral-7B-v0.1",
            "dataset": "open-web-math",
            "streaming": True,
            "shuffle_buffer": 5000,  # Shuffle buffer size for better mixing
            "max_length": 1024,
            "max_steps": 50000,
            "learning_rate": 2e-4,
            "batch_size": 4,         # Smaller batch size due to larger model
            "gradient_accumulation_steps": 8,  # Increased to compensate for smaller batch size
            "num_workers": 4,         # Parallel data loading
            "prefetch_factor": 2,     # Prefetch factor for data loading
            # LoRA specific parameters
            "lora_r": 16,             # LoRA attention dimension
            "lora_alpha": 32,         # LoRA alpha parameter
            "lora_dropout": 0.05,     # Dropout probability for LoRA layers
            "load_in_8bit": True,     # Use 8-bit quantization to reduce memory requirements
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            # Testing parameters
            "testing_mode": args.test,                 # Set from command line args
            "test_sample_size": args.samples           # Set from command line args
    }

    # Set the output directories
    output_dir = "./models/mistral-7b-math-lora"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("./logs", exist_ok=True)
    
    # Initialize wandb - add testing tag if in testing mode
    run_name = "mistral-7b-openwebmath-lora-test" if config.get('testing_mode', False) else "mistral-7b-openwebmath-lora"
    
    run = wandb.init(
        project="mistral-math-lora", 
        name=run_name,
        config=config,
        tags=["testing"] if config.get('testing_mode', False) else None
    )

    # Check for available hardware
    device = get_device()
    
    # Load model and tokenizer
    model_name = config['model_name']
    print(f"Loading pre-trained model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load in 8-bit precision to save GPU memory
    print(f"Loading model in 8-bit precision...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_8bit=config['load_in_8bit'],
        device_map="auto",
        torch_dtype=torch.float16
    )
    
    # Set padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
    
    # Prepare model for LoRA training
    print("Preparing model for LoRA fine-tuning...")
    model = prepare_model_for_kbit_training(model)
    
    # Define LoRA configuration
    lora_config = LoraConfig(
        r=config['lora_r'],
        lora_alpha=config['lora_alpha'],
        target_modules=config['target_modules'],
        lora_dropout=config['lora_dropout'],
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    # Apply LoRA to the model
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters info
    model.print_trainable_parameters()
    
    # Load dataset in streaming mode
    print("Loading OpenWebMath dataset in streaming mode...")
    dataset = load_dataset("open-web-math/open-web-math", streaming=True)
    
    # Shuffle the dataset
    shuffle_buffer_size = config['shuffle_buffer']
    print(f"Setting up streaming pipeline with shuffle buffer size: {shuffle_buffer_size}")
    train_dataset = dataset["train"].shuffle(buffer_size=shuffle_buffer_size)
    
    # Limit dataset size for testing if testing_mode is enabled
    if config.get('testing_mode', False):
        print(f"TESTING MODE: Limiting dataset to {config['test_sample_size']} examples")
        train_dataset = train_dataset.take(config['test_sample_size'])

    # Define custom Tokenization function
    def tokenize_function(examples, config=config):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=config['max_length'],
            padding="max_length",
            return_tensors="pt"
        )
    
    # Apply tokenization to the dataset
    train_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        batch_size=32,  # Process in smaller batches
        remove_columns=["url", "date", "metadata", "text"]
    )
    
    # Create data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # We're doing causal language modeling, not masked
    )
    
    # For testing: set a much smaller number of steps
    max_steps = 100 if config.get('testing_mode', False) else config["max_steps"]
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=config["batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        save_steps=1000,
        save_total_limit=2,
        logging_steps=10,
        logging_dir="./logs",
        
        # Mixed precision settings - using FP16 as BF16 is handled by 8-bit quantization
        fp16=True,
        dataloader_num_workers=config["num_workers"],
        dataloader_pin_memory=True,
        learning_rate=config["learning_rate"],
        weight_decay=0.01,
        warmup_steps=50 if config.get('testing_mode', False) else 500,  # Reduced warmup for testing
        max_steps=max_steps,  # Use reduced steps for testing mode
        evaluation_strategy="no",
        report_to="wandb",
        lr_scheduler_type="cosine",
        
        # Performance options
        disable_tqdm=False,
        
        # Gradient and optimizer settings
        gradient_checkpointing=True,  # Enable gradient checkpointing to save memory
        torch_compile=False,  # Disable torch compile for LoRA compatibility
    )
    
    # Save model every 5000 steps
    wandb_logger = WandbModelLogger(
        output_dir=output_dir,
        tokenizer=tokenizer,
        save_steps=5000,
        model_name_prefix="mistral-7b-math-lora"
    )

    # Clear cache every 50 steps (more frequent due to larger model)
    memory_manager = MemoryManagementCallback(clear_cache_steps=50)
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        callbacks=[wandb_logger, memory_manager]
    )
    
    # Start training
    print("Starting LoRA training with streaming dataset...")
    trainer.train()
    
    # Save model locally and in wandb
    model_save_path = f"{output_dir}/final"
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    print(f"LoRA model saved to {model_save_path}")
    
    # Log model to wandb
    artifact = wandb.Artifact("mistral-7b-math-lora-model", type="model")
    artifact.add_dir(model_save_path)
    run.log_artifact(artifact)
    
    # Sample generation to test the model
    print("\nGenerating sample output...")
    test_prompt = "The solution to the integral of x^2 is"
    
    # Move to CPU for inference if needed (due to memory constraints)
    if torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory > 40 * 1024 * 1024 * 1024:
        # If we have a high-memory GPU (40+ GB), we can do inference there
        input_device = device
    else:
        # Otherwise, do inference on CPU
        input_device = "cpu"
        model = model.to(input_device)
    
    input_ids = tokenizer(test_prompt, return_tensors="pt").input_ids.to(input_device)
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_length=100,
            temperature=0.7,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Prompt: {test_prompt}")
    print(f"Generated: {generated_text}")
    
    # Log the generated text to wandb
    wandb.log({"example_generation": wandb.Html(f"<p><strong>Prompt:</strong> {test_prompt}</p><p><strong>Generated:</strong> {generated_text}</p>")})

    # Log training performance metrics
    if torch.cuda.is_available():
        gpu_stats = torch.cuda.memory_stats()
        wandb.log({
            "gpu_allocated_memory_gb": torch.cuda.memory_allocated() / 1e9,
            "gpu_reserved_memory_gb": torch.cuda.memory_reserved() / 1e9,
            "gpu_max_allocated_memory_gb": gpu_stats.get("allocated_bytes.all.peak", 0) / 1e9,
        })
    
    # Finish the wandb run
    wandb.finish()

if __name__ == "__main__":
    main()