"""
Train GPT-2 Large on OpenWebMath dataset using LoRA for efficient adaptation.
Integrates Weights & Biases (wandb) for tracking.
"""

import os
import torch
import wandb
import sys
from transformers import BitsAndBytesConfig
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    GPT2LMHeadModel,
    GPT2Config
)
from datasets import load_dataset
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)

parent_dir = os.path.abspath(os.path.join(os.getcwd(), '../../..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from utils.helper import get_device

from utils.data import get_mixed_dataset

def main():
    # create wandb config to log parameters
    config = {
        # Model configuration
        "model_name": "gpt2-large",  # GPT-2 Large (774M parameters)
        "use_8bit": False,           # Whether to use 8-bit quantization
        "use_4bit": False,           # 4-bit quantization not typically needed for GPT-2 Large on H100

        # LoRA configuration
        "lora_r": 16,                # Rank of LoRA matrices
        "lora_alpha": 16,            # Alpha parameter for LoRA scaling
        "lora_dropout": 0.,        # Dropout probability for LoRA layers
        "lora_target_modules": ["c_attn", "c_proj", "c_fc"],

        # Dataset configuration
        "openwebmath_dataset": "open-web-math/open-web-math",
        "fineweb_dataset": "HuggingFaceFW/fineweb",
        "fineweb_subset": "CC-MAIN-2024-10",
        "openwebmath_ratio": 0.7,  # 70% from OpenWebMath
        "fineweb_ratio": 0.3,     # 30% from FineWeb
        "streaming": True,
        "shuffle_buffer": 1000,      # Buffer size for better mixing
        "max_length": 1024,

        # Training configuration
        "total_samples": 80000,
        "num_train_epochs": 6,    # Number of complete passes through the dataset
        "learning_rate": 5e-5,       # Higher LR for LoRA
        "batch_size": 4,            # H100 can handle larger batches with GPT-2 Large
        "gradient_accumulation_steps": 4,
        "num_workers": 8,            # Parallel data loading
        "prefetch_factor": 4,        # Prefetch factor for data loading
        
        # Additional optimizations
        "warmup_ratio": 0.03,        # Ratio of steps for learning rate warmup
        "logging_steps": 10,
        "save_steps": 1000
    }

    # Set the output directories
    output_dir = "./models/gpt2-large-lora-math"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("./logs", exist_ok=True)
    
    # Initialize wandb
    run = wandb.init(
        project="gpt2-large-math", 
        name="gpt2-large-lora-openwebmath",
        config=config
    )

    # Check for available hardware
    device = get_device()
    print(f"Using device: {device}")
    
    # Load model and tokenizer
    model_name = config['model_name']
    print(f"Loading pre-trained model: {model_name}")
    
    # Load tokenizer first
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Make sure tokenizer has pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with optional quantization
    if config["use_8bit"]:
        print("Loading model with 8-bit quantization...")
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            quantization_config=quantization_config
        )
        model = prepare_model_for_kbit_training(model)
    else:
        # Standard model loading - H100 has enough memory for GPT-2 Large in full precision
        model = GPT2LMHeadModel.from_pretrained(model_name)
        
    # Set pad token id in the model config
    model.config.pad_token_id = tokenizer.pad_token_id
    
    # Set up LoRA configuration
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        target_modules=config["lora_target_modules"],
        bias="none",
        inference_mode=False,
    )
    
    # Apply LoRA to model
    print("Applying LoRA adapters to model...")
    model = get_peft_model(model, peft_config)
    
    # Print number of trainable parameters
    model.print_trainable_parameters()
    
    # Load dataset in streaming mode
    train_dataset = get_mixed_dataset(config, tokenizer)
    
    # Create data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # Causal language modeling, not masked
    )
    
    # Define training arguments - optimized for H100 with GPT-2 Large
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=config["batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        save_steps=config["save_steps"],
        save_total_limit=3,
        logging_steps=config["logging_steps"],
        logging_dir="./logs",
        
        # Optimizer settings
        learning_rate=config["learning_rate"],
        weight_decay=0.01,
        warmup_ratio=config["warmup_ratio"],
        num_train_epochs=config["num_train_epochs"],

        # H100 specific settings
        fp16=True,                         
        fp16_full_eval=True,
        dataloader_num_workers=config["num_workers"],
        dataloader_pin_memory=True,
        
        # Reporting and evaluation
        evaluation_strategy="no",
        report_to="wandb",
        lr_scheduler_type="cosine",
        
        # Performance options
        disable_tqdm=False
    )
    
    # Create custom callback to save model periodically
    wandb_logger = WandbModelLogger(
        output_dir=output_dir,
        tokenizer=tokenizer,
        save_steps=10000,
        model_name_prefix="gpt2-large-lora-math"
    )

    # Memory management callback
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        callbacks=[wandb_logger]
    )
    # Add CUDA memory configuration to avoid fragmentation
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    # Enable cudnn benchmark for faster training
    torch.backends.cudnn.benchmark = True
    
    # Start training
    print("Starting LoRA training with streaming dataset...")
    trainer.train()
    
    # Save the final adapter
    adapter_save_path = f"{output_dir}/final-adapter"
    model.save_pretrained(adapter_save_path)
    tokenizer.save_pretrained(adapter_save_path)
    print(f"LoRA adapter saved to {adapter_save_path}")
    
    # Log adapter to wandb
    adapter_artifact = wandb.Artifact("gpt2-large-lora-math-adapter", type="model")
    adapter_artifact.add_dir(adapter_save_path)
    run.log_artifact(adapter_artifact)
    
    # Sample generation to test the model
    print("\nGenerating sample output...")
    test_prompt = "The solution to the integral of x^2 is"
    
    model = model.to(device)
    input_ids = tokenizer(test_prompt, return_tensors="pt").input_ids.to(device)
    
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