"""
Train Mistral 7B on OpenWebMath dataset using LoRA.
Integrates Weights & Biases (wandb) for tracking.
Uses Unsloth.ai for faster and more efficient training.
Explicitly adds EOS tokens to ensure proper sequence termination.
Uses batched processing to minimize memory usage.
Supports loading data from local directory.
"""

import os
import torch
import wandb
import sys
from transformers import (
    AutoTokenizer, 
    TrainingArguments
)
from datasets import load_dataset, load_from_disk

# Import Unsloth and the UnslothTrainer for LoRA training
from unsloth import FastLanguageModel, UnslothTrainer, UnslothTrainingArguments, is_bfloat16_supported


parent_dir = os.path.abspath(os.path.join(os.getcwd(), '../../..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from utils.helper import get_device, TrainingSpeedCallback 
from utils.data import get_local_data



def main():
    # Parse command-line arguments for testing mode and local data
    import argparse
    parser = argparse.ArgumentParser(description='Train Mistral 7B with Unsloth on OpenWebMath')
    parser.add_argument('--test', action='store_true', help='Run in testing mode with limited data')
    parser.add_argument('--samples', type=int, default=5000, help='Number of samples to use in testing mode')
    parser.add_argument('--local_data', action='store_true', help='Use locally downloaded dataset')
    args = parser.parse_args()

    # create wandb config to log parameter
    config = {
            "model_name": "unsloth/mistral-7b-bnb-4bit",
            "dataset": "open-web-math",
            "streaming": False,  # We need to use non-streaming but will optimize memory
            "max_length": 1024,
            "total_samples": 20000,
            "num_train_epochs": 6,    # Number of complete passes through the dataset
            "learning_rate": 5e-5,    # Unsloth can handle slightly higher learning rates
            "embedding_learning_rate": 5e-6, 
            "batch_size": 4,          # Unsloth is more memory efficient
            "gradient_accumulation_steps": 4,
            "num_workers": 8,         # Parallel data loading
            "prefetch_factor": 4,     # Prefetch factor for data loading

            # Dataset configuration
            "use_local_data": True,
            "openwebmath_dataset": "open-web-math/open-web-math",
            "fineweb_dataset": "HuggingFaceFW/fineweb",
            "openwebmath_path": "math-reasoning-in-language-models/data/pre-training/open-web-math",
            "fineweb_path": "math-reasoning-in-language-models/data/pre-training/fineweb",
            "fineweb_subset": "sample-10BT",
            "openwebmath_ratio": 0.7,  # 70% from OpenWebMath
            "fineweb_ratio": 0.3,     # 30% from FineWeb
            "streaming": True,
            "shuffle_buffer": 1000,      # Buffer size for better mixing
            "max_length": 1024,
            # LoRA specific parameters
            "lora_r": 16,             # LoRA attention dimension
            "lora_alpha": 16,         # LoRA alpha parameter
            "lora_dropout": 0,        # Currently not supported    

            # Unsloth specific
            "max_seq_length": 2048,   # Unsloth can handle longer sequences efficiently

            # Testing parameters
            "testing_mode": args.test,                 # Set from command line args
            "test_sample_size": args.samples,          # Set from command line args
            "use_local_data": args.local_data          # Set from command line args
    }

    # Set the output directories
    output_dir = "./models/mistral-7b-math-lora-unsloth"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("./logs", exist_ok=True)
    
    # Initialize wandb - add testing tag if in testing mode
    run_name = "mistral-7b-openwebmath-unsloth"
    if config.get('testing_mode', False):
        run_name += "-test"
    if config.get('use_local_data', False):
        run_name += "-local"
    
    run = wandb.init(
        project="mistral-math-lora-test", 
        name=run_name,
        config=config,
        tags=["unsloth", "optimized", "testing", "local_data"] if config.get('use_local_data', False) else ["unsloth", "optimized"]
    )

    # Check for available hardware
    device = get_device()
    print(f"Using device: {device}")
    
    # Load model and tokenizer with Unsloth's FastLanguageModel
    model_name = config['model_name']
    print(f"Loading pre-trained model with Unsloth: {model_name}")
    
    # Load the model with Unsloth optimizations
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=config['max_seq_length'],
        dtype=torch.bfloat16,  # Using bfloat16 for better training stability
        load_in_4bit=True      # Unsloth works very well with 4-bit quantization
    )
    
    # Store the EOS token for use in data processing
    EOS_TOKEN = tokenizer.eos_token
    print(f"Using EOS token: {EOS_TOKEN}")
    
    # Set padding token if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Apply LoRA using Unsloth's adapter
    model = FastLanguageModel.get_peft_model(
        model,
        r=config['lora_r'],
        lora_alpha=config['lora_alpha'],
        lora_dropout=config['lora_dropout'],
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj", 
            "gate_proj", "up_proj", "down_proj",
            "embed_tokens", "lm_head"], # Add for continual pretraining
        bias="none",
        use_gradient_checkpointing="unsloth",
        use_rslora = True, 
        loftq_config = None
    )
    
    # Print trainable parameters info
    model.print_trainable_parameters()

    # Shuffle the dataset
    train_dataset = get_local_data(config)
    
    # Process the dataset with batched operations for memory efficiency
    # Define a format function that adds EOS tokens
    def format_prompt(examples):
        # When batched=True, process multiple examples at once
        return {
            "text": [text + EOS_TOKEN for text in examples["text"]]
        }
    
    # Process the dataset in batches to save memory
    print("Processing dataset with batched operations...")
    train_dataset = train_dataset.map(
        format_prompt,
        batched=True
    )
    
    
    # Create the TrainingSpeedCallback to track training performance
    training_speed_tracker = TrainingSpeedCallback()
    
    # Initialize trainer using UnslothTrainer
    trainer = UnslothTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        dataset_text_field="text",
        max_seq_length=config["max_seq_length"],
        dataset_num_proc=config["num_workers"],
        args=UnslothTrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            per_device_train_batch_size=config["batch_size"],
            gradient_accumulation_steps=config["gradient_accumulation_steps"],
            save_steps=1000,
            save_total_limit=2,
            logging_steps=10,
            logging_dir="./logs",
            
            # Training Parameters
            bf16=is_bfloat16_supported(),
            learning_rate=config["learning_rate"],
            embedding_learning_rate = config["embedding_learning_rate"],
            weight_decay=0.00,
            warmup_ratio=0.1,
            num_train_epochs=config["num_train_epochs"],
            optim = "adamw_8bit",

            
            # No evaluation dataset
            eval_strategy="no",  # Changed from evaluation_strategy to avoid deprecation warning
            report_to="wandb",
            lr_scheduler_type="cosine",
            
            # Performance options
            disable_tqdm=False,
            
            # Gradient and optimizer settings
            gradient_checkpointing=True,
            
            # Seed for reproducibility
            seed=42
        ),
        callbacks=[training_speed_tracker]
    )
    
    # Start training
    print("Starting Unsloth-accelerated training...")
    trainer.train()
    
    # Save model locally and in wandb
    model_save_path = f"{output_dir}/final"
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    print(f"Unsloth-optimized model saved to {model_save_path}")
    
    # Log model to wandb
    artifact = wandb.Artifact("mistral-7b-math-unsloth-model", type="model")
    artifact.add_dir(model_save_path)
    run.log_artifact(artifact)
    
    ############
    ### Sample generation to test the model
    print("\nGenerating sample output...")
    test_prompt = "The solution to the integral of x^2 is"
    
    # For inference, we can use the model as is on GPU if available
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