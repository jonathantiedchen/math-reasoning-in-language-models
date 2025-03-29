"""
Train GPT-2 on a combination of OpenWebMath and FineWeb datasets using streaming.
The combined dataset will have 70% samples from OpenWebMath and 30% from FineWeb.
Total samples: 500,000
Integrates Weights & Biases (wandb) for tracking.
Uses epoch-based training to see samples multiple times.
"""

import os
import torch
import wandb
import sys
import random
from itertools import islice
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer, 
    DataCollatorForLanguageModeling
)
from datasets import load_dataset, Dataset, IterableDataset, interleave_datasets
import math

parent_dir = os.path.abspath(os.path.join(os.getcwd(), '../../..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from utils.helper import get_device, WandbModelLogger
from utils.data import get_mixed_dataset_tokenized


def main():

    # create wandb config to log parameter
    config = {
            "model_name": "gpt2",  # Options: "gpt2", "gpt2-medium", etc.
            "openwebmath_dataset": "open-web-math/open-web-math",
            "fineweb_dataset": "HuggingFaceFW/fineweb",
            "fineweb_subset": "sample-10BT",
            "openwebmath_path": "math-reasoning-in-language-models/data/pre-training/open-web-math",
            "fineweb_path": "math-reasoning-in-language-models/data/pre-training/fineweb",
            "streaming": True,
            "shuffle_buffer": 5000,  # Increased buffer size for better mixing
            "max_length": 1024,
            "total_samples": 500000,  # Total number of samples to use
            "openwebmath_ratio": 0.7,  # 70% from OpenWebMath
            "fineweb_ratio": 0.3,     # 30% from FineWeb
            "num_train_epochs": 6,    # Number of complete passes through the dataset
            "learning_rate": 5e-5,
            "batch_size": 64,          # Samples per device in each forward pass
            "gradient_accumulation_steps": 2,  # Number of forward passes before parameter update
            "num_workers": 4,         # Parallel data loading
            "prefetch_factor": 4      # Prefetch factor for data loading
    }

    # Set the output directories
    output_dir = "./models/gpt2-math-test"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("./logs", exist_ok=True)
    
    # Initialize wandb
    run = wandb.init(
        project="gpt2-math-fineweb", 
        name="gpt2-combined-epoch_training",
        config=config
    )

    # Check for available hardware
    device = get_device()
    
    # Load model and tokenizer
    model_name = config['model_name']
    print(f"Loading pre-trained model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Set padding token
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id
    
    train_dataset = get_mixed_dataset_tokenized(config, tokenizer)
    
    # Create data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # GPT-2 uses causal language modeling, not masked
    )
    
    # Calculate steps per epoch and total training steps
    # For streaming datasets, we need to estimate this based on our dataset size
    total_batch_size = config["batch_size"] * config["gradient_accumulation_steps"]
    steps_per_epoch = math.ceil(config["total_samples"] / total_batch_size)
    total_training_steps = steps_per_epoch * config["num_train_epochs"]
    
    print(f"Estimated steps per epoch: {steps_per_epoch}")
    print(f"Total training steps: {total_training_steps}")
    
    # Configure more frequent checkpoints and logging based on epochs
    logging_steps = max(10, steps_per_epoch // 10)  # Log ~10 times per epoch
    save_steps = steps_per_epoch // 2               # Save twice per epoch
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=config["batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        save_steps=save_steps,
        save_total_limit=3,                        # Keep only the 3 most recent checkpoints
        logging_steps=logging_steps,
        logging_dir="./logs",
        
        # Epoch-based training configuration
        num_train_epochs=config["num_train_epochs"],  # Use epochs instead of max_steps
        
        # H100-specific optimizations
        bf16=True,
        bf16_full_eval=True,
        dataloader_num_workers=config["num_workers"],
        dataloader_pin_memory=True,
        learning_rate=config["learning_rate"],
        weight_decay=0.01,
        warmup_ratio=0.03,                     # 3% of total steps for warmup
        evaluation_strategy="no",
        report_to="wandb",
        lr_scheduler_type="cosine",
        
        # Performance options
        disable_tqdm=False,
        
        # Advanced optimization (PyTorch 2.0+)
        torch_compile=True,
    )
    
    # Save model periodically 
    # Adjust based on total steps to get similar frequency as original script
    wandb_logger = WandbModelLogger(
        output_dir=output_dir,
        tokenizer=tokenizer,
        save_steps=steps_per_epoch * 3,  # Save every ~3 epochs
        model_name_prefix="gpt2-math-fineweb-combined"
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        callbacks=[wandb_logger]
    )
    
    # Enable cudnn benchmark for faster training
    torch.backends.cudnn.benchmark = True
    
    # Start training
    print("Starting training with combined streaming datasets...")
    print(f"Training for {config['num_train_epochs']} epochs")
    trainer.train()
    
    # Save model locally and in wandb
    model_save_path = f"{output_dir}/final"
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    print(f"Model saved to {model_save_path}")
    
    # Log model to wandb
    artifact = wandb.Artifact("gpt2-math-fineweb-model", type="model")
    artifact.add_dir(model_save_path)
    run.log_artifact(artifact)
    
    # Sample generation to test the model
    print("\nGenerating sample output...")
    test_prompt = "The solution to the integral of x^2 is"
    input_ids = tokenizer(test_prompt, return_tensors="pt").input_ids.to(device)
    model = model.to(device)
    
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