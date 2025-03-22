"""
Train GPT-2 on OpenWebMath dataset with different GPU optimization configurations.
Integrates Weights & Biases (wandb) for tracking performance metrics.
"""

import os
import time
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

# Import configurations
from gpu_study_configs import config_1, config_2, config_3, config_4, config_5, config_6, config_7, config_8

parent_dir = os.path.abspath(os.path.join(os.getcwd(), '../..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from utils.helper import get_device, WandbModelLogger, MemoryManagementCallback, TrainingSpeedCallback


def log_gpu_metrics():
    """Log GPU utilization metrics to wandb"""
    if torch.cuda.is_available():
        gpu_stats = torch.cuda.memory_stats()
        wandb.log({
            "gpu_allocated_memory_gb": torch.cuda.memory_allocated() / 1e9,
            "gpu_reserved_memory_gb": torch.cuda.memory_reserved() / 1e9,
            "gpu_max_allocated_memory_gb": gpu_stats.get("allocated_bytes.all.peak", 0) / 1e9,
            "gpu_utilization": torch.cuda.utilization()
        })


def run_experiment(config, experiment_name):
    """Run a single experiment with the given configuration"""
    
    # Set the output directories
    output_dir = f"./models/gpt2-math-{experiment_name}"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("./logs", exist_ok=True)
    
    # Initialize wandb
    run = wandb.init(
        project="gpt2-openwebmath-optimization", 
        name=f"experiment-{experiment_name}",
        config=config,
        reinit=True  # Allow reinitializing for multiple experiments
    )

    # Check for available hardware
    device = get_device()
    
    # Log initial GPU state
    log_gpu_metrics()
    
    # Load model and tokenizer
    model_name = config['model_name']
    print(f"[{experiment_name}] Loading pre-trained model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Set padding token
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id
    
    # Load dataset in streaming mode
    print(f"[{experiment_name}] Loading OpenWebMath dataset in streaming mode...")
    dataset = load_dataset("open-web-math/open-web-math", streaming=True)
    
    # Shuffle the dataset
    shuffle_buffer_size = config['shuffle_buffer']
    print(f"[{experiment_name}] Setting up streaming pipeline with shuffle buffer size: {shuffle_buffer_size}")
    train_dataset = dataset["train"].shuffle(buffer_size=shuffle_buffer_size)

    # Define custom Tokenization function
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=config['max_length'],
            padding="max_length",
            return_tensors="pt"
        )
    
    # Apply tokenization
    train_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        batch_size=64,
        remove_columns=["url", "date", "metadata", "text"]
    )
    
    # Create data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        evaluation_strategy="no",
        report_to="wandb",
        max_steps=config["max_steps"],
        save_steps=1000,
        save_total_limit=2,
        logging_steps=10,
        logging_dir="./logs",
        weight_decay=config["weight_decay"],
        warmup_steps=config["warmup_steps"],
        
        # GPU optimizations
        per_device_train_batch_size=config["per_device_train_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"] or 1,
        gradient_checkpointing=config["gradient_checkpointing"],
        fp16=config["fp16"],
        bf16=config["bf16"],
        dataloader_num_workers=config["num_workers"] or 0,
        dataloader_pin_memory=config["dataloader_pin_memory"],
        learning_rate=config["learning_rate"],
        lr_scheduler_type=config["lr_scheduler_type"] or "linear",
        
        # Performance options
        disable_tqdm=False,
        torch_compile=True,
    )
    
    if config["optimizer"] is not None:
        training_args.optim = config["optimizer"]
    
    # Create callbacks
    wandb_logger = WandbModelLogger(
        output_dir=output_dir,
        tokenizer=tokenizer,
        save_steps=10000,
        model_name_prefix=f"gpt2-math-{experiment_name}"
    )
    
    memory_manager = MemoryManagementCallback(clear_cache_steps=100)
    training_speed_tracker = TrainingSpeedCallback()
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        callbacks=[wandb_logger, memory_manager, training_speed_tracker]
    )
    
    # Enable cudnn benchmark for faster training
    torch.backends.cudnn.benchmark = True
    
    # Start training
    print(f"[{experiment_name}] Starting training with optimization configuration...")
    start_time = time.time()
    trainer.train()
    training_time = time.time() - start_time
    
    # Log final metrics
    wandb.log({
        "total_training_time": training_time,
        "steps_per_second": config["max_steps"] / training_time,
    })
    
    # Log GPU metrics after training
    log_gpu_metrics()
    
    # Log final GPU metrics only
    log_gpu_metrics()
    
    # Finish the wandb run
    wandb.finish()


def main():
    """Run all experiments with different configurations"""
    
    # List of configurations and their names
    experiments = [
        (config_1, "vanilla"),
        (config_2, "gradient_accumulation"),
        (config_3, "gradient_checkpointing"),
        (config_4, "fp16"),
        (config_5, "bf16"),
        (config_6, "adafactor_optimizer"),
        (config_7, "pin_memory"),
        (config_8, "multi_workers")
    ]
    
    # Run each experiment
    for config, name in experiments:
        print(f"\n{'='*50}")
        print(f"Starting experiment: {name}")
        print(f"{'='*50}\n")
        run_experiment(config, name)
        
        # Optional: clear CUDA cache between experiments
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()