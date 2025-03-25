"""
Train Mistral 7B on OpenWebMath dataset using LoRA.
Integrates Weights & Biases (wandb) for tracking.
Uses Unsloth.ai for faster and more efficient training.
Explicitly adds EOS tokens to ensure proper sequence termination.
"""

import os
import torch
import wandb
import sys
from transformers import (
    AutoTokenizer, 
    TrainingArguments
)
from datasets import load_dataset
# Import Unsloth and the UnslothTrainer
from unsloth import FastLanguageModel, UnslothTrainer, UnslothTrainingArguments, is_bfloat16_supported
# Keep PEFT for LoRA config
from peft import TaskType

parent_dir = os.path.abspath(os.path.join(os.getcwd(), '../../..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from utils.helper import get_device, TrainingSpeedCallback  # Import custom logger


def main():
    # Parse command-line arguments for testing mode
    import argparse
    parser = argparse.ArgumentParser(description='Train Mistral 7B with Unsloth on OpenWebMath')
    parser.add_argument('--test', action='store_true', help='Run in testing mode with limited data')
    parser.add_argument('--samples', type=int, default=1000, help='Number of samples to use in testing mode')
    args = parser.parse_args()

    # create wandb config to log parameter
    config = {
            "model_name": "mistralai/Mistral-7B-v0.1",
            "dataset": "open-web-math",
            "streaming": False,  # Changed to non-streaming
            "max_length": 1024,
            "max_steps": 50000,
            "learning_rate": 2e-4,    # Unsloth can handle slightly higher learning rates
            "batch_size": 8,          # Unsloth is more memory efficient
            "gradient_accumulation_steps": 4,
            "num_workers": 4,         # Parallel data loading
            "prefetch_factor": 2,     # Prefetch factor for data loading
            # LoRA specific parameters
            "lora_r": 16,             # LoRA attention dimension
            "lora_alpha": 32,         # LoRA alpha parameter
            "lora_dropout": 0.05,     # Dropout probability for LoRA layers
            # Unsloth specific
            "max_seq_length": 2048,   # Unsloth can handle longer sequences efficiently
            # Testing parameters
            "testing_mode": args.test,                 # Set from command line args
            "test_sample_size": args.samples           # Set from command line args
    }

    # Set the output directories
    output_dir = "./models/mistral-7b-math-lora-unsloth"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("./logs", exist_ok=True)
    
    # Initialize wandb - add testing tag if in testing mode
    run_name = "mistral-7b-openwebmath-unsloth"
    if config.get('testing_mode', False):
        run_name += "-test"
    
    run = wandb.init(
        project="mistral-math-lora", 
        name=run_name,
        config=config,
        tags=["unsloth", "testing"] if config.get('testing_mode', False) else ["unsloth"]
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
            "gate_proj", "up_proj", "down_proj"
        ],
        bias="none"
    )
    
    # Print trainable parameters info
    model.print_trainable_parameters()
    
    # Load dataset in non-streaming mode
    print("Loading OpenWebMath dataset in non-streaming mode...")
    
    # If in testing mode, only load a small subset
    if config.get('testing_mode', False):
        print(f"TESTING MODE: Loading only {config['test_sample_size']} examples")
        dataset = load_dataset(
            "open-web-math/open-web-math", 
            streaming=False,
            split=f"train[:{config['test_sample_size']}]"  # Load just a slice of the dataset
        )
    else:
        # Load the full dataset
        dataset = load_dataset(
            "open-web-math/open-web-math", 
            streaming=False,
            split="train"
        )
    
    # Shuffle the dataset (non-streaming version)
    train_dataset = dataset.shuffle(seed=42)

    # Define prompt formatting function that explicitly adds EOS token
    def format_prompt(example):
        # When batched=True, example["text"] is a list of texts
        if isinstance(example["text"], list):
            # Process a batch of examples
            return {
                "text": [text + EOS_TOKEN for text in example["text"]]
            }
        else:
            # Process a single example
            return {
                "text": example["text"] + EOS_TOKEN
            }

    # Map formatting to dataset
    train_dataset = train_dataset.map(
        format_prompt,
        remove_columns=["url", "date", "metadata"]
        # Removed batched=True to avoid the list concatenation error
    )
    
    # For testing: set a much smaller number of steps
    max_steps = 100 if config.get('testing_mode', False) else config["max_steps"]
    
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
            
            # Mixed precision settings
            bf16=is_bfloat16_supported(),
            fp16=not is_bfloat16_supported(),
            
            learning_rate=config["learning_rate"],
            weight_decay=0.01,
            warmup_steps=50 if config.get('testing_mode', False) else 500,
            max_steps=max_steps,
            
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