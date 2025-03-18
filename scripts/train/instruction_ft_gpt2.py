"""
Instruction fine-tuning for a pre-trained GPT-2 math model using MathInstruct dataset.
Allows filtering by source, limiting sample count, and random sampling.
Integrates Weights & Biases (wandb) for tracking.
"""

import os
import torch
import wandb
import sys
import random
from collections import Counter
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer, 
    DataCollatorForLanguageModeling
)
from datasets import load_dataset

parent_dir = os.path.abspath(os.path.join(os.getcwd(), '../..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from utils.helper import get_device
except ImportError:
    # Fallback implementation if the helper module is not available
    def get_device():
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

def main():
    # Create wandb config to log parameters
    config = {
        "model_name": "gpt2-math-curr",  # Your pre-trained model
        "dataset": "TIGER-Lab/MathInstruct",
        "max_length": 1024,
        "max_steps": 10000,            # Adjust based on your needs
        "learning_rate": 2e-5,         # Slightly lower for instruction tuning
        "batch_size": 16,              # Adjusted for instruction tuning
        "gradient_accumulation_steps": 2,
        "num_workers": 4,
        "max_samples": 10000,          # Maximum number of samples to use
        "sources_to_include": [],      # Empty means include all (populated later)
        "random_seed": 42              # For reproducibility
    }

    # Set the output directories
    output_dir = f"./models/{config['model_name']}-instruct"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("./logs", exist_ok=True)
    
    # Initialize wandb
    run = wandb.init(
        project="math-instruct", 
        name="gpt2-math-curr-instruct",
        config=config
    )

    # Set random seed for reproducibility
    random.seed(config["random_seed"])
    torch.manual_seed(config["random_seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["random_seed"])

    # Check for available hardware
    device = get_device()
    print(f"Using device: {device}")
    
    # Load model and tokenizer
    model_name = config['model_name']
    print(f"Loading pre-trained model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Set padding token if not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
    
    # Load MathInstruct dataset
    print("Loading MathInstruct dataset...")
    dataset = load_dataset("TIGER-Lab/MathInstruct", split="train")
    
    # Print distinct source values
    source_counts = Counter(dataset["source"])
    print("\nDistinct sources in the dataset:")
    for source, count in source_counts.items():
        print(f"- {source}: {count} examples")
    
    # Log source distribution to wandb
    wandb.log({"source_distribution": wandb.Table(
        columns=["Source", "Count"], 
        data=[[source, count] for source, count in source_counts.items()]
    )})
    
    # Filter dataset by source (if specified)
    if config["sources_to_include"]:
        print(f"Filtering to include only the following sources: {config['sources_to_include']}")
        dataset = dataset.filter(lambda example: example["source"] in config["sources_to_include"])
        print(f"Dataset filtered to {len(dataset)} examples")
    else:
        print("Using all sources (no filtering applied)")
    
    # Randomly select a subset of samples if needed
    if config["max_samples"] and config["max_samples"] < len(dataset):
        print(f"Randomly selecting {config['max_samples']} examples from {len(dataset)} available examples")
        selected_indices = random.sample(range(len(dataset)), config["max_samples"])
        dataset = dataset.select(selected_indices)
        print(f"Dataset sampled to {len(dataset)} examples")
    
    # Prepare the prompt template function
    def prepare_instruction_data(examples):
        """Format the instruction data in a format suitable for GPT-2"""
        
        # Create formatted instruction texts
        formatted_texts = []
        
        for i in range(len(examples["instruction"])):
            instruction = examples["instruction"][i]
            input_text = examples["input"][i] if examples["input"][i] else ""
            output = examples["output"][i]
            
            # Format as a single text: instruction + input (if any) + output
            if input_text:
                formatted_text = f"Instruction: {instruction}\nInput: {input_text}\nOutput: {output}"
            else:
                formatted_text = f"Instruction: {instruction}\nOutput: {output}"
            
            formatted_texts.append(formatted_text)
        
        return {"formatted_text": formatted_texts}
    
    # Apply the formatting
    processed_dataset = dataset.map(
        prepare_instruction_data,
        batched=True,
        remove_columns=dataset.column_names  # Remove original columns
    )
    
    print(f"Sample formatted instruction data:\n{processed_dataset[0]['formatted_text']}")
    
    # Tokenize the formatted text
    def tokenize_function(examples):
        return tokenizer(
            examples["formatted_text"],
            truncation=True,
            max_length=config['max_length'],
            padding="max_length",
            return_tensors="pt"
        )
    
    tokenized_dataset = processed_dataset.map(
        tokenize_function,
        batched=True,
        batch_size=64,
        remove_columns=["formatted_text"]
    )
    
    # Create data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # GPT-2 uses causal language modeling, not masked
    )
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,                          # Directory where model checkpoints and logs will be saved
        overwrite_output_dir=True,                      # If output_dir exists, overwrite instead of erroring
        per_device_train_batch_size=config["batch_size"], # Number of samples processed per GPU during training
        gradient_accumulation_steps=config["gradient_accumulation_steps"], # Number of forward passes before updating parameters
        save_steps=1000,                                # Save a checkpoint every 1000 steps
        save_total_limit=2,                             # Keep only the 2 most recent checkpoints to save disk space
        logging_steps=10,                               # Log metrics every 10 steps for more detailed monitoring
        logging_dir="./logs",                           # Directory for storing training logs
        
        # Hardware optimizations
        bf16=torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8,  # Use BF16 on Ampere or newer GPUs
        bf16_full_eval=torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8,  # Use BF16 during evaluation as well
        fp16=torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 8,  # Fallback to FP16 on older GPUs
        dataloader_num_workers=config["num_workers"],   # Number of CPU workers for data loading
        dataloader_pin_memory=True,                     # Pin memory in CPU to accelerate CPU to GPU transfer
        learning_rate=config["learning_rate"],          # Initial learning rate for the optimizer
        weight_decay=0.01,                              # L2 regularization factor to prevent overfitting
        warmup_steps=200,                               # Gradually increase learning rate for first 200 steps for stability
        max_steps=config["max_steps"],                  # Total number of training steps
        evaluation_strategy="no",                       # Disable evaluation during training
        report_to="wandb",                              # Log metrics to Weights & Biases for visualization
        
        # Performance options
        disable_tqdm=False,                             # Show progress bar for monitoring training progress
        
        # Advanced optimization (PyTorch 2.0+)
        torch_compile=True,                             # Enable PyTorch compiler for just-in-time optimization
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset,
    )
    
    # Enable cudnn benchmark for faster training
    torch.backends.cudnn.benchmark = True
    
    # Start training
    print("Starting instruction fine-tuning...")
    trainer.train()
    
    # Save model locally and in wandb
    model_save_path = f"{output_dir}/final"
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    print(f"Model saved to {model_save_path}")
    
    # Log model to wandb
    artifact = wandb.Artifact(f"{config['model_name']}-instruct", type="model")
    artifact.add_dir(model_save_path)
    run.log_artifact(artifact)
    
    # Sample generation to test the model
    print("\nGenerating sample outputs...")
    test_prompts = [
        "Instruction: Solve the equation 2x + 3 = 7\nOutput:",
        "Instruction: Find the derivative of f(x) = x^2 * sin(x)\nOutput:",
        "Instruction: Calculate the area of a circle with radius 5 cm\nOutput:"
    ]
    
    model = model.to(device)
    for prompt in test_prompts:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_length=200,
                temperature=0.7,
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\nPrompt: {prompt}")
        print(f"Generated: {generated_text}")
        
        # Log the generated text to wandb
        wandb.log({"example_generation": wandb.Html(f"<p><strong>Prompt:</strong> {prompt}</p><p><strong>Generated:</strong> {generated_text}</p>")})
    
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