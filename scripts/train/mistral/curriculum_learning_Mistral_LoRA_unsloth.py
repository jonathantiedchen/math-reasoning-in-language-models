"""
Train Mistral 7B with curriculum learning approach using Unsloth and LoRA.
Integrates Weights & Biases (wandb) for tracking.
Uses Unsloth.ai for faster and more efficient training.
"""

import os
import torch
import wandb
import sys
import pandas as pd
from datasets import Dataset
# Import Unsloth and the UnslothTrainer
from unsloth import FastLanguageModel, UnslothTrainer, UnslothTrainingArguments, is_bfloat16_supported
# Keep PEFT for printing trainable parameters
from peft import TaskType

parent_dir = os.path.abspath(os.path.join(os.getcwd(), '../..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from utils.helper import get_device, TrainingSpeedCallback, WandbModelLogger
from utils.data import get_cl_learning_data, prepare_datasets_qa

# Define wandb configuration
wandb_config = {
    "model_name": "mistralai/Mistral-7B-v0.1",  # Changed to use direct model name
    "learning_rate": 2e-4,                     # Increased for Unsloth
    "batch_size": 8,
    "max_steps": 5,
    "warmup_steps": 100,
    "save_steps": 100,
    "eval_steps": 100,
    "gradient_accumulation_steps": 8,
    "lr_scheduler": "cosine",
    "training_approach": "curriculum_learning",
    "datasets": ["ASDiv", "ParaMAWPS", "SVAMP", "DMath", "AQuA"],
    "samples_per_dataset": 5,
    "test_size": 0.1,
    # LoRA specific parameters
    "lora_r": 16,             # LoRA attention dimension
    "lora_alpha": 32,         # LoRA alpha parameter
    "lora_dropout": 0.05,     # Dropout probability for LoRA layers
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    # Unsloth specific parameters
    "max_seq_length": 2048,   # Unsloth can handle longer sequences efficiently
    "num_workers": 4,         # Parallel data loading
    "prefetch_factor": 2,     # Prefetch factor for data loading
}

# Initialize wandb run for tracking overall process
run = wandb.init(
    project="mistral-math-lora", 
    name="mistral-7b-curriculum-learning-unsloth",
    config=wandb_config
)

# Check for available hardware
device = get_device()
print(f"Using device: {device}")

# Load model and tokenizer with Unsloth's FastLanguageModel
model_name = wandb_config['model_name']
print(f"Loading pre-trained model with Unsloth: {model_name}")

# Load the model with Unsloth optimizations - only once
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=wandb_config['max_seq_length'],
    dtype=torch.bfloat16 if is_bfloat16_supported() else torch.float16,
    load_in_4bit=True      # Unsloth works very well with 4-bit quantization
)

# Set padding token if needed
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Store the EOS token for use in data processing
EOS_TOKEN = tokenizer.eos_token
print(f"Using EOS token: {EOS_TOKEN}")

# Load necessary data for curriculum learning
dataset_dict = get_cl_learning_data()

def tokenize_function(examples):
    """Tokenize examples for Unsloth training"""
    # Add EOS token to each prompt
    prompts = [text + EOS_TOKEN for text in examples["prompt"]]
    return tokenizer(prompts, padding="max_length", truncation=True, max_length=wandb_config["max_seq_length"])

# Create the TrainingSpeedCallback to track training performance
training_speed_tracker = TrainingSpeedCallback()

# Iterate through datasets in curriculum order
for dataset_name, dataset_samples in dataset_dict.items():
    print(f"\n\n{'='*50}")
    print(f"Training on {dataset_name} dataset")
    print(f"{'='*50}")
    
    # Initialize a new wandb run for each dataset
    dataset_run = wandb.init(
        project="mistral-math-lora", 
        name=f"mistral-7b-{dataset_name}-unsloth",
        config=wandb_config,
        # Add this to ensure a new run is created each time
        reinit=True
    )
    
    # For each dataset, we need to start with a fresh LoRA configuration
    # Apply LoRA using Unsloth's adapter
    model = FastLanguageModel.get_peft_model(
        model,
        r=wandb_config['lora_r'],
        lora_alpha=wandb_config['lora_alpha'],
        lora_dropout=wandb_config['lora_dropout'],
        target_modules=wandb_config['target_modules'],
        bias="none"
    )
    
    # Print trainable parameters info
    model.print_trainable_parameters()
    
    # Cut AQuA Dataset to 15000 samples (or whatever limit you need)
    dataset_samples = dataset_samples[:15000] if dataset_name == "AQuA" else dataset_samples
    
    # Convert to pandas and then to Dataset
    df = pd.DataFrame(dataset_samples)
    dataset = Dataset.from_pandas(df)
    
    # Apply formatting
    dataset = dataset.map(
        prepare_datasets_qa,
        remove_columns=['question', 'answer']
    )
    
    # Split dataset
    split_dataset = dataset.shuffle(seed=42).train_test_split(test_size=0.1, seed=42)

    # Correctly access the train and test datasets
    train_dataset = split_dataset["train"]
    test_dataset = split_dataset["test"]
    
    # Set output directory for this dataset's model
    output_dir = f"./models/mistral-7b-{dataset_name}-unsloth"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("./logs", exist_ok=True)
    
    # Initialize trainer using UnslothTrainer specifically for this dataset
    trainer = UnslothTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        dataset_text_field="prompt",  # Field containing the text in your dataset
        max_seq_length=wandb_config["max_seq_length"],
        dataset_num_proc=wandb_config["num_workers"],
        args=UnslothTrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            per_device_train_batch_size=wandb_config["batch_size"],
            per_device_eval_batch_size=wandb_config["batch_size"],
            gradient_accumulation_steps=wandb_config["gradient_accumulation_steps"],
            save_steps=wandb_config["save_steps"],
            save_total_limit=2,
            logging_steps=50,
            logging_dir="./logs",
            
            # Mixed precision settings
            bf16=is_bfloat16_supported(),
            fp16=not is_bfloat16_supported(),
            
            learning_rate=wandb_config["learning_rate"],
            weight_decay=0.01,
            warmup_steps=wandb_config["warmup_steps"],
            max_steps=wandb_config["max_steps"],
            
            # Evaluation settings
            eval_strategy="steps",  # Evaluate during training
            eval_steps=wandb_config["eval_steps"],
            
            report_to="wandb",
            lr_scheduler_type=wandb_config["lr_scheduler"],
            
            # Performance options
            disable_tqdm=False,
            
            # Gradient and optimizer settings
            gradient_checkpointing=True,
            
            # Seed for reproducibility
            seed=42
        ),
        callbacks=[training_speed_tracker]
    )
    
    # Train on this dataset
    print(f"Starting Unsloth-accelerated training for {dataset_name}...")
    trainer.train()
    
    # Save model for this dataset
    model_name = "final" if dataset_name == "AQuA" else dataset_name
    model_save_path = f"{output_dir}/{model_name}"
    trainer.model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    print(f"Unsloth-optimized model for {dataset_name} saved to {model_save_path}")
    
    # Log model to wandb
    artifact = wandb.Artifact(f"mistral-math-unsloth-{model_name}", type="model")
    artifact.add_dir(model_save_path)
    dataset_run.log_artifact(artifact)
    
    # Sample generation to test the model
    print(f"\nGenerating sample output for {dataset_name}...")
    
    # Get a sample question from the test dataset
    if len(test_dataset) > 0:
        test_sample = test_dataset[0]["prompt"]
        input_ids = tokenizer(test_sample, return_tensors="pt").input_ids.to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_length=512,
                temperature=0.7,
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Prompt: {test_sample}")
        print(f"Generated: {generated_text}")
        
        # Log the generated text to wandb
        wandb.log({"example_generation": wandb.Html(f"<p><strong>Prompt:</strong> {test_sample}</p><p><strong>Generated:</strong> {generated_text}</p>")})
    
    # Log training performance metrics
    if torch.cuda.is_available():
        gpu_stats = torch.cuda.memory_stats()
        wandb.log({
            "gpu_allocated_memory_gb": torch.cuda.memory_allocated() / 1e9,
            "gpu_reserved_memory_gb": torch.cuda.memory_reserved() / 1e9,
            "gpu_max_allocated_memory_gb": gpu_stats.get("allocated_bytes.all.peak", 0) / 1e9,
        })
    
    # Finish the dataset-specific wandb run
    wandb.finish()

# Log the final model (after all curriculum steps)
print("\nCurriculum learning complete!")
final_model_path = f"./models/mistral-7b-curriculum-unsloth/final"
os.makedirs(final_model_path, exist_ok=True)
model.save_pretrained(final_model_path)
tokenizer.save_pretrained(final_model_path)

# Log final model to main wandb run
final_artifact = wandb.Artifact("mistral-math-curriculum-final", type="model")
final_artifact.add_dir(final_model_path)
run.log_artifact(final_artifact)

# Finish the main wandb run
wandb.finish()