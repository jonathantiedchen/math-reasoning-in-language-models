"""
Train Mistral 7B with curriculum learning approach using Weights & Biases pretrained model and SFTTrainer from Unsloth.
Replace the pretrained_artifact_name value with your actual W&B artifact reference
"""

import os
import torch
import wandb
import sys
import pandas as pd
from datasets import Dataset
# Import Unsloth and SFTTrainer
from unsloth import FastLanguageModel, is_bfloat16_supported
# Import SFTTrainer and TrainingArguments from transformers
from trl import SFTTrainer
from transformers import TrainingArguments
# Keep PEFT for printing trainable parameters
from peft import TaskType

parent_dir = os.path.abspath(os.path.join(os.getcwd(), '../../..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from utils.helper import get_device, TrainingSpeedCallback, WandbModelLogger
from utils.data import get_cl_learning_data, prepare_datasets_qa

# Define wandb configuration
wandb_config = {
    "model_name": "mistralai/Mistral-7B-v0.1",  # This will be replaced with W&B artifact
    "learning_rate": 2e-4,
    "batch_size": 2,                          # Changed to match your requested config
    "max_steps": 5000,
    #"num_train_epochs": 7,                    # Added per your request
    "warmup_steps": 5,                        # Updated per your request
    "save_steps": 100,
    "eval_steps": 100,
    "gradient_accumulation_steps": 4,         # Changed to match your requested config
    "lr_scheduler": "linear",                 # Changed to linear per your request
    "training_approach": "curriculum_learning",
    "datasets": ["ASDiv", "ParaMAWPS", "SVAMP", "DMath"],  # Removed AQuA
    "samples_per_dataset": 5,
    "test_size": 0.1,
    # LoRA specific parameters
    "lora_r": 16,
    "lora_alpha": 16,
    "lora_dropout": 0,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    # Model specific parameters
    "max_seq_length": 2048,
    "num_workers": 2,                         # Changed to match your requested config
    "optim": "adamw_8bit",                    # Added per your request
    "weight_decay": 0.01,
    "seed": 3407,                             # Changed per your request
}


# Check for available hardware
device = get_device()
print(f"Using device: {device}")

# Download pretrained model from Weights & Biases
# You'll need to specify the actual artifact name and version
pretrained_artifact_name = "master_thesis_math_lm/mistral-math-final/mistral-7b-math-unsloth-model:v0"
print(f"Downloading pretrained model from W&B: {pretrained_artifact_name}")

api = wandb.Api()
# Use wandb.use_artifact to download the model
model_artifact = api.artifact(pretrained_artifact_name, type="model")
model_dir = model_artifact.download()

# Load the model with Unsloth optimizations from the downloaded artifact
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_dir,  # Use the downloaded model path instead of the model name
    max_seq_length=wandb_config['max_seq_length'],
    dtype=torch.bfloat16 if is_bfloat16_supported() else torch.float16,
    load_in_4bit=True
)



# Set padding token if needed
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Store the EOS token for use in data processing
EOS_TOKEN = tokenizer.eos_token
print(f"Using EOS token: {EOS_TOKEN}")

# Load necessary data for curriculum learning
dataset_dict = get_cl_learning_data()

# Create the TrainingSpeedCallback to track training performance
training_speed_tracker = TrainingSpeedCallback()

# Iterate through datasets in curriculum order
for dataset_name, dataset_samples in dataset_dict.items():
    # Implement the naming logic: use "final" if dataset is DMath, otherwise use dataset_name
    dataset_name = "final" if dataset_name == "DMath" else dataset_name
    
    print(f"\n\n{'='*50}")
    print(f"Training on {dataset_name} dataset")
    print(f"{'='*50}")
    
    # Initialize a new wandb run for each dataset
    # Initialize wandb run for tracking overall process
    run = wandb.init(
        entity="master_thesis_math_lm",
        project="mistral-cl-final", 
        name=f'mistral-7b-cl-{dataset_name}',
        config=wandb_config,
        reinit=True
    )
    
    
    # Print trainable parameters info
    model.print_trainable_parameters()
        
    # Convert to pandas and then to Dataset
    df = pd.DataFrame(dataset_samples)
    dataset = Dataset.from_pandas(df)
    
    # Apply formatting
    dataset = dataset.map(
        prepare_datasets_qa,
        remove_columns=['question', 'answer']
    )
    
    # Split dataset
    split_dataset = dataset.shuffle(seed=wandb_config["seed"]).train_test_split(test_size=0.1, seed=wandb_config["seed"])

    # Correctly access the train and test datasets
    train_dataset = split_dataset["train"]
    test_dataset = split_dataset["test"]
    
    # Set output directory for this dataset's model
    output_dir = f"./models/mistral-7b-{dataset_name}-sft"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("./logs", exist_ok=True)
    
    # Initialize SFTTrainer for this dataset (replacing UnslothTrainer)
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset = test_dataset,
        
        dataset_text_field="prompt",  # Field containing the text in your dataset
        max_seq_length=wandb_config["max_seq_length"],
        dataset_num_proc=wandb_config["num_workers"],
        packing=False,  # Can make training 5x faster for short sequences, but disabled as per your request
        args=TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            per_device_train_batch_size=wandb_config["batch_size"],
            per_device_eval_batch_size=wandb_config["batch_size"],
            gradient_accumulation_steps=wandb_config["gradient_accumulation_steps"],
            do_eval=True,
            save_steps=wandb_config["save_steps"],
            save_total_limit=2,
            logging_steps=50,
            logging_dir="./logs",
            
            # Mixed precision settings
            bf16=is_bfloat16_supported(),
            fp16=not is_bfloat16_supported(),
            
            # Learning rate settings
            learning_rate=wandb_config["learning_rate"],
            weight_decay=wandb_config["weight_decay"],
            warmup_steps=wandb_config["warmup_steps"],
            
            # Use either max_steps or num_train_epochs
            max_steps=wandb_config["max_steps"],
            #num_train_epochs=wandb_config["num_train_epochs"],
            
            # Evaluation settings
            evaluation_strategy="steps",
            eval_steps=wandb_config["eval_steps"],
            
            # Reporting
            report_to="wandb",
            
            # Scheduler and optimizer
            lr_scheduler_type=wandb_config["lr_scheduler"],
            optim=wandb_config["optim"],
            
            # Performance options
            disable_tqdm=False,
            
            # Gradient and optimizer settings
            gradient_checkpointing=True,
            
            # Seed for reproducibility
            seed=wandb_config["seed"]
        ),
    )
    
    # Train on this dataset
    print(f"Starting SFT training for {dataset_name}...")
    trainer.train()
    
    # Save model for this dataset - using dataset_name for directory but model_name for final path
    model_save_path = f"{output_dir}/{dataset_name}"
    #trainer.model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    trainer.save_model(model_save_path)
    print(f"Model for {dataset_name} saved to {model_save_path}")
    
    # Log model to wandb - using model_name for the artifact naming
    artifact = wandb.Artifact(f"mistral-math-sft-{dataset_name}", type="model")
    artifact.add_dir(model_save_path)
    run.log_artifact(artifact)
    
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