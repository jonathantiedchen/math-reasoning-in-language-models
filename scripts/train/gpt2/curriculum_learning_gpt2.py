import torch
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling
from datasets import Dataset, concatenate_datasets
from trl import SFTTrainer, SFTConfig
import pandas as pd
import os
import sys

parent_dir = os.path.abspath(os.path.join(os.getcwd(), '../../..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from utils.helper import get_device, WandbModelLogger
from utils.data import get_cl_learning_data, prepare_datasets_qa
    
# Define wandb configuration
wandb_config = {
    "model_name": "gpt2",
    "learning_rate": 2e-5,
    "batch_size": 8,
    "max_steps": 10000,
    "warmup_steps": 100,
    "save_steps": 1000,
    "eval_steps": 100,
    "fp16": True,
    "gradient_accumulation_steps": 8,
    "lr_scheduler": "cosine",
    "training_approach": "curriculum_learning",
    "datasets": ["ASDiv", "ParaMAWPS", "SVAMP", "DMath"],
    "samples_per_dataset": 5,
    "test_size": 0.1
}

# Download the model only once before starting the dataset loops
api = wandb.Api()
artifact = api.artifact('master_thesis_math_lm/gpt2-math-final/gpt2-math-fineweb-model:v0', type='model')
artifact_dir = artifact.download()

# Load the tokenizer and model outside the loop
tokenizer = AutoTokenizer.from_pretrained(artifact_dir)
model = AutoModelForCausalLM.from_pretrained(artifact_dir)

tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = model.config.eos_token_id

# Get device
device = get_device()
print(f"Using device: {device}")

# Load all datasets
dataset_dict = get_cl_learning_data()

def tokenize_datasets(dataset):
    tokenized_dataset = dataset.map(
      lambda example: tokenizer(
          example['prompt'],
          truncation=True,
          max_length=512,
          ),
      batched=True,
      remove_columns=['prompt'])
    return tokenized_dataset

# Create output directory
os.makedirs("./models/mathgpt2sft/", exist_ok=True)

for dataset_name, dataset_samples in dataset_dict.items():
    # Implement the naming logic: use "final" if dataset is DMath, otherwise use dataset_name
    model_name = "final" if dataset_name == "DMath" else dataset_name
    
    # Initialize a new wandb run for each dataset
    run = wandb.init(
        entity = "master_thesis_math_lm",
        project="gpt2-cl-final", 
        name=f"curriculum-learning-sft-{dataset_name}",
        config=wandb_config,
        reinit=True  # This ensures a new run is created each time
    )
    
    print(f"Training on {dataset_name} dataset")
    
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
    
    # Log dataset size information
    wandb.log({
        "dataset_name": dataset_name,
        "train_size": len(train_dataset),
        "eval_size": len(test_dataset),
    })
    
    # Manually tokenize both train and eval datasets
    tokenized_train = tokenize_datasets(train_dataset)
    tokenized_eval = tokenize_datasets(test_dataset)
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    # Using values from wandb_config
    batch_size = wandb_config["batch_size"]
    max_steps = wandb_config["max_steps"]
    
    # Create dataset-specific output directory
    dataset_output_dir = f"./models/mathgpt2sft/{dataset_name}"
    os.makedirs(dataset_output_dir, exist_ok=True)
    
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=tokenized_train,  
        eval_dataset=tokenized_eval,
        args=SFTConfig(
            output_dir=dataset_output_dir,
            gradient_accumulation_steps=wandb_config["gradient_accumulation_steps"],
            do_eval=True,
            evaluation_strategy="steps",
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            log_level="info",
            save_strategy="steps",
            save_steps=wandb_config["save_steps"],
            save_total_limit=2,
            save_safetensors=True,
            fp16=wandb_config["fp16"],
            logging_steps=50,
            learning_rate=wandb_config["learning_rate"],
            eval_steps=wandb_config["eval_steps"],
            max_steps=max_steps,
            warmup_steps=wandb_config["warmup_steps"],
            lr_scheduler_type=wandb_config["lr_scheduler"],
            report_to="wandb"
        ),
        data_collator=data_collator
    )
    
    # Start training on this dataset
    trainer.train()

    # Save final model for this dataset
    final_model_path = f"{dataset_output_dir}/final"
    trainer.save_model(final_model_path)
    
    # Log model to wandb, using the model_name variable we defined
    final_artifact = wandb.Artifact(f"gpt2-math-sft-{model_name}", type="model")
    final_artifact.add_dir(final_model_path)
    run.log_artifact(final_artifact)
    
    # Finish this wandb run before starting the next one
    wandb.finish()

print("Curriculum learning completed for all datasets.")