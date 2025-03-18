import os
import random
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
import wandb
import sys

parent_dir = os.path.abspath(os.path.join(os.getcwd(), '../..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from utils.helper import get_device
from utils.data import load_asdiv_data, load_paramawps_data, load_svamp_data, load_aqua_data, load_dmath_data, prepare_dataset

# Define the data paths
data_paths = [
    "/Users/jonathan/Library/Mobile Documents/com~apple~CloudDocs/Master/Master Thesis/math-reasoning-in-language-models/data/curriculum_learning/1_ASDiv/ASDiv.xml",
    "/Users/jonathan/Library/Mobile Documents/com~apple~CloudDocs/Master/Master Thesis/math-reasoning-in-language-models/data/curriculum_learning/2_ParaMAWPS/ParaMAWPS_trainset.json",
    "/Users/jonathan/Library/Mobile Documents/com~apple~CloudDocs/Master/Master Thesis/math-reasoning-in-language-models/data/curriculum_learning/3_SVAMP/SVAMP.json",
    "/Users/jonathan/Library/Mobile Documents/com~apple~CloudDocs/Master/Master Thesis/math-reasoning-in-language-models/data/curriculum_learning/4_Dmath/dmath_train.json",
    "/Users/jonathan/Library/Mobile Documents/com~apple~CloudDocs/Master/Master Thesis/math-reasoning-in-language-models/data/curriculum_learning/5_AQuA/AQuA_train.json"
]

# Dataset names for logging
dataset_names = ["ASDiv", "ParaMAWPS", "SVAMP", "DMath", "AQuA"]

# Dictionary mapping dataset file paths to their respective loading functions
data_loaders = {
    data_paths[0]: load_asdiv_data,
    data_paths[1]: load_paramawps_data,
    data_paths[2]: load_svamp_data,
    data_paths[3]: load_dmath_data,
    data_paths[4]: load_aqua_data
}

def curriculum_training():
    # Define wandb config
    config = {
        "model": "gpt2",
        "base_artifact": "master_thesis_math_lm/gpt2-math/gpt2-math-model:v0",
        "learning_rate": 5e-5,
        "weight_decay": 0.01,
        "epochs_per_dataset": 2,  # Reduced epochs for testing
        "batch_size": 4,
        "gradient_accumulation_steps": 4,
        "fp16": True,
        "datasets": dataset_names,
        "curriculum_order": dataset_names,
        "seed": 42,
        "samples_per_dataset": 5,  # Only use 5 samples per dataset for testing
        "evaluation": "external_gsm8k",
        "test_mode": True  # Flag to indicate this is a test run
    }
    
    # Initialize wandb with config
    run = wandb.init(
        project="gpt-math", 
        name="gpt2-math-curr-test-run",
        config=config
    )
    
    # Download the pre-trained model
    artifact = run.use_artifact(config["base_artifact"], type='model')
    artifact_dir = artifact.download()
    
    # Load the model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(artifact_dir)
    tokenizer = AutoTokenizer.from_pretrained(artifact_dir)
    
    # Add padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Move model to GPU if available
    device = get_device()
    
    model.to(device)
    
    # Training arguments from config
    training_args = TrainingArguments(
        output_dir="./test_results",
        overwrite_output_dir=True,
        num_train_epochs=config["epochs_per_dataset"],
        per_device_train_batch_size=config["batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        save_steps=10,  # More frequent saving for testing
        save_total_limit=1,  # Save less checkpoints to save space
        logging_dir="./test_logs",
        logging_steps=5,  # More frequent logging for testing
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        fp16=config["fp16"],
        report_to="wandb",
        dataloader_num_workers=2,  # Reduced for testing
    )
    
    # Curriculum training loop
    for stage, (data_path, dataset_name) in enumerate(zip(data_paths, dataset_names)):
        print(f"Starting curriculum stage {stage+1}: {dataset_name}")
        
        # Load the dataset
        data = data_loaders[data_path](data_path)
        
        # Shuffle and limit to specified number of samples
        random.shuffle(data)
        train_data = data[:config["samples_per_dataset"]]
        
        print(f"Using {len(train_data)} samples from {dataset_name} (limited for testing)")
        
        # Prepare the dataset
        train_dataset = prepare_dataset(train_data, tokenizer)
        
        # Define the trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
        )
        
        # Log dataset information to wandb
        wandb.log({
            "current_dataset": dataset_name, 
            "dataset_size": len(train_data),
            "curriculum_stage": stage + 1
        })
        
        # Train the model
        train_result = trainer.train()
        
        # Log training stats to wandb
        wandb.log({
            f"{dataset_name}_training_loss": train_result.training_loss,
            f"{dataset_name}_train_runtime": train_result.metrics["train_runtime"],
            f"{dataset_name}_train_samples_per_second": train_result.metrics["train_samples_per_second"]
        })
        
        # Save model checkpoint after training on this dataset
        model_path = f"./test_checkpoints/gpt2-test-after-{dataset_name}"
        print(f"Saving model checkpoint after {dataset_name} to {model_path}")
        model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)
        
        # Log model checkpoint to wandb
        checkpoint_artifact = wandb.Artifact(
            f"gpt2-test-after-{dataset_name}", 
            type="model",
            description=f"Test GPT-2 Math model after training on {dataset_name}"
        )
        checkpoint_artifact.add_dir(model_path)
        run.log_artifact(checkpoint_artifact)
        
        # Log a summary of the model's training on this dataset
        wandb.run.summary[f"{dataset_name}_tested"] = True
        
        print(f"Completed test curriculum stage {stage+1}: {dataset_name}")
    
    # Save the final model
    final_model_path = "./test_checkpoints/gpt2-test-curriculum-final"
    print(f"Saving final test model to {final_model_path}")
    model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    
    # Log final model to wandb
    final_artifact = wandb.Artifact(
        "gpt2-test-curriculum-final", 
        type="model",
        description="Final test GPT-2 Math model after curriculum learning"
    )
    final_artifact.add_dir(final_model_path)
    run.log_artifact(final_artifact)
    
    # Create a summary table of model checkpoints
    checkpoint_summary = [
        {
            "Stage": i+1, 
            "Dataset": dataset_name, 
            "Samples Used": config["samples_per_dataset"],
            "Checkpoint Location": f"./test_checkpoints/gpt2-test-after-{dataset_name}"
        } 
        for i, dataset_name in enumerate(dataset_names)
    ]
    
    # Log the summary table to wandb
    checkpoint_table = wandb.Table(
        columns=["Stage", "Dataset", "Samples Used", "Checkpoint Location"],
        data=[[item["Stage"], item["Dataset"], item["Samples Used"], item["Checkpoint Location"]] 
              for item in checkpoint_summary]
    )
    wandb.log({"test_checkpoints": checkpoint_table})
    
    print("Test curriculum training completed successfully!")
    return model, tokenizer

if __name__ == "__main__":
    # Set the seed at runtime, but it will be tracked in the config
    seed = 42
    # Set random seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # Ensure output directories exist
    os.makedirs("./test_results", exist_ok=True)
    os.makedirs("./test_logs", exist_ok=True)
    os.makedirs("./test_checkpoints", exist_ok=True)
    
    # Create dataset-specific checkpoint directories
    for dataset_name in dataset_names:
        os.makedirs(f"./test_checkpoints/gpt2-test-after-{dataset_name}", exist_ok=True)
    
    print("Starting test curriculum training with 5 samples per dataset...")
    
    # Run the curriculum training
    model, tokenizer = curriculum_training()
    
    # Finish the wandb run
    wandb.finish()