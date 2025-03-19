import torch
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling
from datasets import Dataset, concatenate_datasets
from trl import SFTTrainer, SFTConfig
import pandas as pd
import os
import sys

parent_dir = os.path.abspath(os.path.join(os.getcwd(), '../..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from utils.helper import get_device, WandbModelLogger
from utils.data import get_cl_learning_data, prepare_datasets_qa
    

# Define wandb configuration
wandb_config = {
    "model_name": "gpt2",
    "learning_rate": 2e-5,
    "batch_size": 8,
    "max_steps": 5,
    "warmup_steps": 100,
    "save_steps": 100,
    "eval_steps": 100,
    "fp16": True,
    "gradient_accumulation_steps": 8,
    "lr_scheduler": "cosine",
    "training_approach": "curriculum_learning",
    "datasets": ["ASDiv", "ParaMAWPS", "SVAMP", "DMath", "AQuA"],
    "samples_per_dataset": 5,
    "test_size": 0.1
}

run = wandb.init(
    project="gpt2-math-test", 
    name="curriculum-learning-sft",
    config=wandb_config
)
artifact = run.use_artifact('master_thesis_math_lm/gpt2-math/gpt2-math-model:v0', type='model')
artifact_dir = artifact.download()

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(artifact_dir)
model = AutoModelForCausalLM.from_pretrained(artifact_dir)

tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = model.config.eos_token_id

# replace with get device
device = get_device()
device

dataset_dict = get_cl_learning_data()

# Now you have access to each dataset separately:
asdiv_data = dataset_dict["ASDiv"]
paramawps_data = dataset_dict["ParaMAWPS"]
svamp_data = dataset_dict["SVAMP"]
dmath_data = dataset_dict["DMath"]
aqua_data = dataset_dict["AQuA"]


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

for dataset_name, dataset_samples in dataset_dict.items():
    print(f"Training on {dataset_name} dataset")
    
    # Take only 5 samples
    #dataset_small = dataset_samples[:5]
    
    # Cut AQuA Dataset to 15000 samples
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
    
    # Manually tokenize both train and eval datasets
    tokenized_train = tokenize_datasets(train_dataset)
    tokenized_eval = tokenize_datasets(test_dataset)
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    # Using values from wandb_config
    batch_size = wandb_config["batch_size"]
    max_steps = wandb_config["max_steps"]
    
    # Create wandb callback
    wandb_callback = WandbModelLogger(
        output_dir="./models/mathgpt2sft/",
        tokenizer=tokenizer,
        save_steps=200,
        model_name_prefix="gpt2-math-sft"
    )
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=tokenized_train,  
        eval_dataset=tokenized_eval,
        args=SFTConfig(
            output_dir="./models/mathgpt2sft/",
            gradient_accumulation_steps=wandb_config["gradient_accumulation_steps"],
            #evaluation_strategy="steps",
            do_eval=True,
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
            #dataset_text_field="prompt", #performs automatic tokenization in training when specified
            lr_scheduler_type=wandb_config["lr_scheduler"],
            report_to="wandb"
        ),
        data_collator=data_collator
    )
    
    trainer.train()

    model_name = "final" if dataset_name == "AQuA" else dataset_name
    
    # Save final model
    final_model_path = f"./models/mathgpt2sft/model_{model_name}"
    trainer.save_model(final_model_path)
    
    # Log final model to wandb
    final_artifact = wandb.Artifact(f"gpt2-math-sft-{model_name}", type="model")
    final_artifact.add_dir(final_model_path)
    run.log_artifact(final_artifact)

# Finish wandb run
wandb.finish()