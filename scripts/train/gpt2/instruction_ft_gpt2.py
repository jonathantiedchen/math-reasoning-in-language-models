import os
import torch
import wandb
from datasets import load_dataset, Dataset
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    DataCollatorForLanguageModeling
)
from trl import SFTTrainer, SFTConfig
from collections import Counter

# Initialize Weights & Biases with more configuration options
wandb.init(
    entity="master_thesis_math_lm",
    project="gpt2-math-instruct",
    name="GPT-2-small-IL-final02",
    config={
        "model_name": "master_thesis_math_lm/gpt2-cl-final/gpt2-math-cl-final:v0",
        "dataset": "TIGER-Lab/MathInstruct",
        "batch_size": 16,
        "gradient_accumulation_steps": 4,
        "learning_rate": 5e-5,
        "epochs": 1,
        "max_steps": 20000,
        "num_workers": 8,
        "test_size": 0.1,
        "lr_scheduler": "cosine",
        "warmup_steps": 100
    }
)

# Load pre-trained model and tokenizer from Weights & Biases
model_name = "master_thesis_math_lm/gpt2-cl-final/gpt2-math-cl-final:v0"
# First, download the model from W&B
wandb_artifact = wandb.use_artifact(model_name)
model_dir = wandb_artifact.download()
# Load the model and tokenizer from the downloaded directory
tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
model = GPT2LMHeadModel.from_pretrained(model_dir)

# GPT-2 tokenizer doesn't have a padding token by default
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

# Load the TIGER-Lab/MathInstruct dataset - just the train split
dataset = load_dataset("TIGER-Lab/MathInstruct", split="train")
print(f"Dataset loaded: {len(dataset)} examples")

# Print the source distribution before filtering
source_counts = Counter(dataset["source"])
print("\nSource distribution before filtering:")
for source, count in sorted(source_counts.items(), key=lambda x: x[1], reverse=True):
    print(f"{source}: {count} examples")

# Print count of examples that will be filtered out
pot_examples = [example for example in dataset["source"] if example.startswith("data/PoT/")]
print(f"\nNumber of examples with source starting with 'data/PoT/': {len(pot_examples)}")

# Define sources to be filtered out (original + new ones)
sources_to_filter = [
    "data/PoT/",  # Original filter (filter anything starting with "data/PoT/")
    "data/CoT/gsm_rft.json",
    "data/CoT/gsm_train.json",
    "data/CoT/aqua_rat.json",
]

# Modified filter function to exclude all specified sources
def filter_sources(example):
    # Check if the source starts with "data/PoT/" or exactly matches any of the other sources to filter
    for source in sources_to_filter:
        if source.endswith("/"):
            # For paths ending with "/", check if the source starts with this path
            if example["source"].startswith(source):
                return False
        else:
            # For specific files, check for exact match
            if example["source"] == source:
                return False
    return True

# Apply the filter to the dataset
filtered_dataset = dataset.filter(filter_sources)
print(f"\nFiltered dataset: {len(filtered_dataset)} examples (removed {len(dataset) - len(filtered_dataset)} examples)")

# Print the source distribution after filtering
filtered_source_counts = Counter(filtered_dataset["source"])
print("\nSource distribution after filtering:")
for source, count in sorted(filtered_source_counts.items(), key=lambda x: x[1], reverse=True):
    print(f"{source}: {count} examples")

# Log source distribution to W&B
wandb.log({"source_distribution_before": source_counts})
wandb.log({"source_distribution_after": filtered_source_counts})

# Prepare data for SFTTrainer
def prepare_dataset_for_sft(dataset):
    # Format each example as needed for SFTTrainer
    formatted_data = []
    for example in dataset:
        formatted_text = f"Instruction: {example['instruction']}\nOutput: {example['output']}"
        formatted_data.append({"prompt": formatted_text})
    return formatted_data

# Format the filtered dataset
formatted_data = prepare_dataset_for_sft(filtered_dataset)
formatted_dataset = Dataset.from_dict({
    "prompt": [item["prompt"] for item in formatted_data]
})

# Split the filtered dataset into training and validation sets
split_dataset = formatted_dataset.shuffle(seed=42).train_test_split(
    test_size=wandb.config["test_size"],
    seed=42
)

train_dataset = split_dataset["train"]
val_dataset = split_dataset["test"]  # Note: "test" is the key for validation set in datasets library

# Log dataset sizes
print(f"\nSplit dataset into {len(train_dataset)} training examples and {len(val_dataset)} validation examples")
wandb.log({
    "train_size": len(train_dataset),
    "eval_size": len(val_dataset),
})

# Tokenizer function for preprocessing datasets
def tokenize_datasets(dataset):
    tokenized_dataset = dataset.map(
        lambda example: tokenizer(
            example['prompt'],
            truncation=True,
            max_length=1024,
        ),
        batched=True,
        remove_columns=['prompt']
    )
    return tokenized_dataset

# Tokenize train and validation datasets
tokenized_train = tokenize_datasets(train_dataset)
tokenized_val = tokenize_datasets(val_dataset)

print(f"\nPrepared training dataset: {len(tokenized_train)} examples")
print(f"Prepared validation dataset: {len(tokenized_val)} examples")

# Free up some memory
torch.cuda.empty_cache()

# Data collator for language modeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # We're not doing masked language modeling
)

# Create output directory
output_dir = "./models/mathgpt2instruct"
os.makedirs(output_dir, exist_ok=True)

# Initialize the SFTTrainer
print("Initializing SFTTrainer...")
trainer = SFTTrainer(
    model=model,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    args=SFTConfig(
        output_dir=output_dir,
        gradient_accumulation_steps=wandb.config["gradient_accumulation_steps"],
        do_eval=True,
        evaluation_strategy="steps",
        per_device_train_batch_size=wandb.config["batch_size"],
        per_device_eval_batch_size=wandb.config["batch_size"],
        log_level="info",
        save_strategy="steps",
        save_steps=1000,
        save_total_limit=2,
        save_safetensors=True,
        fp16=True,
        logging_steps=100,
        learning_rate=wandb.config["learning_rate"],
        eval_steps=500,
        max_steps=wandb.config["max_steps"],
        warmup_steps=wandb.config["warmup_steps"],
        lr_scheduler_type="cosine",
        report_to="wandb"
    ),
    data_collator=data_collator
)

# Train the model
print("Starting training...")
trainer.train()
print("Training completed!")

# Save the fine-tuned model
model_save_path = f"{output_dir}/final"
os.makedirs(model_save_path, exist_ok=True)
trainer.save_model(model_save_path)
tokenizer.save_pretrained(model_save_path)
print(f"Model saved to {model_save_path}")

# Log the model to Weights & Biases
artifact = wandb.Artifact(
    name="gpt2-small-ft-final",
    type="model"
)
artifact.add_dir(model_save_path)
wandb.log_artifact(artifact)

# Finish the W&B run
wandb.finish()

print("Fine-tuning process completed successfully!")