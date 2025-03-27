import os
import torch
import wandb
import math
from datasets import load_dataset
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    integrations
)
from transformers.integrations import WandbCallback
from collections import Counter

# Initialize Weights & Biases with more configuration options
wandb.init(
    entity="master_thesis_math_lm",
    project="gpt2-math-instruct",
    name="instruction-learning-new",
    config={
        "model_name": "master_thesis_math_lm/gpt2-math/gpt2-math-sft-final:v1",
        "dataset": "TIGER-Lab/MathInstruct",
        "batch_size": 32,  # Reduced batch size
        "gradient_accumulation_steps": 4,  # Added gradient accumulation
        "learning_rate": 5e-5,
        "epochs": 1,
        "max_steps": 20000,
        "num_workers": 8,
    }
)

# Load pre-trained model and tokenizer from Weights & Biases
model_name = "master_thesis_math_lm/gpt2-math/gpt2-math-sft-final:v1"
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

# Filter out examples where source starts with "data/PoT/"
def filter_pot_sources(example):
    return not example["source"].startswith("data/PoT/")

# Apply the filter to the dataset
filtered_dataset = dataset.filter(filter_pot_sources)
print(f"\nFiltered dataset: {len(filtered_dataset)} examples (removed {len(dataset) - len(filtered_dataset)} PoT examples)")

# Print the source distribution after filtering
filtered_source_counts = Counter(filtered_dataset["source"])
print("\nSource distribution after filtering:")
for source, count in sorted(filtered_source_counts.items(), key=lambda x: x[1], reverse=True):
    print(f"{source}: {count} examples")

# Log source distribution to W&B
wandb.log({"source_distribution_before": source_counts})
wandb.log({"source_distribution_after": filtered_source_counts})

# Function to tokenize and prepare the dataset
def tokenize_function(examples):
    # Concatenate instructions and outputs with appropriate formatting
    texts = []
    for instruction, output in zip(examples["instruction"], examples["output"]):
        # Format: Instruction: [instruction] Output: [output]
        formatted_text = f"Instruction: {instruction}\nOutput: {output}"
        texts.append(formatted_text)
    
    # Tokenize with padding but use a smaller max_length
    tokenized_inputs = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=1024,  # Reduced from 1024 to save memory
        return_tensors="pt"
    )
    
    # Set up labels for language modeling (same as input_ids)
    tokenized_inputs["labels"] = tokenized_inputs["input_ids"].clone()
    
    return tokenized_inputs

# Tokenize the FILTERED dataset - this is the important fix
tokenized_dataset = filtered_dataset.map(
    tokenize_function,
    batched=True,
    batch_size=32,  # Process in smaller batches to avoid OOM during tokenization
    remove_columns=filtered_dataset.column_names,
    desc="Tokenizing training dataset"
)

print(f"\nPrepared training dataset: {len(tokenized_dataset)} examples")

# Free up some memory
torch.cuda.empty_cache()

# Data collator for language modeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # We're not doing masked language modeling
)

# Set up training arguments with memory optimizations
training_args = TrainingArguments(
    output_dir="./models/mathgpt2instruct",
    overwrite_output_dir=True,
    num_train_epochs=1,
    dataloader_num_workers=8,
    per_device_train_batch_size=32,
    gradient_accumulation_steps=4,  # Accumulate gradients to simulate larger batch
    save_steps=1000,
    save_total_limit=2,
    max_steps=20000,
    logging_steps=100,
    learning_rate=5e-5,
    weight_decay=0.01,
    warmup_steps=100,
    report_to="wandb",
    fp16=True,  # Keep mixed precision
    optim="adamw_torch_fused",  # Use memory-efficient optimizer
    dataloader_pin_memory=True,  # Reduce CPU->GPU transfer overhead
    gradient_checkpointing=True,  # Trade compute for memory
    group_by_length=True,  # Reduce padding by grouping similar lengths
)

# Initialize the Trainer with W&B callback for enhanced logging
wandb_callback = WandbCallback()
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset,
    callbacks=[wandb_callback],
)

# Train the model
print("Starting training...")
trainer.train()
print("Training completed!")

# Save the fine-tuned model
model_save_path = "./models/gpt2-math-instruct"
trainer.save_model(model_save_path)
tokenizer.save_pretrained(model_save_path)
print(f"Model saved to {model_save_path}")

# Log the model to Weights & Biases
artifact = wandb.Artifact(
    name="gpt2-math-model-finetuned",
    type="model"
)
artifact.add_dir(model_save_path)
wandb.log_artifact(artifact)
#run.log_artifact(artifact)

# Finish the W&B run
wandb.finish()

print("Fine-tuning process completed successfully!")