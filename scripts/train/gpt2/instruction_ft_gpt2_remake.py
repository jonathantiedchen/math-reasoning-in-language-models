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

# Initialize Weights & Biases with more configuration options
wandb.init(
    project="gpt2-math-instruct",
    name="first-remake-instruction-learning",
    config={
        "model_name": "master_thesis_math_lm/gpt2-math/gpt2-math-sft-final:v0",
        "dataset": "TIGER-Lab/MathInstruct",
        "batch_size": 16,  # Reduced batch size
        "gradient_accumulation_steps": 4,  # Added gradient accumulation
        "learning_rate": 5e-5,
        "epochs": 1,
        "max_steps": 10,
        "sequence_length": 1024
    }
)

# Load pre-trained model and tokenizer from Weights & Biases
model_name = "master_thesis_math_lm/gpt2-math/gpt2-math-sft-final:v0"
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
        max_length=512,  # Reduced from 1024 to save memory
        return_tensors="pt"
    )
    
    # Set up labels for language modeling (same as input_ids)
    tokenized_inputs["labels"] = tokenized_inputs["input_ids"].clone()
    
    return tokenized_inputs

# For the initial assessment, we'll take a small subset of the dataset
sample_size = 16 * 4 * 10  # Enough for 10 steps with batch size 16 and grad accumulation 4

# Select a subset of the data for faster assessment
subset = dataset.select(range(min(len(dataset), sample_size)))

# Tokenize the subset
tokenized_dataset = subset.map(
    tokenize_function,
    batched=True,
    batch_size=32,  # Process in smaller batches to avoid OOM during tokenization
    remove_columns=dataset.column_names,
    desc="Tokenizing training dataset"
)

print(f"Prepared training dataset: {len(tokenized_dataset)} examples")

# Free up some memory
torch.cuda.empty_cache()

# Data collator for language modeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # We're not doing masked language modeling
)

# Set up training arguments with memory optimizations
batch_size = 16  # Reduced batch size
training_args = TrainingArguments(
    output_dir="./models/mathgpt2instruct/",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=4,  # Accumulate gradients to simulate larger batch
    save_steps=5,
    save_total_limit=2,
    max_steps=10,
    logging_steps=1,
    learning_rate=5e-5,
    weight_decay=0.01,
    warmup_steps=0,
    report_to="wandb",
    fp16=True,  # Keep mixed precision
    # Memory optimizations
    optim="adamw_torch_fused",  # Use memory-efficient optimizer
    dataloader_pin_memory=False,  # Reduce CPU->GPU transfer overhead
    gradient_checkpointing=True,  # Trade compute for memory
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
model_save_path = "./gpt2-math-instruct"
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

# Finish the W&B run
wandb.finish()

print("Fine-tuning process completed successfully!")