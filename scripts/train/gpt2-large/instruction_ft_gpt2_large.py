import os
import torch
import wandb
import math
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling
)
from transformers.integrations import WandbCallback
from collections import Counter
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from trl import SFTTrainer, SFTConfig  # Import SFTTrainer and SFTConfig from trl

# Define wandb configuration
wandb_config = {
    "model_name": "master_thesis_math_lm/gpt2-large-cl-final/gpt2-large-curriculum-learning-final:v0",
    "learning_rate": 2e-5,
    "batch_size": 8,
    "max_steps": 3,
    "warmup_steps": 100,
    "save_steps": 1000,
    "eval_steps": 500,
    "fp16": True,
    "gradient_accumulation_steps": 8,
    "lr_scheduler": "cosine",
    "training_approach": "instruction-fine-tuning",
    "samples_per_dataset": 5,
    "test_size": 0.1,
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05
}

# Initialize Weights & Biases with more configuration options
run = wandb.init(
    entity="master_thesis_math_lm",
    project="gpt2-large-math-instruct",
    name="GPT-2-large-IL-chained-final",
    config=wandb_config
)

print("Loading base GPT-2 model...")
base_model_name = "gpt2-large"
base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

# GPT-2 tokenizer doesn't have a padding token by default
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    base_model.config.pad_token_id = base_model.config.eos_token_id

# Now, load the curriculum learning adapter from W&B
api = wandb.Api()
print("Downloading curriculum learning adapter from W&B...")
artifact = api.artifact('master_thesis_math_lm/gpt2-large-cl-final/gpt2-large-curriculum-learning-final:v0', type='model')
adapter_dir = artifact.download()


# Load the adapter onto the base model
print("Loading curriculum learning adapter...")
try:
    # First attempt to load the adapter as a PEFT model
    adapted_model = PeftModel.from_pretrained(
        base_model,
        adapter_dir,
        is_trainable=False  # Set to False since we'll merge it
    )
    
    print("Successfully loaded curriculum learning adapter")
    
    # Merge the adapter with the base model
    print("Merging curriculum learning adapter into base model...")
    merged_model = adapted_model.merge_and_unload()
    
    print("Successfully merged adapter with base model")
    
    # Free up memory from the original models
    del base_model
    del adapted_model
    torch.cuda.empty_cache()
    
    # Use the merged model for further training
    model = merged_model
    
except Exception as e:
    print(f"Error loading/merging adapter: {e}")
    print("Falling back to using base model only")
    break

# Ensure model is in training mode before applying new LoRA
model.train()
model.gradient_checkpointing_enable()  # Enable gradient checkpointing for memory efficiency

# Prepare model for training with new LoRA
model = prepare_model_for_kbit_training(model)

# Define NEW LoRA configuration for instruction tuning
print("Applying new LoRA configuration for instruction fine-tuning...")
lora_config = LoraConfig(
    r=wandb_config["lora_r"],
    lora_alpha=wandb_config["lora_alpha"],
    lora_dropout=wandb_config["lora_dropout"],
    bias="none",
    target_modules=["c_attn", "c_proj", "c_fc"],  # Specify target modules for GPT-2
    task_type="CAUSAL_LM"
)

# Get a fresh PEFT model with these adapters
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # This should show parameters as trainable

# Load the TIGER-Lab/MathInstruct dataset - just the train split
dataset = load_dataset("TIGER-Lab/MathInstruct", split="train")
print(f"Dataset loaded: {len(dataset)} examples")

# Print the source distribution before filtering
source_counts = Counter(dataset["source"])
print("\nSource distribution before filtering:")
for source, count in sorted(source_counts.items(), key=lambda x: x[1], reverse=True):
    print(f"{source}: {count} examples")

# Print count of examples that will be filtered out
pot_examples = [example for example in dataset if example["source"].startswith("data/PoT/")]
print(f"\nNumber of examples with source starting with 'data/PoT/': {len(pot_examples)}")

# Define sources to be filtered out (original + new ones)
sources_to_filter = [
    "data/PoT/", 
    "data/CoT/gsm_rft.json",
    "data/CoT/gsm_train.json",
    "data/CoT/aqua_rat.json"
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

# Split the filtered dataset into training and validation sets
split_dataset = filtered_dataset.shuffle(seed=42).train_test_split(
    test_size=wandb_config["test_size"],
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

# Tokenize both train and validation datasets
tokenized_train_dataset = train_dataset.map(
    tokenize_function,
    batched=True,
    batch_size=32,  # Process in smaller batches to avoid OOM during tokenization
    remove_columns=train_dataset.column_names,
    desc="Tokenizing training dataset"
)

tokenized_val_dataset = val_dataset.map(
    tokenize_function,
    batched=True,
    batch_size=32,
    remove_columns=val_dataset.column_names,
    desc="Tokenizing validation dataset"
)

print(f"\nPrepared training dataset: {len(tokenized_train_dataset)} examples")
print(f"Prepared validation dataset: {len(tokenized_val_dataset)} examples")

# Free up some memory
torch.cuda.empty_cache()

# Data collator for language modeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # We're not doing masked language modeling
)

# Create output directory
output_dir = "./models/mathgpt2instruct_lora"
os.makedirs(output_dir, exist_ok=True)

# Setup SFTConfig instead of TrainingArguments
training_args = SFTConfig(
    output_dir=output_dir,
    overwrite_output_dir=True,
    dataloader_num_workers=8,
    per_device_train_batch_size=wandb_config["batch_size"],
    per_device_eval_batch_size=wandb_config["batch_size"],
    gradient_accumulation_steps=wandb_config["gradient_accumulation_steps"],
    save_steps=wandb_config["save_steps"],
    save_total_limit=2,
    max_steps=wandb_config["max_steps"],
    logging_steps=100,
    learning_rate=wandb_config["learning_rate"],
    weight_decay=0.01,
    warmup_steps=wandb_config["warmup_steps"],
    eval_strategy="steps",
    eval_steps=wandb_config["eval_steps"],
    report_to="wandb",
    fp16=wandb_config["fp16"],
    dataloader_pin_memory=True,
    gradient_checkpointing=True,
    group_by_length=True,
    save_safetensors=True,
    lr_scheduler_type=wandb_config["lr_scheduler"],
    do_eval=True
)

# Initialize SFTTrainer REMOVING the peft_config parameter
trainer = SFTTrainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset
    # REMOVED: peft_config=lora_config - Do not pass this if model already has LoRA applied
)

# Train the model
print("Starting training...")
trainer.train()
print("Training completed!")

# First save the adapter only (small file)
adapter_save_path = "./models/gpt2-math-instruct-lora-adapter"
trainer.save_model(adapter_save_path)
tokenizer.save_pretrained(adapter_save_path)
print(f"LoRA adapter saved to {adapter_save_path}")

# Now merge the adapter with the model and save the full model
print("Merging LoRA adapter with model...")
merged_model = model.merge_and_unload()

# Save the full merged model (this will be large, ~1-3GB)
full_model_save_path = "./models/gpt2-math-instruct-full"
merged_model.save_pretrained(full_model_save_path)
tokenizer.save_pretrained(full_model_save_path)
print(f"Full merged model saved to {full_model_save_path}")

# Log both to W&B
adapter_artifact = wandb.Artifact(
    name="gpt2-large-ft-adapter-final-chained",
    type="model"
)
adapter_artifact.add_dir(adapter_save_path)
run.log_artifact(adapter_artifact)

full_model_artifact = wandb.Artifact(
    name="gpt2-large-ft-full-final-chained",
    type="model"
)
full_model_artifact.add_dir(full_model_save_path)
run.log_artifact(full_model_artifact)

# Finish the W&B run
wandb.finish()

print("Fine-tuning process with chained LoRA adapters completed successfully!")