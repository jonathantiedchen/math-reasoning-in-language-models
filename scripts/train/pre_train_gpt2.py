"""
Train GPT-2 on OpenWebMath dataset using streaming to avoid downloading the full dataset.
Integrates Weights & Biases (wandb) for tracking.
"""

import os
import torch
import wandb
import sys
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer, 
    DataCollatorForLanguageModeling
)
from datasets import load_dataset

parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from utils.helper import get_device

def main():

    # create wandb config to log parameter
    config={
            "model_name": "gpt2",  # Options: "gpt2", "gpt2-medium", etc.
            "dataset": "open-web-math",
            "streaming": True,
            "shuffle_buffer": 1000,
            "max_length": 1024,
            "max_steps": 10000,
            "learning_rate": 5e-5,
            "batch_size": 8,
            "gradient_accumulation_steps": 4
    }

    # Set the output directories
    output_dir = "./models/gpt2-math-streaming"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("./logs", exist_ok=True)
    
    # Initialize wandb
    run = wandb.init(
        project="gpt2-math-reasoning", 
        name="gpt2-openwebmath-streaming",
        config=config
    )

    # Check for available hardware
    device = get_device()
    
    # Load model and tokenizer
    model_name = config['model_name']
    print(f"Loading pre-trained model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Set padding token
    tokenizer.pad_token = tokenizer.eos_token # End-of-Sequence (EOS) token as the padding token for the tokenizer.
    model.config.pad_token_id = model.config.eos_token_id # model to recognize the EOS token ID as the padding token ID.
    
    
    # Load dataset in streaming mode
    print("Loading OpenWebMath dataset in streaming mode...")
    dataset = load_dataset("open-web-math/open-web-math", streaming=True)
    
    # Shuffle the dataset
    shuffle_buffer_size = config['shuffle_buffer']
    print(f"Setting up streaming pipeline with shuffle buffer size: {shuffle_buffer_size}")

    # defining train dataset
    train_dataset = dataset["train"].shuffle(buffer_size=shuffle_buffer_size)

    # Define custom Tokenization function
    def tokenize_function(examples, config=config):
        return tokenizer(
            examples["text"], # Processes the "text" field from each example in the batch
            truncation=True, # Cuts off texts that are longer than the maximum allowed length
            max_length=config['max_length'], #  Sets the maximum sequence length from the config
            padding="max_length", # Pads all sequences to exactly the maximum length
            return_tensors="pt" # Returns PyTorch tensors (pt) instead of lists
        )
    
    # applies a transformation function to all examples in the dataset
    train_dataset = train_dataset.map(
        tokenize_function, # converts text into token IDs that the model can understand
        batched=True, # enables batch processing of examples
        batch_size=16,  # Process 16 examples at a time
        remove_columns=["url", "date", "metadata", "text"] # after tokenization, the original columns are removed from the dataset
    )
    
    # Create data collator for language modeling - forms batches from the training dataset
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # GPT-2 uses causal language modeling, not masked
    )
    
    # define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir, # directory where the model predictions and checkpoints will be writtendirectory where the model predictions and checkpoints will be written
        overwrite_output_dir=True, # overwrite the content of the output directory
        per_device_train_batch_size=8, # The batch size per GPU/CPU for training
        gradient_accumulation_steps=4, # Gradients from multiple mini-batches (4 in this case) are accumulated before performing a parameter update
        save_steps=1000, # Save a checkpoint every 1,000 training steps
        save_total_limit=2, # Keep only the 2 most recent checkpoint files to save disk space
        logging_steps=100, # Log training metrics every 100 steps
        logging_dir="./logs", # Directory where training logs will be written
        fp16=torch.cuda.is_available(),  # Use mixed precision training if a GPU is available
        learning_rate=config["learning_rate"], # initial learning rate for the optimizer 
        weight_decay=0.01, # L2 regularization strength to prevent overfitting
        warmup_steps=500, # Gradually increase the learning rate from 0 to the specified learning rate over the first 500 steps -> stabilizes early training
        # Streaming mode specific settings
        max_steps=10000,  # Limit training to a fixed number of steps
        evaluation_strategy="no",  # No evaluation during training for streaming
        report_to="wandb",  # Log to Weights & Biases
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )
    
    # Start training
    print("Starting training with streaming dataset...")
    trainer.train()
    

    ## Save model locally and in wandb
    # Save the model locally
    model_save_path = f"{output_dir}/final"
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path) # storing all the tokenizer's configuration and vocabulary files in the same directory as your mode
    print(f"Model saved to {model_save_path}")
    
    # Log model to wandb
    artifact = wandb.Artifact("gpt2-math-model", type="model")
    artifact.add_dir(model_save_path)
    run.log_artifact(artifact)
    

    ### ONLY Sample Generation
    # Sample generation to test the model
    print("\nGenerating sample output...")
    test_prompt = "The solution to the integral of x^2 is"
    input_ids = tokenizer(test_prompt, return_tensors="pt").input_ids.to(device)
    model = model.to(device)
    

    with torch.no_grad(): # no_ disables gradient calculations - use model for inference
        outputs = model.generate(
            input_ids,
            max_length=100,
            temperature=0.7,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True) # skip_special_tokens to ignore padding tokens in text generation
    print(f"Prompt: {test_prompt}")
    print(f"Generated: {generated_text}")
    
    # Log the generated text to wandb
    wandb.log({"example_generation": wandb.Html(f"<p><strong>Prompt:</strong> {test_prompt}</p><p><strong>Generated:</strong> {generated_text}</p>")})
    
    # Finish the wandb run
    wandb.finish()

if __name__ == "__main__":
    main()