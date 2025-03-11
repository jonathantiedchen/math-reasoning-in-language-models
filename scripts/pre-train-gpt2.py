"""
Train GPT-2 on OpenWebMath dataset using streaming to avoid downloading the full dataset.
Integrates Weights & Biases (wandb) for experiment tracking.
"""

import os
import torch
import wandb
from transformers import (
    GPT2LMHeadModel, 
    GPT2TokenizerFast, 
    TrainingArguments, 
    Trainer, 
    DataCollatorForLanguageModeling
)
from datasets import load_dataset

def main():
    # Initialize wandb
    run = wandb.init(
        project="gpt2-math-reasoning", 
        name="gpt2-openwebmath-streaming",
        config={
            "model_name": "gpt2",
            "dataset": "open-web-math",
            "streaming": True,
            "shuffle_buffer": 1000,
            "max_steps": 10000,
            "learning_rate": 5e-5,
            "batch_size": 8,
            "gradient_accumulation_steps": 4
        }
    )
    
    # Check for available hardware
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple Silicon)")
    else:
        device = torch.device("cpu")
        print("Using CPU (no GPU detected)")
    
    # Load model and tokenizer
    model_name = "gpt2"  # Options: "gpt2", "gpt2-medium", etc.
    print(f"Loading pre-trained model: {model_name}")
    tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    
    # Set padding token
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id
    
    # Tokenization function
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=512,
            padding="max_length",
            return_tensors="pt"
        )
    
    # Load dataset in streaming mode
    print("Loading OpenWebMath dataset in streaming mode...")
    dataset = load_dataset("open-web-math/open-web-math", streaming=True)
    
    # Shuffle and prepare the streaming dataset
    shuffle_buffer_size = 1000
    print(f"Setting up streaming pipeline with shuffle buffer size: {shuffle_buffer_size}")
    train_dataset = dataset["train"].shuffle(buffer_size=shuffle_buffer_size)
    train_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        batch_size=16,  # Process 16 examples at a time
        remove_columns=["url", "date", "metadata", "text"]
    )
    
    # Define training arguments
    output_dir = "./models/gpt2-math-streaming"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("./logs", exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        save_steps=1000,
        save_total_limit=2,
        logging_steps=100,
        logging_dir="./logs",
        fp16=torch.cuda.is_available(),  # Use mixed precision if available
        learning_rate=5e-5,
        weight_decay=0.01,
        warmup_steps=500,
        # Streaming mode specific settings
        max_steps=10000,  # Limit training to a fixed number of steps
        evaluation_strategy="no",  # No evaluation during training for streaming
        report_to="wandb",  # Log to Weights & Biases
    )
    
    # Create data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # GPT-2 uses causal language modeling, not masked
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
    
    # Save the model
    model_save_path = f"{output_dir}/final"
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    print(f"Model saved to {model_save_path}")
    
    # Log model to wandb
    artifact = wandb.Artifact("gpt2-math-model", type="model")
    artifact.add_dir(model_save_path)
    run.log_artifact(artifact)
    
    # Sample generation to test the model
    print("\nGenerating sample output...")
    test_prompt = "The solution to the integral of x^2 is"
    input_ids = tokenizer(test_prompt, return_tensors="pt").input_ids.to(device)
    model = model.to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_length=100,
            temperature=0.7,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Prompt: {test_prompt}")
    print(f"Generated: {generated_text}")
    
    # Log the generated text to wandb
    wandb.log({"example_generation": wandb.Html(f"<p><strong>Prompt:</strong> {test_prompt}</p><p><strong>Generated:</strong> {generated_text}</p>")})
    
    # Finish the wandb run
    wandb.finish()

if __name__ == "__main__":
    main()