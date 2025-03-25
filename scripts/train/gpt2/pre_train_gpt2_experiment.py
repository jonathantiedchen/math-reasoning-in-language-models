"""
Train GPT-2 on OpenWebMath dataset using streaming to avoid downloading the full dataset.
Integrates Weights & Biases (wandb) for tracking.
Fixed generation stopping behavior with proper EOS token handling.
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

parent_dir = os.path.abspath(os.path.join(os.getcwd(), '../../..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from utils.helper import get_device, WandbModelLogger, MemoryManagementCallback  # Import custom logger


def main():

    # create wandb config to log parameter
    config = {
            "model_name": "gpt2",  # Options: "gpt2", "gpt2-medium", etc.
            "dataset": "open-web-math",
            "streaming": True,
            "shuffle_buffer": 5000,  # Increased buffer size for better mixing
            "max_length": 1024,
            "max_steps": 50000,
            "learning_rate": 5e-5,
            "batch_size": 32,  # Increased from 8 to better utilize H100
            "gradient_accumulation_steps": 1,  
            "num_workers": 4,  # Parallel data loading
            "prefetch_factor": 4  # Prefetch factor for data loading
    }

    # Set the output directories
    output_dir = "./models/gpt2-math-streaming"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("./logs", exist_ok=True)
    
    # Initialize wandb
    run = wandb.init(
        project="gpt2-math-test", 
        name="gpt2-openwebmath-pre_training",
        config=config
    )

    # Check for available hardware
    device = get_device()
    
    # Load model and tokenizer
    model_name = config['model_name']
    print(f"Loading pre-trained model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Create a separate PAD token instead of using EOS as PAD
    # This ensures EOS tokens are properly learned during training
    print("Adding special PAD token to tokenizer")
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    # Resize model embeddings to account for the new token
    print("Resizing model embeddings to accommodate new PAD token")
    model.resize_token_embeddings(len(tokenizer))
    
    # Initialize the new token embedding to the mean of all other embeddings
    with torch.no_grad():
        word_embeddings = model.get_input_embeddings().weight
        avg_embedding = word_embeddings[:-1].mean(dim=0)
        word_embeddings[-1] = avg_embedding
    
    # Update model configuration to use the new pad token
    model.config.pad_token_id = tokenizer.pad_token_id
    
    # Ensure EOS token is properly configured
    print(f"Using distinct tokens - EOS: '{tokenizer.eos_token}' (ID: {tokenizer.eos_token_id}), PAD: '{tokenizer.pad_token}' (ID: {tokenizer.pad_token_id})")
    
    # Print token configuration for debugging
    print(f"EOS token: '{tokenizer.eos_token}', ID: {tokenizer.eos_token_id}")
    print(f"PAD token: '{tokenizer.pad_token}', ID: {tokenizer.pad_token_id}")
    print(f"Model EOS token ID: {model.config.eos_token_id}")
    print(f"Model PAD token ID: {model.config.pad_token_id}")
    
    # Load dataset in streaming mode
    print("Loading OpenWebMath dataset in streaming mode...")
    dataset = load_dataset("open-web-math/open-web-math", streaming=True)
    
    # Shuffle the dataset
    shuffle_buffer_size = config['shuffle_buffer']
    print(f"Setting up streaming pipeline with shuffle buffer size: {shuffle_buffer_size}")
    train_dataset = dataset["train"].shuffle(buffer_size=shuffle_buffer_size)

    # Tokenization function with explicit EOS token
    def tokenize_function(examples, config=config):
        # Basic tokenization with room for EOS token
        result = tokenizer(
            examples["text"],
            truncation=True,
            max_length=config['max_length'] - 1,  # Reserve space for EOS token
            padding=False,  # Let the data collator handle padding
            return_tensors=None
        )
        
        # Append EOS token to each example
        for i in range(len(result['input_ids'])):
            # Add EOS token if not already present
            if result['input_ids'][i][-1] != tokenizer.eos_token_id:
                result['input_ids'][i].append(tokenizer.eos_token_id)
                result['attention_mask'][i].append(1)  # EOS is a real token, not padding
        
        return result
    
    # Apply tokenization to the dataset
    train_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        batch_size=64,
        remove_columns=["url", "date", "metadata", "text"]
    )
    
    # Create custom data collator that properly handles EOS tokens and PAD tokens
    class DataCollatorForLanguageModelingWithEOS(DataCollatorForLanguageModeling):
        """Custom data collator that ensures EOS tokens are properly added and distinguished from PAD tokens."""
        
        def __init__(self, tokenizer, mlm=False, mlm_probability=0.15, add_eos_token=True):
            super().__init__(tokenizer=tokenizer, mlm=mlm, mlm_probability=mlm_probability)
            self.add_eos_token = add_eos_token
            
        def torch_call(self, examples):
            # Process examples before calling the parent's collator
            if self.add_eos_token:
                # Make sure each example has an EOS token
                for i, example in enumerate(examples):
                    input_ids = example['input_ids']
                    
                    # Find the last non-pad token position
                    non_pad_positions = [j for j, token in enumerate(input_ids) if token != self.tokenizer.pad_token_id]
                    if not non_pad_positions:
                        continue  # Skip empty sequences
                        
                    last_pos = non_pad_positions[-1]
                    
                    # Add EOS token after content if not already present and there's room
                    if input_ids[last_pos] != self.tokenizer.eos_token_id and last_pos < len(input_ids) - 1:
                        input_ids[last_pos + 1] = self.tokenizer.eos_token_id
            
            # Now call the parent's collator
            batch = super().torch_call(examples)
            
            # Make sure we don't mask out EOS tokens (this is critical!)
            labels = batch['labels']
            input_ids = batch['input_ids']
            
            # Unmask any EOS tokens that might have been masked
            # This ensures the model learns to predict EOS tokens
            for i in range(labels.size(0)):
                for j in range(labels.size(1)):
                    if input_ids[i, j] == self.tokenizer.eos_token_id and labels[i, j] == -100:
                        labels[i, j] = self.tokenizer.eos_token_id
            
            return batch
    
    # Use the custom data collator with proper EOS token handling
    data_collator = DataCollatorForLanguageModelingWithEOS(
        tokenizer=tokenizer,
        mlm=False,  # GPT-2 uses causal language modeling, not masked
        add_eos_token=True  # Enable EOS token insertion
    )
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=config["batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        save_steps=1000,
        save_total_limit=2,
        logging_steps=10,
        logging_dir="./logs",
        
        # H100-specific optimizations
        bf16=True,
        bf16_full_eval=True,
        dataloader_num_workers=config["num_workers"],
        dataloader_pin_memory=True,
        learning_rate=config["learning_rate"],
        weight_decay=0.01,
        warmup_steps=200,
        max_steps=config["max_steps"],
        evaluation_strategy="no",
        report_to="wandb",
        lr_scheduler_type="cosine",
        
        # Performance options
        disable_tqdm=False,
        
        # Advanced optimization (PyTorch 2.0+)
        torch_compile=True,
    )
    
    # Save model every 10000 steps
    wandb_logger = WandbModelLogger(
        output_dir=output_dir,
        tokenizer=tokenizer,
        save_steps=10000,
        model_name_prefix="gpt2-math-100000"
    )

    # Clear cache every 100 steps
    memory_manager = MemoryManagementCallback(clear_cache_steps=100)
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset
    )
    
    # Enable cudnn benchmark for faster training
    torch.backends.cudnn.benchmark = True
    
    # Start training
    print("Starting training with streaming dataset...")
    trainer.train()
    
    ## Save model locally and in wandb
    # Save the model locally
    model_save_path = f"{output_dir}/final"
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    print(f"Model saved to {model_save_path}")
    
    # Log model to wandb
    artifact = wandb.Artifact("gpt2-math-model", type="model")
    artifact.add_dir(model_save_path)
    run.log_artifact(artifact)
    
    ### Sample Generation with improved stopping behavior
    # Sample generation to test the model
    print("\nGenerating sample output...")
    test_prompt = "The solution to the integral of x^2 is"
    input_ids = tokenizer(test_prompt, return_tensors="pt").input_ids.to(device)
    model = model.to(device)
    
    # Debugging generation parameters
    print("Generation parameters:")
    print(f"- EOS token ID: {tokenizer.eos_token_id}")
    print(f"- PAD token ID: {tokenizer.pad_token_id}")
    print(f"- Model config EOS token ID: {model.config.eos_token_id}")
    print(f"- Model config PAD token ID: {model.config.pad_token_id}")
    
    # Verify that the EOS token is in the vocabulary
    if tokenizer.eos_token_id in tokenizer.get_vocab().values():
        print(f"✓ EOS token '{tokenizer.eos_token}' is in vocabulary")
    else:
        print("⚠ EOS token not found in vocabulary!")
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_length=100,
            min_length=5,  # Ensure at least some generation happens
            temperature=0.7,
            num_return_sequences=1,
            do_sample=True,  # Use sampling for more natural text
            top_p=0.95,  # Nucleus sampling
            top_k=50,  # Limit to top 50 tokens
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,  # Explicitly set EOS token
            early_stopping=True,  # Stop when EOS is generated
            no_repeat_ngram_size=3  # Prevent repetitive patterns
        )
    
    # Show generation with token IDs for debugging
    print(f"Generated token IDs: {outputs[0].tolist()}")
    print(f"Contains EOS token? {tokenizer.eos_token_id in outputs[0].tolist()}")
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Prompt: {test_prompt}")
    print(f"Generated: {generated_text}")
    
    # Test another generation with stricter stopping criteria
    print("\nGenerating another sample with stricter stopping...")
    with torch.no_grad():
        outputs2 = model.generate(
            input_ids,
            max_length=100,
            temperature=0.9,  # Slightly more randomness
            num_return_sequences=1,
            do_sample=True,
            top_p=0.92,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            early_stopping=True,
            repetition_penalty=1.2,  # Penalize repetitions more heavily
            length_penalty=1.5  # Encourage shorter generations
        )
    
    generated_text2 = tokenizer.decode(outputs2[0], skip_special_tokens=True)
    print(f"Prompt: {test_prompt}")
    print(f"Generated (stricter stopping): {generated_text2}")
    
    # Log both generated texts to wandb
    wandb.log({
        "example_generation": wandb.Html(
            f"<p><strong>Prompt:</strong> {test_prompt}</p>"
            f"<p><strong>Generated:</strong> {generated_text}</p>"
            f"<p><strong>Generated (stricter stopping):</strong> {generated_text2}</p>"
        )
    })

    # Log training performance metrics
    gpu_stats = torch.cuda.memory_stats()
    wandb.log({
        "gpu_allocated_memory_gb": torch.cuda.memory_allocated() / 1e9,
        "gpu_reserved_memory_gb": torch.cuda.memory_reserved() / 1e9,
        "gpu_max_allocated_memory_gb": gpu_stats.get("allocated_bytes.all.peak", 0) / 1e9,
    })
    
    # Finish the wandb run
    wandb.finish()

if __name__ == "__main__":
    main()