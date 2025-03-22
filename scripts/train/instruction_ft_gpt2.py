"""
Instruction fine-tuning for a pre-trained GPT-2 math model using MathInstruct dataset.
Loads the base model from wandb artifacts.
Allows filtering by source, limiting sample count, and uses streaming for efficiency.
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
from collections import Counter

# Define this filter function outside main() so it can be pickled properly
def exclude_sources_by_prefix(example, prefixes_to_exclude):
    """Filter function to exclude sources starting with specified prefixes."""
    source = example["source"]
    for prefix in prefixes_to_exclude:
        if source.startswith(prefix):
            return False
    return True

parent_dir = os.path.abspath(os.path.join(os.getcwd(), '../..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from utils.helper import get_device
except ImportError:
    # Fallback implementation if the helper module is not available
    def get_device():
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

def main():
    # create wandb config to log parameter
    config = {
        "model_name": "gpt2-math-model",  # The artifact name in wandb
        "output_model_name": "gpt2-math-instruct",  # Name for the fine-tuned model
        "dataset": "TIGER-Lab/MathInstruct",
        "streaming": True,
        "shuffle_buffer": 10000,  # Buffer size for better mixing
        "max_length": 512,
        "max_steps": 50000,            
        "learning_rate": 2e-5,         
        "batch_size": 8,              
        "gradient_accumulation_steps": 8,
        "num_workers": 4,              
        "prefetch_factor": 2,          
        "max_samples": 50000,          # Maximum number of samples to use
        "sources_to_exclude": ["data/PoT/"],      # Use just the prefix to exclude all PoT sources
        "wandb_artifact_path": "master_thesis_math_lm/gpt2-math/gpt2-math-model:v0",  # Path to the artifact
    }

    # Set the output directories
    output_dir = "./models/mathgpt2sft/"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("./logs", exist_ok=True)
    
    # Initialize wandb
    run = wandb.init(
        project="math-instruct", 
        name=config['output_model_name'],
        config=config
    )

    # Check for available hardware
    device = get_device()
    print(f"Using device: {device}")
    
    # Load model from wandb artifact
    print(f"Loading pre-trained model from wandb artifact: {config['wandb_artifact_path']}")
    artifact = run.use_artifact(config['wandb_artifact_path'])
    model_dir = artifact.download()
    
    # Load tokenizer and model from the downloaded directory
    print(f"Loading model and tokenizer from {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    
    # Set padding token directly
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id
    
    # Load MathInstruct dataset in streaming mode
    print("Loading MathInstruct dataset in streaming mode...")
    dataset = load_dataset("TIGER-Lab/MathInstruct", streaming=config["streaming"])
    
    # For streaming mode, collect source information from a small sample
    if config["streaming"]:
        # Sample some examples to analyze sources
        sample_size = 1000
        source_sample = []
        sample_iter = iter(dataset["train"])
        for _ in range(sample_size):
            try:
                example = next(sample_iter)
                source_sample.append(example["source"])
            except StopIteration:
                break
        
        source_counts = Counter(source_sample)
        print(f"\nSources in sample of {len(source_sample)} examples:")
    else:
        # If not streaming, analyze the full dataset
        source_counts = Counter(dataset["train"]["source"])
        print("\nDistinct sources in the dataset:")
    
    # Print source distribution
    for source, count in source_counts.items():
        print(f"- {source}: {count} examples")
    
    # Log source distribution to wandb
    wandb.log({"source_distribution": wandb.Table(
        columns=["Source", "Count"], 
        data=[[source, count] for source, count in source_counts.items()]
    )})
    
    # Apply source filtering if needed
    train_dataset = dataset["train"]
    
    if config["sources_to_exclude"]:
        print(f"Filtering to exclude sources starting with: {config['sources_to_exclude']}")
        # Define a proper function to use with filter (avoid using lambda)
        def filter_function(example):
            return exclude_sources_by_prefix(example, config["sources_to_exclude"])
        
        train_dataset = train_dataset.filter(filter_function)
    
    # Shuffle the dataset using buffer
    shuffle_buffer_size = config['shuffle_buffer']
    print(f"Setting up streaming pipeline with shuffle buffer size: {shuffle_buffer_size}")
    train_dataset = train_dataset.shuffle(buffer_size=shuffle_buffer_size)
    
    # Limit to max_samples if specified
    if config["max_samples"]:
        print(f"Taking first {config['max_samples']} examples after shuffling")
        train_dataset = train_dataset.take(config["max_samples"])
    
    # Prepare the prompt template function
    def prepare_instruction_data(examples):
        """Format the instruction data in a format suitable for GPT-2"""
        
        # Create formatted instruction texts
        formatted_texts = []
        
        for i in range(len(examples["instruction"])):
            instruction = examples["instruction"][i]
            output = examples["output"][i]
            
            # Format with Instruction and Response prefixes
            formatted_text = f"Instruction\n{instruction}\nResponse\n{output}"
            formatted_texts.append(formatted_text)
        
        return {"formatted_text": formatted_texts}
    
    # Apply the formatting
    processed_dataset = train_dataset.map(
        prepare_instruction_data,
        batched=True,
        remove_columns=["instruction", "output", "source"]
    )
    
    # Print a sample of the formatted data
    sample_iter = iter(processed_dataset)
    sample_example = next(sample_iter)
    print(f"\nSample formatted instruction data:\n{sample_example['formatted_text']}")
    
    # Tokenize the formatted text
    def tokenize_function(examples):
        return tokenizer(
            examples["formatted_text"],
            truncation=True,
            max_length=config['max_length'],
            padding="max_length",
            return_tensors="pt"
        )
    
    tokenized_dataset = processed_dataset.map(
        tokenize_function,
        batched=True,
        batch_size=64,
        remove_columns=["formatted_text"]
    )
    
    # Create data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # GPT-2 uses causal language modeling, not masked
    )
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,                          # Directory where model checkpoints and logs will be saved
        overwrite_output_dir=True,                      # If output_dir exists, overwrite instead of erroring
        per_device_train_batch_size=config["batch_size"], # Number of samples processed per GPU during training
        gradient_accumulation_steps=config["gradient_accumulation_steps"], # Number of forward passes before updating parameters
        save_steps=1000,                                # Save a checkpoint every 1000 steps
        save_total_limit=2,                             # Keep only the 2 most recent checkpoints to save disk space
        logging_steps=10,                               # Log metrics every 10 steps for more detailed monitoring
        logging_dir="./logs",                           # Directory for storing training logs
        
        # Hardware optimizations
        bf16=torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8,  # Use BF16 on Ampere or newer GPUs
        bf16_full_eval=torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8,  # Use BF16 during evaluation as well
        fp16=torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 8,  # Fallback to FP16 on older GPUs
        dataloader_num_workers=0,                       # Fix the pickling issue by disabling multiprocessing (or set to 0)
        dataloader_pin_memory=True,                     # Pin memory in CPU to accelerate CPU to GPU transfer
        learning_rate=config["learning_rate"],          # Initial learning rate for the optimizer
        weight_decay=0.01,                              # L2 regularization factor to prevent overfitting
        warmup_steps=200,                               # Gradually increase learning rate for first 200 steps for stability
        max_steps=config["max_steps"],                  # Total number of training steps
        evaluation_strategy="no",                       # Disable evaluation during training
        report_to="wandb",                              # Log metrics to Weights & Biases for visualization
        
        # Performance options
        disable_tqdm=False,                             # Show progress bar for monitoring training progress
        
        # Advanced optimization (PyTorch 2.0+)
        torch_compile=True,                             # Enable PyTorch compiler for just-in-time optimization
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset,
    )
    
    # Enable cudnn benchmark for faster training
    torch.backends.cudnn.benchmark = True
    
    # Start training
    print("Starting instruction fine-tuning...")
    trainer.train()
    
    # Save model locally and in wandb
    model_save_path = f"{output_dir}/final"
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    print(f"Model saved to {model_save_path}")
    
    # Log model to wandb
    artifact = wandb.Artifact(config['output_model_name'], type="model")
    artifact.add_dir(model_save_path)
    run.log_artifact(artifact)
    
    # Sample generation to test the model
    print("\nGenerating sample outputs...")
    test_prompts = [
        "Instruction\nSolve the equation 2x + 3 = 7\nResponse\n",
        "Instruction\nFind the derivative of f(x) = x^2 * sin(x)\nResponse\n",
        "Instruction\nCalculate the area of a circle with radius 5 cm\nResponse\n"
    ]
    
    model = model.to(device)
    for prompt in test_prompts:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_length=200,
                temperature=0.7,
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\nPrompt: {prompt}")
        print(f"Generated: {generated_text}")
        
        # Log the generated text to wandb
        wandb.log({"example_generation": wandb.Html(f"<p><strong>Prompt:</strong> {prompt}</p><p><strong>Generated:</strong> {generated_text}</p>")})
    
    # Log training performance metrics
    if torch.cuda.is_available():
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