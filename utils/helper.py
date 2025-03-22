from datasets import load_dataset
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainerCallback
import torch
import random
import re
import gc
import time
import wandb

def get_device():
    DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    return DEVICE

def load_gsm8k(config):
    print("Loading GSM8K dataset...")
    dataset = load_dataset("gsm8k", "main")
    # Get both train and test sets
    train_set = dataset["train"]
    test_set = dataset["test"]
    
    if config["num_samples"]:
        # Randomly sample a subset if num_samples is specified
        indices = np.random.choice(len(test_set), min(config["num_samples"], len(test_set)), replace=False)
        test_set = test_set.select(indices)
    
    print(f"Loaded {len(train_set)} training examples and {len(test_set)} test examples")
    return train_set, test_set

def load_model(config,DEVICE):
    print(f"Loading {config['model_name']} model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"], use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(config["model_name"])
    
    # Handle padding token if not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
    
    model.to(DEVICE)
    return model, tokenizer

def extract_answer(text, eos=None):
    """
    Extracts a numerical answer from model output text, ensuring it returns a float.
    
    Args:
        text (str): The model-generated text containing an answer
        eos (str, optional): Custom end-of-sequence marker
    
    Returns:
        float or None: The extracted numerical answer or None if not found
    """
    # Handle custom EOS marker if provided
    if eos and eos in text:
        text = text.split(eos)[0].strip()
    
    # Primary method: extract after #### delimiter
    if "####" in text:
        answer_part = text.split("####")[-1].strip()
        
        # Clean the answer part
        answer_part = clean_text_for_number_extraction(answer_part)
        
        # Extract the numerical answer
        numbers = re.findall(r'[-+]?\d*\.?\d+', answer_part)
        if numbers:
            return float(numbers[0])  # Always return as float
    
    # Secondary method: look for the default EOS marker
    if "<|endoftext|>" in text:
        parts = text.split("<|endoftext|>")
        for part in reversed(parts):  # Start from the end
            clean_part = clean_text_for_number_extraction(part)
            numbers = re.findall(r'[-+]?\d*\.?\d+', clean_part)
            if numbers:
                return float(numbers[0])  # Always return as float
    
    # Fallback: extract the last number in the full text
    cleaned_text = clean_text_for_number_extraction(text)
    numbers = re.findall(r'[-+]?\d*\.?\d+', cleaned_text)
    if numbers:
        return float(numbers[-1])  # Always return as float
    
    # If no number found
    return None

# Also update the evaluation is_correct check to:
def is_correct_check(predicted, target):
    """Safe comparison function for numerical answers"""
    if predicted is None or target is None:
        return False
        
    # Ensure both are floats
    float_predicted = float(predicted)
    float_target = float(target)
    
    # For whole numbers, check exact match
    if float_predicted == int(float_predicted) and float_target == int(float_target):
        return int(float_predicted) == int(float_target)
    else:
        # For floating point, allow small relative error
        relative_error = abs(float_predicted - float_target) / (abs(float_target) + 1e-10)
        return relative_error < 0.01  # 1% relative error tolerance

def clean_text_for_number_extraction(text):
    """
    Cleans text to prepare for number extraction.
    
    Args:
        text (str): Text to clean
    
    Returns:
        str: Cleaned text
    """
    # Remove thousand separators, currency symbols, and percentages
    text = re.sub(r'[$€£¥,]', '', text)
    # Remove units of measurement that follow numbers
    text = re.sub(r'(\d+)\s*(?:dollars|USD|euros|EUR|pounds|GBP|yen|JPY|yuan|CNY|rupees|INR|%|percent|units|kg|g|mg|m|cm|mm|km|mph|lbs|oz|inches|feet|ft|tons|hours|hrs|minutes|mins|seconds|secs)', r'\1', text)
    return text

def convert_to_number(num_str):
    """
    Converts a string to the appropriate number type (int or float).
    
    Args:
        num_str (str): String representing a number
    
    Returns:
        int or float: The converted number
    """
    try:
        # Try to convert to int first if it's a whole number
        value = float(num_str)
        if value.is_integer():
            return int(value)
        return value
    except ValueError:
        # If conversion fails, return the original string
        return num_str
    
# Create 8-shot chain-of-thought prompt
def create_cot_prompt(train_examples, n_shot=8):
    # Set random seed for reproducibility
    random.seed(42)
    
    # Sample n examples from the training set
    shot_examples = random.sample(train_examples, n_shot)
    
    # Build the few-shot prompt
    prompt = ""
    for ex in shot_examples:
        # Add the question
        prompt += f"Question: {ex['question']}\n"
        
        # Add the answer with reasoning
        # Extract the answer part, assuming it contains the reasoning steps
        prompt += f"Answer: {ex['answer']}\n\n"
    
    return prompt


def generate_answer_hf(model, tokenizer, prompt, config, DEVICE, model_type="default"):
    # Prepare inputs
    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(DEVICE)
    
    # Get input length to respect model's context window
    input_length = inputs["input_ids"].shape[1]
    
    # Set max tokens based on model type
    if config["max_context"]: 
        max_context=config["max_length"]
    elif model_type == "gpt2":
        max_context = 1024
    elif "deepseek" in model_type:
        max_context = 4096
    else:
        max_context = 2048  # Safe default
    
    max_new_tokens = min(config["max_length"], max_context - input_length)
    
    # Enable sampling for GPT-2 (to fix the warnings and improve diversity)
    generation_config = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "max_new_tokens": max_new_tokens,
        "do_sample": True,  # Enable sampling
        "temperature": config["temperature"],
        "top_p": config["top_p"],
        "num_return_sequences": 1,
        "pad_token_id": tokenizer.pad_token_id,
        "repetition_penalty": 1.2,  # Add repetition penalty
        "no_repeat_ngram_size": 3,  # Prevent repetition of 3-grams
    }

    # Generate text
    with torch.no_grad():
        output = model.generate(**generation_config)
    
    # Decode the output
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Extract response based on model type
    if model_type in ["llama", "deepseek", "qwen"]:
        # For chat models that might include prefixes
        try:
            generated_text = response.split("Answer:")[-1].strip()
        except:
            # Handle uncommon formats
            generated_text = response[len(prompt):].strip()
    else:
        # Default extraction
        generated_text = response[len(prompt):].strip()
    
    return generated_text

# Function to extract the numerical answer
def extract_answer_gsm8k(text):
    if "####" in text:
        answer_part = text.split("####")[-1].strip()
        numbers = re.findall(r'[-+]?\d*\.?\d+', answer_part)
        if numbers:
            return float(numbers[0])
    
    # Fallback: extract the last number in the full text
    numbers = re.findall(r'[-+]?\d*\.?\d+', text)
    if numbers:
        return float(numbers[-1])
    
    return None

### To store Model regularly
class WandbModelLogger(TrainerCallback):
    def __init__(self, output_dir, tokenizer, save_steps=10000, model_name_prefix="gpt2-math-model"):
        self.save_steps = save_steps
        self.output_dir = output_dir
        self.tokenizer = tokenizer
        self.model_name_prefix = model_name_prefix
    
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.save_steps == 0 and state.global_step > 0:
            # Get model from kwargs
            model = kwargs.get('model')
            
            # Save the model temporarily
            tmp_save_path = f"{self.output_dir}/step_{state.global_step}"
            model.save_pretrained(tmp_save_path)
            self.tokenizer.save_pretrained(tmp_save_path)
            
            # Log model to wandb
            artifact = wandb.Artifact(f"{self.model_name_prefix}-step-{state.global_step}", type="model")
            artifact.add_dir(tmp_save_path)
            wandb.log_artifact(artifact)
            
            print(f"Logged model at step {state.global_step} to wandb")

# To regularly empty GPU cache
class MemoryManagementCallback(TrainerCallback):
    def __init__(self, clear_cache_steps=100):
        self.clear_cache_steps = clear_cache_steps
        
    def on_step_end(self, args, state, control, **kwargs):
        # Clear cache periodically
        if state.global_step % self.clear_cache_steps == 0:
            # Force garbage collection first
            gc.collect()
            # Clear CUDA cache
            torch.cuda.empty_cache()
            print(f"Step {state.global_step}: Cleared CUDA cache")
            
    def on_save(self, args, state, control, **kwargs):
        # Always clear cache when saving
        gc.collect()
        torch.cuda.empty_cache()
        print(f"Checkpoint at step {state.global_step}: Cleared CUDA cache")

class TrainingSpeedCallback:
    """Callback to track training speed (samples/second)"""
    def __init__(self):
        self.start_time = None
        self.step_count = 0
        self.total_samples = 0
        
    def on_step_begin(self, args, state, control, **kwargs):
        if self.start_time is None:
            self.start_time = time.time()
        
    def on_step_end(self, args, state, control, **kwargs):
        self.step_count += 1
        self.total_samples += args.per_device_train_batch_size * args.gradient_accumulation_steps
        
        if self.step_count % 10 == 0:  # Log every 10 steps
            elapsed = time.time() - self.start_time
            samples_per_second = self.total_samples / elapsed
            wandb.log({
                "training_step": self.step_count,
                "samples_per_second": samples_per_second,
                "seconds_per_step": elapsed / self.step_count
            })
            print(f"Step {self.step_count}: {samples_per_second:.2f} samples/second")