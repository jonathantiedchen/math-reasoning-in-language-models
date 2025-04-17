#!/usr/bin/env python3
"""
Token Count Estimator for Language Model Training

This script estimates the total number of tokens processed during training
for different model training runs in the math-reasoning-in-language-models project.

Usage:
    python token_count_estimator.py --model [gpt2|gpt2-large|mistral] --training_type [pretraining|curriculum|instruction]

The script will output the estimated token counts based on the configuration parameters 
found in your repository's training scripts.
"""

import argparse
import math
import pandas as pd
import json
from tabulate import tabulate

def calculate_tokens_processed(config, dataset_tokens=None):
    """
    Calculate the total number of tokens processed during training.
    
    Parameters:
    - config: A dictionary with training configuration.
    - dataset_tokens: Optional direct number of tokens in dataset.
    
    Returns:
    - Dictionary with token count details.
    """
    result = {}
    
    # Extract common parameters
    sequence_length = config.get("max_length", config.get("max_seq_length", 1024))
    result["sequence_length"] = sequence_length
    
    # Option 1: If we have explicit dataset size in tokens
    if dataset_tokens:
        result["dataset_tokens"] = dataset_tokens
    
    # Option 2: Calculate from samples
    else:
        samples = config.get("total_samples", 0)
        # If we have epoch-based training, estimate sample count
        if samples == 0 and "num_train_epochs" in config:
            # Estimate based on typical dataset sizes
            if "open-web-math" in config.get("dataset", ""):
                samples = config.get("num_train_epochs", 1) * 250000  # Estimate for OpenWebMath dataset
            elif "gsm8k" in config.get("dataset", ""):
                samples = config.get("num_train_epochs", 1) * 7500  # Estimate for GSM8K dataset
            elif "curriculum_learning" in config.get("training_approach", ""):
                # Curriculum learning across multiple datasets
                samples = config.get("num_train_epochs", 1) * 5000  # Estimate for curriculum datasets
            else:
                # Default fallback
                samples = config.get("num_train_epochs", 1) * 50000
        elif samples == 0 and "max_steps" in config:
            # Calculate from steps
            batch_size = config.get("batch_size", 4) * config.get("gradient_accumulation_steps", 1)
            samples = config.get("max_steps", 1000) * batch_size
        
        result["samples"] = samples
        result["dataset_tokens"] = samples * sequence_length
    
    # Training setup
    batch_size = config.get("batch_size", 4)
    grad_accum_steps = config.get("gradient_accumulation_steps", 1)
    
    # Calculate effective batch size
    effective_batch_size = batch_size * grad_accum_steps
    result["batch_size"] = batch_size
    result["gradient_accumulation"] = grad_accum_steps
    result["effective_batch_size"] = effective_batch_size
    
    # Training duration - epoch based or step based
    if "num_train_epochs" in config:
        epochs = config.get("num_train_epochs", 1)
        # Calculate steps as well
        steps = math.ceil(result["samples"] / effective_batch_size) * epochs
        result["num_epochs"] = epochs
    else:
        steps = config.get("max_steps", 1000)
        # Calculate equivalent epochs
        if "samples" in result and result["samples"] > 0:
            epochs = steps * effective_batch_size / result["samples"]
            result["equivalent_epochs"] = epochs
    
    result["training_steps"] = steps
    
    # Total tokens processed
    # For each step, we process batch_size sequences of sequence_length tokens
    total_tokens = steps * effective_batch_size * sequence_length
    result["total_tokens_processed"] = total_tokens
    
    # Convert to more readable formats
    result["total_tokens_millions"] = total_tokens / 1e6
    result["total_tokens_billions"] = total_tokens / 1e9
    
    return result

def get_pretraining_config(model):
    """Return pretraining configuration for the specified model."""
    if model == "gpt2":
        return {
            "model_name": "gpt2",
            "dataset": "open-web-math",
            "max_length": 1024,
            "total_samples": 500000,
            "num_train_epochs": 6,
            "batch_size": 64,
            "gradient_accumulation_steps": 2,
        }
    elif model == "gpt2-large":
        return {
            "model_name": "gpt2-large",
            "dataset": "open-web-math",
            "max_length": 1024,
            "total_samples": 80000,
            "num_train_epochs": 6,
            "batch_size": 4,
            "gradient_accumulation_steps": 4,
        }
    elif model == "mistral":
        return {
            "model_name": "unsloth/mistral-7b-bnb-4bit",
            "dataset": "open-web-math",
            "max_seq_length": 2048,
            "total_samples": 20000,
            "num_train_epochs": 6,
            "batch_size": 4,
            "gradient_accumulation_steps": 4,
        }
    else:
        raise ValueError(f"Unknown model for pretraining: {model}")

def get_curriculum_config(model):
    """Return curriculum learning configuration for the specified model."""
    if model == "gpt2":
        # Separate configs for each dataset in curriculum 
        datasets = ["ASDiv", "ParaMAWPS", "DMath"]
        configs = []
        
        base_config = {
            "model_name": "gpt2",
            "max_length": 512,
            "batch_size": 8,
            "gradient_accumulation_steps": 8,
            "max_steps": 5000,
            "training_approach": "curriculum_learning",
        }
        
        # Estimated sample sizes for each dataset
        sample_sizes = {
            "ASDiv": 3000,
            "ParaMAWPS": 4000,
            "DMath": 5000
        }
        
        for dataset in datasets:
            config = base_config.copy()
            config["dataset"] = dataset
            config["total_samples"] = sample_sizes[dataset]
            configs.append(config)
            
        return configs
    
    elif model == "gpt2-large":
        # Similar to GPT-2 but with LoRA
        datasets = ["ASDiv", "ParaMAWPS", "DMath"]
        configs = []
        
        base_config = {
            "model_name": "gpt2-large",
            "max_length": 512,
            "batch_size": 8,
            "gradient_accumulation_steps": 8,
            "max_steps": 5000,
            "training_approach": "curriculum_learning_lora",
        }
        
        # Estimated sample sizes for each dataset
        sample_sizes = {
            "ASDiv": 3000,
            "ParaMAWPS": 4000,
            "DMath": 5000
        }
        
        for dataset in datasets:
            config = base_config.copy()
            config["dataset"] = dataset
            config["total_samples"] = sample_sizes[dataset]
            configs.append(config)
            
        return configs
    
    elif model == "mistral":
        # Mistral curriculum learning
        datasets = ["ASDiv", "ParaMAWPS", "DMath"]
        configs = []
        
        base_config = {
            "model_name": "mistralai/Mistral-7B-v0.1",
            "max_seq_length": 2048,
            "batch_size": 2,
            "gradient_accumulation_steps": 4,
            "max_steps": 5000,
            "training_approach": "curriculum_learning",
        }
        
        # Estimated sample sizes for each dataset
        sample_sizes = {
            "ASDiv": 3000,
            "ParaMAWPS": 4000,
            "DMath": 5000
        }
        
        for dataset in datasets:
            config = base_config.copy()
            config["dataset"] = dataset
            config["total_samples"] = sample_sizes[dataset]
            configs.append(config)
            
        return configs
    
    else:
        raise ValueError(f"Unknown model for curriculum learning: {model}")

def get_instruction_config(model):
    """Return instruction fine-tuning configuration for the specified model."""
    if model == "gpt2":
        return {
            "model_name": "gpt2",
            "dataset": "TIGER-Lab/MathInstruct",
            "max_length": 1024,
            "batch_size": 16,
            "gradient_accumulation_steps": 4,
            "max_steps": 20000,
            # MathInstruct dataset size after filtering (approximate)
            "total_samples": 80000,
        }
    elif model == "gpt2-large":
        return {
            "model_name": "gpt2-large",
            "dataset": "TIGER-Lab/MathInstruct",
            "max_length": 1024,
            "batch_size": 8,
            "gradient_accumulation_steps": 4,
            "max_steps": 10000,
            # MathInstruct dataset size after filtering (approximate)
            "total_samples": 80000,
        }
    elif model == "mistral":
        return {
            "model_name": "mistralai/Mistral-7B-v0.1",
            "dataset": "TIGER-Lab/MathInstruct",
            "max_seq_length": 1024,
            "batch_size": 2,
            "gradient_accumulation_steps": 4,
            "max_steps": 10000,
            # MathInstruct dataset size after filtering (approximate)
            "total_samples": 80000,
        }
    else:
        raise ValueError(f"Unknown model for instruction fine-tuning: {model}")

def format_report_for_single_config(config_name, token_stats):
    """Format the token statistics into a nice report string."""
    report = [f"Token Count Estimate for {config_name}"]
    report.append("=" * 50)
    report.append(f"Sequence Length: {token_stats.get('sequence_length', 'N/A')} tokens")
    
    if "samples" in token_stats:
        report.append(f"Dataset Samples: {token_stats['samples']:,}")
    
    report.append(f"Dataset Size: {token_stats.get('dataset_tokens', 'N/A'):,} tokens")
    
    report.append(f"Batch Size: {token_stats.get('batch_size', 'N/A')}")
    report.append(f"Gradient Accumulation Steps: {token_stats.get('gradient_accumulation', 'N/A')}")
    report.append(f"Effective Batch Size: {token_stats.get('effective_batch_size', 'N/A')}")
    
    if "num_epochs" in token_stats:
        report.append(f"Training Epochs: {token_stats['num_epochs']}")
    elif "equivalent_epochs" in token_stats:
        report.append(f"Equivalent Epochs: {token_stats['equivalent_epochs']:.2f}")
    
    report.append(f"Training Steps: {token_stats.get('training_steps', 'N/A'):,}")
    
    report.append(f"Total Tokens Processed: {token_stats.get('total_tokens_processed', 'N/A'):,}")
    report.append(f"Total Tokens (Millions): {token_stats.get('total_tokens_millions', 'N/A'):.2f}M")
    report.append(f"Total Tokens (Billions): {token_stats.get('total_tokens_billions', 'N/A'):.4f}B")
    
    return "\n".join(report)

def make_report_table(all_stats):
    """Create a tabular report of token statistics for all configurations."""
    rows = []
    
    for config_name, token_stats in all_stats.items():
        row = {
            "Configuration": config_name,
            "Sequence Length": token_stats.get("sequence_length", "N/A"),
            "Dataset Size (tokens)": f"{token_stats.get('dataset_tokens', 0) / 1e6:.2f}M",
            "Batch Size": token_stats.get("effective_batch_size", "N/A"),
            "Training Steps": f"{token_stats.get('training_steps', 0):,}",
            "Total Tokens": f"{token_stats.get('total_tokens_billions', 0):.4f}B",
        }
        rows.append(row)
    
    return pd.DataFrame(rows)

def main():
    parser = argparse.ArgumentParser(description="Estimate token counts for LM training")
    parser.add_argument("--model", choices=["gpt2", "gpt2-large", "mistral"], required=True,
                        help="Model to estimate tokens for")
    parser.add_argument("--training_type", choices=["pretraining", "curriculum", "instruction", "all"], 
                        default="all", help="Type of training to estimate tokens for")
    parser.add_argument("--output", choices=["text", "json", "table"], default="table",
                        help="Output format")
    parser.add_argument("--output_file", type=str, default=None,
                        help="File to save output to")
    
    args = parser.parse_args()
    
    training_types = [args.training_type] if args.training_type != "all" else ["pretraining", "curriculum", "instruction"]
    
    # Collect all configurations to evaluate
    all_stats = {}
    
    for training_type in training_types:
        if training_type == "pretraining":
            config = get_pretraining_config(args.model)
            stats = calculate_tokens_processed(config)
            all_stats[f"{args.model} Pretraining"] = stats
            
        elif training_type == "curriculum":
            configs = get_curriculum_config(args.model)
            curriculum_total = {
                "sequence_length": 0,
                "samples": 0,
                "dataset_tokens": 0,
                "training_steps": 0,
                "total_tokens_processed": 0,
                "total_tokens_millions": 0,
                "total_tokens_billions": 0
            }
            
            # Process each dataset in the curriculum
            for i, config in enumerate(configs):
                dataset_name = config.get("dataset", f"Dataset {i+1}")
                stats = calculate_tokens_processed(config)
                all_stats[f"{args.model} Curriculum - {dataset_name}"] = stats
                
                # Accumulate totals
                curriculum_total["sequence_length"] = max(curriculum_total["sequence_length"], 
                                                       stats.get("sequence_length", 0))
                curriculum_total["samples"] += stats.get("samples", 0)
                curriculum_total["dataset_tokens"] += stats.get("dataset_tokens", 0)
                curriculum_total["training_steps"] += stats.get("training_steps", 0)
                curriculum_total["total_tokens_processed"] += stats.get("total_tokens_processed", 0)
                curriculum_total["total_tokens_millions"] += stats.get("total_tokens_millions", 0)
                curriculum_total["total_tokens_billions"] += stats.get("total_tokens_billions", 0)
            
            # Add curriculum totals
            all_stats[f"{args.model} Curriculum - TOTAL"] = curriculum_total
            
        elif training_type == "instruction":
            config = get_instruction_config(args.model)
            stats = calculate_tokens_processed(config)
            all_stats[f"{args.model} Instruction Fine-tuning"] = stats
    
    # Create grand total for the full training pipeline
    if args.training_type == "all":
        pipeline_total = {
            "sequence_length": 0,
            "samples": 0,
            "dataset_tokens": 0,
            "training_steps": 0,
            "total_tokens_processed": 0,
            "total_tokens_millions": 0,
            "total_tokens_billions": 0
        }
        
        for config_name, stats in all_stats.items():
            if "TOTAL" not in config_name:  # Don't double-count curriculum totals
                pipeline_total["sequence_length"] = max(pipeline_total["sequence_length"], 
                                                     stats.get("sequence_length", 0))
                pipeline_total["samples"] += stats.get("samples", 0)
                pipeline_total["dataset_tokens"] += stats.get("dataset_tokens", 0)
                pipeline_total["training_steps"] += stats.get("training_steps", 0)
                pipeline_total["total_tokens_processed"] += stats.get("total_tokens_processed", 0)
                pipeline_total["total_tokens_millions"] += stats.get("total_tokens_millions", 0)
                pipeline_total["total_tokens_billions"] += stats.get("total_tokens_billions", 0)
        
        # Add pipeline totals
        all_stats[f"{args.model} FULL PIPELINE"] = pipeline_total
    
    # Generate output
    if args.output == "text":
        output = []
        for config_name, stats in all_stats.items():
            output.append(format_report_for_single_config(config_name, stats))
            output.append("\n")
        output_content = "\n".join(output)
    
    elif args.output == "json":
        output_content = json.dumps(all_stats, indent=2)
    
    elif args.output == "table":
        df = make_report_table(all_stats)
        output_content = tabulate(df, headers="keys", tablefmt="pretty", showindex=False)
    
    # Output result
    if args.output_file:
        with open(args.output_file, "w") as f:
            f.write(output_content)
        print(f"Output written to {args.output_file}")
    else:
        print(output_content)
        
if __name__ == "__main__":
    main()