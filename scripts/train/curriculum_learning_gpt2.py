"""
Train GPT-2 with Curriculum Learning.
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

### Rest of code will follow here: 
## Load pre-trained model from local disk or from wandb 
## Code used to save the pre-trained model:
"""
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
"""

## Load datasets ordered with increasing difficulty/ complexity
"""
Datasets which will be used, in correct order
- Original Datasets: 
    ASDiv (elementary) https://github.com/chaochun/nlu-asdiv-dataset/tree/master
    ParaMAWPS (elementary) - https://huggingface.co/datasets/Starscream-11813/ParaMAWPS
    SVAMP (elementary based on ASDiv and MAWPS but harder than ASDiv) - https://github.com/arkilpatel/SVAMP?tab=readme-ov-file  
    DMath (Middle School) - https://github.com/JiwooKimAR/dmath 
    AQUA (High School) - https://github.com/google-deepmind/AQuA
    Mathematics Dataset (High School) - https://github.com/google-deepmind/mathematics_dataset
"""

## Train GPT2 
"""
Code to train GPT2. All weights should be trained.
"""

## Save model so later use in 
