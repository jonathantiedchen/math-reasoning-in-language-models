"""
Train GPT-2 with Instruction Fine-Tuning.
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

### Code will follow here: 
## WandB config file
## Laod Curriculum Learning Fine-Tuned Model
## Load Dataset for Instruction Finetuning
"""
    MathInstruct - needs to filter out gsm8k https://huggingface.co/datasets/TIGER-Lab/MathInstruct 
    OpenMathInstruct-2 - uses only GSM8k and MATH https://huggingface.co/datasets/nvidia/OpenMathInstruct-2
    MathOdyssey - https://huggingface.co/datasets/mengfn/MathOdyssey
"""
## Train Model with Instruction Finetuning
## Log with WandB
## Save model locally and in Wandb 