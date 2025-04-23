# Mathematical Capabilities of Language Models

This repository contains the code, experiments, and analysis for my Master's thesis exploring the mathematical capabilities of language models. The project investigates how different techniques and training approaches can optimize mathematical reasoning in both small (GPT-2) and large (Mistral-7B) language models.

## 🔍 Project Overview

This research investigates how language models can be optimized for mathematical reasoning through a progressive training strategy that combines:

1. **Pre-training** on mathematics-focused corpora
2. **Curriculum learning** across increasingly complex mathematical datasets
3. **Instruction fine-tuning** for better alignment with mathematical prompts

The project compares different model architectures (GPT-2, GPT-2 Large, Mistral-7B) and evaluates their performance on mathematical reasoning benchmarks.

## 📊 Results

The research demonstrates:
- The effectiveness of multi-stage training for mathematical reasoning
- Performance differences between model sizes and architectures
- Tradeoffs between computational resources and mathematical capabilities

Evaluation results on GSM8K and other benchmarks are available in the `scripts/benchmarking/eval_results/` directory.
Cudos to https://github.com/tianlwang/eval_gsm8k/tree/main for publishing code on how to benchmark language models on GSM8K, such transpacency is musch appreciated.

## 🛠️ Repository Structure

- **scripts/**
  - **benchmarking/**: Evaluation scripts and benchmark result analysis
  - **gpu_experiment/**: GPU optimization experiments for training
  - **token_estimation/**: Scripts to estimate token usage during training
  - **train/**: Training scripts for different models and approaches
    - **gpt2/**: GPT-2 training scripts
    - **gpt2-large/**: GPT-2 Large training scripts
    - **mistral/**: Mistral-7B training scripts
- **utils/**: Utility functions for data processing, model helpers, and evaluation
- **data/**: Data processing and dataset preparation (fetch data locally)

## 🔬 Training Methodology

The project employed a three-stage training approach:

### 1. Pre-training
Models were further pre-trained on mathematical content from OpenWebMath and FineWeb datasets to develop foundational mathematical knowledge.

### 2. Curriculum Learning
Models progressed through increasingly complex mathematical datasets:
- ASDiv (Arithmetic)
- ParaMAWPS (Word problems)
- DMath (Advanced mathematical problems)

### 3. Instruction Tuning
Final fine-tuning on the MathInstruct dataset to align models with instruction-following capabilities for mathematical problems.

## 📋 Training Details

All training was conducted using:
- Weights & Biases for experiment tracking
- PyTorch and Hugging Face Transformers
- LoRA for parameter-efficient fine-tuning (for larger models)
- GPU optimizations for efficient training

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- CUDA-compatible GPU (recommended)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/jonathantiedchen/math-reasoning-in-language-models.git
cd math-reasoning-in-language-models
```

2. Initialize the environment:
```bash
source init.sh
```

3. Set up Weights & Biases:
```bash
wandb login
```

### Running Experiments

#### Pre-training
```bash
python scripts/train/gpt2/pre_train_gpt2.py
# or
python scripts/train/mistral/pre_train_mistral_unsloth.py
```

#### Curriculum Learning
```bash
python scripts/train/gpt2/curriculum_learning_gpt2.py
# or
python scripts/train/mistral/curriculum_learning_Mistral_7B.py
```

#### Instruction Fine-tuning
```bash
python scripts/train/gpt2/instruction_ft_gpt2.py
# or
python scripts/train/mistral/instruction_ft_Mistral.py
```

#### Evaluation
```bash
python scripts/benchmarking/eval_gsm8k_zero_shot.py --model [model_name_or_path]
# or
python scripts/benchmarking/eval_gsm8k_few_shot.py --model [model_name_or_path]
```

## 📊 Experiment Tracking

All experiments are tracked using Weights & Biases. Models, metrics, and training logs are stored and can be accessed through the W&B dashboard.

## 🔄 GPU Optimization Studies

The repository includes GPU optimization experiments that test different training configurations:
- Gradient accumulation
- Gradient checkpointing
- Mixed precision training
- Memory-efficient optimizers
- Data loading optimizations

Results are available in the `scripts/gpu_experiment/` directory.

## 📚 References

- OpenWebMath dataset: [open-web-math/open-web-math](https://huggingface.co/datasets/open-web-math/open-web-math)
- GSM8K: [gsm8k](https://huggingface.co/datasets/gsm8k)
- MathInstruct: [TIGER-Lab/MathInstruct](https://huggingface.co/datasets/TIGER-Lab/MathInstruct)
- Unsloth: [unsloth](https://github.com/unslothai/unsloth)

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📧 Contact

Jonathan Tiedchen - [GitHub](https://github.com/jonathantiedchen)
