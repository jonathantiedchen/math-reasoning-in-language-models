import requests
import os
import json
import logging
import random
from datasets import load_dataset, interleave_datasets, concatenate_datasets, load_from_disk, Dataset as HFDataset
import xml.etree.ElementTree as ET
import sys
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer, 
    DataCollatorForLanguageModeling
)


#################################################################
################# GSM8K Download ##########################
#################################################################
# Function to download GSM8K dataset from GitHub
def download_gsm8k():
    
    # Create data directory if it doesn't exist
    os.makedirs("gsm8k_data", exist_ok=True)
    
    # URLs for train and test data
    train_url = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl"
    test_url = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
    
    # Download train set
    train_response = requests.get(train_url)
    with open("gsm8k_data/train.jsonl", "wb") as f:
        f.write(train_response.content)
    
    # Download test set
    test_response = requests.get(test_url)
    with open("gsm8k_data/test.jsonl", "wb") as f:
        f.write(test_response.content)
    
    print("GSM8K dataset downloaded successfully.")


# Function to load the GSM8K dataset from files
def load_gsm8k_from_file():
    train_data = []
    test_data = []
    
    # Check if files exist, download if not
    if not (os.path.exists("gsm8k_data/train.jsonl") and os.path.exists("gsm8k_data/test.jsonl")):
        download_gsm8k()
    
    # Load train data
    with open("gsm8k_data/train.jsonl", "r") as f:
        for line in f:
            train_data.append(json.loads(line))
    
    # Load test data
    with open("gsm8k_data/test.jsonl", "r") as f:
        for line in f:
            test_data.append(json.loads(line))
    
    print(f"Loaded {len(train_data)} training examples and {len(test_data)} test examples")
    return train_data, test_data


####################################################
############# Pre-Training ##########################
####################################################

def get_mixed_dataset(config, tokenizer):
    """
    Creates a combined dataset with samples from OpenWebMath and FineWeb based on config parameters.
    """
    # Calculate number of samples from each dataset
    openwebmath_samples = int(config['total_samples'] * config['openwebmath_ratio'])
    fineweb_samples = config['total_samples'] - openwebmath_samples
    
    print(f"Loading datasets in streaming mode...")
    print(f"OpenWebMath samples: {openwebmath_samples} ({config['openwebmath_ratio']*100}%)")
    print(f"FineWeb samples: {fineweb_samples} ({config['fineweb_ratio']*100}%)")
    
    # Load OpenWebMath dataset
    openwebmath_dataset = load_dataset(config['openwebmath_dataset'], streaming=True)["train"]
    
    # Load and filter FineWeb dataset by token count
    print(f"Filtering FineWeb samples to include only those with token_count <= {config['max_length']}")
    fineweb_dataset = load_dataset(
        config['fineweb_dataset'],
        name=config['fineweb_subset'],
        split="train",
        streaming=True
    )
    
    # Filter based on token_count column
    def token_count_filter(example):
        # The token_count column provides the exact token count
        return example["token_count"] <= config['max_length']
    
    fineweb_filtered = fineweb_dataset.filter(token_count_filter)
    
    # Define tokenization function (shared for both datasets)
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=config['max_length'],
            padding="max_length",
            return_tensors="pt"
        )
    
    # Process OpenWebMath dataset
    # We need to ensure the columns match after processing
    openwebmath_processed = openwebmath_dataset.map(
        tokenize_function,
        batched=True,
        batch_size=64,
        remove_columns=openwebmath_dataset.column_names
    )
    
    # Process FineWeb dataset
    # Note: Check the actual column names in your FineWeb dataset and adjust accordingly
    fineweb_processed = fineweb_dataset.map(
        tokenize_function,
        batched=True,
        batch_size=64,
        remove_columns=fineweb_dataset.column_names  # Remove all original columns
    )
    
    # Create subsets with the desired number of samples
    # For IterableDatasets, we'll take the first N samples
    openwebmath_subset = openwebmath_processed.take(openwebmath_samples)
    fineweb_subset = fineweb_processed.take(fineweb_samples)
    
    # Check that the datasets have matching column formats
    print(f"OpenWebMath columns: {openwebmath_subset.column_names}")
    print(f"FineWeb columns: {fineweb_subset.column_names}")
    
    # Combine the datasets using interleave_datasets
    # This function allows us to specify the mixing rates
    combined_dataset = interleave_datasets(
        [openwebmath_subset, fineweb_subset],
        probabilities=[config['openwebmath_ratio'], config['fineweb_ratio']],
        seed=42,
        stopping_strategy="first_exhausted"
    )
    
    # Shuffle the combined dataset
    shuffle_buffer_size = config['shuffle_buffer']
    print(f"Setting up streaming pipeline with shuffle buffer size: {shuffle_buffer_size}")
    train_dataset = combined_dataset.shuffle(buffer_size=shuffle_buffer_size, seed=42)
    
    return train_dataset

def get_local_data(config):
    """
    Load and mix OpenWebMath and FineWeb datasets from local disk or HuggingFace hub.
    
    Args:
        config: Dictionary containing configuration parameters:
            - test_sample_size: Number of samples to use if in testing mode
            - use_local_data: Whether to use local data or download from HF
            - testing_mode: Whether we're in testing mode (using fewer samples)
            - openwebmath_ratio: Ratio of samples from OpenWebMath (default 0.7)
            - fineweb_ratio: Ratio of samples from FineWeb (default 0.3)
            - total_samples: Total number of samples to use from both datasets combined
    
    Returns:
        A dataset with mixed samples from both sources
    """
    total_samples = config.get('total_samples', 500000)
    openwebmath_ratio = config.get('openwebmath_ratio', 0.7)
    fineweb_ratio = config.get('fineweb_ratio', 0.3)
    
    # Calculate samples for each dataset
    openwebmath_samples = int(total_samples * openwebmath_ratio)
    fineweb_samples = total_samples - openwebmath_samples
    
    print(f"Loading mixed dataset with:")
    print(f" - OpenWebMath: {openwebmath_samples} samples ({openwebmath_ratio*100:.1f}%)")
    print(f" - FineWeb: {fineweb_samples} samples ({fineweb_ratio*100:.1f}%)")
    
    # Testing mode uses fewer samples
    if config.get('testing_mode', False):
        print(f"Testing mode enabled, only using {config['test_sample_size']} samples total")
        test_ratio = config['test_sample_size'] / total_samples
        openwebmath_samples = int(openwebmath_samples * test_ratio)
        fineweb_samples = int(fineweb_samples * test_ratio)
        
    # Load OpenWebMath dataset
    if config.get('use_local_data', False):
        # Load OpenWebMath from local directory
        openwebmath_path = "math-reasoning-in-language-models/data/pre-training/open-web-math"
        print(f"Loading OpenWebMath from local path: {openwebmath_path}")
        try:
            full_openwebmath = load_from_disk(openwebmath_path)
            openwebmath_dataset = full_openwebmath["train"].select(range(openwebmath_samples))
        except Exception as e:
            print(f"Error loading local OpenWebMath dataset: {e}")
            print("Falling back to loading from HuggingFace hub...")
            openwebmath_dataset = load_dataset(
                "open-web-math/open-web-math",
                split=f"train[:{openwebmath_samples}]"
            )
        
        # Load FineWeb from local directory
        fineweb_path = "math-reasoning-in-language-models/data/pre-training/fineweb"
        print(f"Loading FineWeb from local path: {fineweb_path}")
        try:
            full_fineweb = load_from_disk(fineweb_path)
            fineweb_dataset = full_fineweb["train"].select(range(fineweb_samples))
        except Exception as e:
            print(f"Error loading local FineWeb dataset: {e}")
            print("Falling back to loading from HuggingFace hub...")
            fineweb_dataset = load_dataset(
                "HuggingFaceFW/fineweb",
                name=config.get('fineweb_subset', 'CC-MAIN-2024-10'),
                split=f"train[:{fineweb_samples}]"
            )
    else:
        # Load both datasets from HuggingFace hub
        print("Loading datasets from HuggingFace hub")
        openwebmath_dataset = load_dataset(
            "open-web-math/open-web-math",
            split=f"train[:{openwebmath_samples}]"
        )
        
        fineweb_dataset = load_dataset(
            "HuggingFaceFW/fineweb",
            name=config.get('fineweb_subset', 'CC-MAIN-2024-10'),
            split=f"train[:{fineweb_samples}]"
        )
    
    # Ensure datasets have compatible column formats
    print(f"OpenWebMath columns: {openwebmath_dataset.column_names}")
    print(f"FineWeb columns: {fineweb_dataset.column_names}")
    
    # Make sure both datasets have a 'text' column
    # Adapt as needed based on actual dataset structure
    if 'text' not in fineweb_dataset.column_names and 'content' in fineweb_dataset.column_names:
        fineweb_dataset = fineweb_dataset.rename_column('content', 'text')
    
    # Combine datasets using interleave_datasets for proper mixing
    combined_dataset = interleave_datasets(
        [openwebmath_dataset, fineweb_dataset],
        probabilities=[openwebmath_ratio, fineweb_ratio],
        seed=42,
        stopping_strategy="first_exhausted"
    )
    
    print(f"Created interleaved dataset with appropriate mixing ratios")
    return combined_dataset

#######################################
######## CURRICULUM LEARNING #############
#######################################
def prepare_datasets_qa(example):
    # Simple Question-Answer format
    example['prompt'] = f"###Question: {example['question']}\n###Answer: {example['answer']}"
    return example

# Data loading functions for each dataset format
def load_asdiv_data(file_path):
    """Load and parse ASDiv XML data"""
    tree = ET.parse(file_path)
    root = tree.getroot()
    problems = []
    
    for problem in root.findall(".//Problem"):
        body_elem = problem.find("Body")
        question_elem = problem.find("Question")
        answer_elem = problem.find("Answer")
        formula_elem = problem.find("Formula")
        
        if question_elem is not None and answer_elem is not None:
            body = body_elem.text.strip() if body_elem is not None else ""
            question = question_elem.text.strip()
            answer = answer_elem.text.strip()
            formula = formula_elem.text.strip() if formula_elem is not None else ""
            
            # Format the text
            if body and formula:
                text = f"Question: {body} {question}\nSolution: {formula}\nAnswer: {answer}"
            elif formula:
                text = f"Question: {question}\nSolution: {formula}\nAnswer: {answer}"
            elif body:
                text = f"Question: {body} {question}\nAnswer: {answer}"
            else:
                text = f"Question: {question}\nAnswer: {answer}"
                
            problems.append({"text": text})
    
    print(f"Loaded {len(problems)} problems from ASDiv")
    return problems

def load_paramawps_data(file_path):
    """Load and parse ParaMAWPS JSON data"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    problems = []
    for item in data:
        question = item.get("original_text", "").strip()
        equation = item.get("equation", "").strip()
        answer = str(item.get("ans", "")).strip()  # Changed from "answer" to "ans"
        
        text = f"Question: {question}\nEquation: {equation}\nAnswer: {answer}"
        problems.append({"text": text})
    
    print(f"Loaded {len(problems)} problems from ParaMAWPS")
    return problems

def load_svamp_data(file_path):
    """Load and parse SVAMP JSON data"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    problems = []
    for item in data:
        question = item.get("Body", "") + " " + item.get("Question", "")
        question = question.strip()
        equation = str(item.get("Equation", "")).strip()
        answer = str(item.get("Answer", "")).strip()
        
        # Include equation if available
        if equation:
            text = f"Question: {question}\nEquation: {equation}\nAnswer: {answer}"
        else:
            text = f"Question: {question}\nAnswer: {answer}"
            
        problems.append({"text": text})
    
    print(f"Loaded {len(problems)} problems from SVAMP")
    return problems

def load_dmath_data(file_path):
    """Load and parse DMath JSON data"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    problems = []
    for item_id, item_data in data.items():
        question = item_data.get("question_en", "").strip()
        solution = item_data.get("solution_code_en", "").strip()  # Using solution_code_en
        answer = item_data.get("answer_en", "").strip()
        
        text = f"Question: {question}\nSolution: {solution}\nAnswer: {answer}"
        problems.append({"text": text})
    
    print(f"Loaded {len(problems)} problems from DMath")
    return problems

"""def load_aqua_data(file_path):
    Load and parse AQuA JSON data
    problems = []
    
    with open(file_path, 'r') as f:
        for line in f:
            item = json.loads(line.strip())
            
            question = item.get("question", "").strip()
            options = item.get("options", [])
            rationale = item.get("rationale", "").strip()
            correct = item.get("correct", "")
            
            # Format options
            options_text = ""
            for i, opt in enumerate(options):
                options_text += f"{chr(65+i)}. {opt}\n"
            
            text = f"Question: {question}\nOptions:\n{options_text}Rationale: {rationale}\nAnswer: {correct}"
            problems.append({"text": text})
    
    print(f"Loaded {len(problems)} problems from AQuA")
    return problems"""


def get_cl_learning_data():
    # Add parent directory to path for importing modules
    parent_dir = os.path.abspath(os.path.join(os.getcwd(), '../..'))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    
    # Import data loading functions
    from utils.data import load_asdiv_data, load_paramawps_data, load_svamp_data, load_dmath_data
    
    # Find data root directory
    data_root = None
    for root in [os.getcwd()] + [os.path.abspath(os.path.join(os.getcwd(), *['..'] * i)) for i in range(1, 4)]:
        if os.path.exists(os.path.join(root, "data")):
            data_root = root
            break
    
    if data_root is None:
        raise FileNotFoundError("Could not find data directory")
    
    # Unified formatting function for most datasets
    def format_with_solution(item, solution_key):
        text = item['text']
        parts = text.split('Question: ')[1].split(f'\n{solution_key}:')
        question = parts[0].strip()
        
        solution_answer_parts = parts[1].split('\nAnswer:')
        solution = solution_answer_parts[0].strip()
        answer = solution_answer_parts[1].strip()
        
        return {
            'question': question,
            'answer': f"Let me solve this step by step.\n{solution}\nTherefore, the answer is {answer}."
        }
    
    # Special formatter for AQuA
    """def format_aqua(item):
        text = item['text']
        
        # Extract question and options
        question_part = text.split('Question: ')[1].split('Rationale:')[0].strip()
        if 'Options:' in question_part:
            question_parts = question_part.split('Options:')
            question = f"{question_parts[0].strip()}\nOptions:\n{question_parts[1].strip()}"
        else:
            question = question_part
        
        # Extract rationale and answer
        rationale = text.split('Rationale:')[1].split('Answer:')[0].strip() if 'Rationale:' in text else ""
        answer = text.split('Answer:')[1].strip() if 'Answer:' in text else ""
        
        return {
            'question': question,
            'answer': f"Let me solve this step by step.\n{rationale}\nTherefore, the answer is {answer}."
        }"""
    
    # Define dataset configurations
    datasets_config = [
        {
            "name": "ASDiv",
            "path": os.path.join(data_root, "data", "curriculum_learning", "1_ASDiv", "ASDiv.xml"),
            "loader": load_asdiv_data,
            "format": lambda item: format_with_solution(item, 'Solution')
        },
        {
            "name": "ParaMAWPS",
            "path": os.path.join(data_root, "data", "curriculum_learning", "2_ParaMAWPS", "ParaMAWPS_trainset.json"),
            "loader": load_paramawps_data,
            "format": lambda item: format_with_solution(item, 'Equation')
        },
        {
            "name": "SVAMP",
            "path": os.path.join(data_root, "data", "curriculum_learning", "3_SVAMP", "SVAMP.json"),
            "loader": load_svamp_data,
            "format": lambda item: format_with_solution(item, 'Equation')
        },
        {
            "name": "DMath",
            "path": os.path.join(data_root, "data", "curriculum_learning", "4_Dmath", "dmath_train.json"),
            "loader": load_dmath_data,
            "format": lambda item: format_with_solution(item, 'Solution')
        }
    ]
    
    # Process all datasets
    standardized_datasets = {}
    total_examples = 0
    
    for dataset_config in datasets_config:
        try:
            data = dataset_config["loader"](dataset_config["path"])
            standardized_data = []
            
            for item in data:
                try:
                    formatted_item = dataset_config["format"](item)
                    standardized_data.append(formatted_item)
                except Exception:
                    continue
            
            standardized_datasets[dataset_config["name"]] = standardized_data
            total_examples += len(standardized_data)
            
        except Exception:
            standardized_datasets[dataset_config["name"]] = []
    
    return standardized_datasets
