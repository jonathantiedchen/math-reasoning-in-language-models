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
def debug_paths():
    """
    Debug function to check paths and directory structure.
    Add this to your script and call it before trying to load datasets.
    """
    import os
    
    # Print current working directory
    cwd = os.getcwd()
    print(f"\n--- PATH DEBUGGING ---")
    print(f"Current working directory: {cwd}")
    
    # Check different possible paths for the datasets
    possible_paths = [
        # Relative to current directory
        "data/pre-training/fineweb",
        "data/pre-training/open-web-math",
        "./data/pre-training/fineweb",
        "./data/pre-training/open-web-math",
        
        # With repository name
        "math-reasoning-in-language-models/data/pre-training/fineweb",
        "math-reasoning-in-language-models/data/pre-training/open-web-math",
        "/work/math-reasoning-in-language-models/data/pre-training/fineweb",
        "/work/math-reasoning-in-language-models/data/pre-training/open-web-math",
        "work/math-reasoning-in-language-models/data/pre-training/fineweb",
        "work/math-reasoning-in-language-models/data/pre-training/open-web-math",
        # Absolute paths starting from parent directories
        os.path.join(os.path.dirname(cwd), "data/pre-training/fineweb"),
        os.path.join(os.path.dirname(cwd), "data/pre-training/open-web-math"),
        os.path.join(os.path.dirname(os.path.dirname(cwd)), "data/pre-training/fineweb"),
        os.path.join(os.path.dirname(os.path.dirname(cwd)), "data/pre-training/open-web-math"),
    ]
    
    print("\nChecking possible dataset paths:")
    for path in possible_paths:
        exists = os.path.exists(path)
        status = "✅ EXISTS" if exists else "❌ NOT FOUND"
        print(f"{status}: {path}")
    
    # List contents of the current directory to see what's actually there
    print("\nContents of current directory:")
    for item in os.listdir(cwd):
        item_path = os.path.join(cwd, item)
        item_type = "Directory" if os.path.isdir(item_path) else "File"
        print(f"{item_type}: {item}")
        
        # If it's a directory, check one level deeper for data directories
        if os.path.isdir(item_path) and (item == "data" or item == "math-reasoning-in-language-models"):
            print(f"  Contents of {item}:")
            try:
                for subitem in os.listdir(item_path):
                    print(f"  - {subitem}")
                    
                    # Look one more level if needed
                    subitem_path = os.path.join(item_path, subitem)
                    if os.path.isdir(subitem_path) and subitem == "pre-training":
                        print(f"    Contents of {item}/{subitem}:")
                        for subsubitem in os.listdir(subitem_path):
                            print(f"    - {subsubitem}")
            except PermissionError:
                print(f"  Permission denied to list contents of {item}")
    
    print("\n--- END PATH DEBUGGING ---\n")
    
    # Return paths that actually exist
    return [path for path in possible_paths if os.path.exists(path)]

def get_mixed_dataset_tokenized(config, tokenizer):
    """
    Creates a combined dataset with improved path detection logic.
    """    
    # Debug paths first to understand the directory structure
    valid_paths = debug_paths()
    
    # Extract configuration
    total_samples = config.get('total_samples', 500000)
    openwebmath_ratio = config.get('openwebmath_ratio', 0.7)
    fineweb_ratio = config.get('fineweb_ratio', 0.3)
    
    # Calculate samples for each dataset
    openwebmath_samples = int(total_samples * openwebmath_ratio)
    fineweb_samples = total_samples - openwebmath_samples
    
    print(f"Loading mixed dataset with:")
    print(f" - OpenWebMath: {openwebmath_samples} samples ({openwebmath_ratio*100:.1f}%)")
    print(f" - FineWeb: {fineweb_samples} samples ({fineweb_ratio*100:.1f}%)")
    
    # Testing mode
    if config.get('testing_mode', False):
        test_sample_size = config.get('test_sample_size', 10000)
        print(f"Testing mode enabled, only using {test_sample_size} samples total")
        test_ratio = test_sample_size / total_samples
        openwebmath_samples = int(openwebmath_samples * test_ratio)
        fineweb_samples = int(fineweb_samples * test_ratio)
    
    # Find the correct paths for local data
    openwebmath_dataset = None
    fineweb_dataset = None
    
    if config.get('use_local_data', False):
        # Find a valid path for OpenWebMath
        openwebmath_path = None
        for path in valid_paths:
            if 'open-web-math' in path or 'openwebmath' in path:
                openwebmath_path = path
                break
                
        # Find a valid path for FineWeb
        fineweb_path = None  
        for path in valid_paths:
            if 'fineweb' in path:
                fineweb_path = path
                break
        
        # Try loading OpenWebMath dataset
        if openwebmath_path:
            print(f"Found valid OpenWebMath path: {openwebmath_path}")
            try:
                full_openwebmath = load_from_disk(openwebmath_path)
                openwebmath_dataset = full_openwebmath["train"].select(range(min(openwebmath_samples, len(full_openwebmath["train"]))))
                print(f"Successfully loaded OpenWebMath dataset with {len(openwebmath_dataset)} examples")
            except Exception as e:
                print(f"Error loading OpenWebMath dataset: {e}")
                openwebmath_dataset = None
        
        # Try loading FineWeb dataset
        if fineweb_path:
            print(f"Found valid FineWeb path: {fineweb_path}")
            try:
                full_fineweb = load_from_disk(fineweb_path)
                fineweb_dataset = full_fineweb["train"].select(range(min(fineweb_samples, len(full_fineweb["train"]))))
                print(f"Successfully loaded FineWeb dataset with {len(fineweb_dataset)} examples")
            except Exception as e:
                print(f"Error loading FineWeb dataset: {e}")
                fineweb_dataset = None
    
    # Fall back to HuggingFace if local datasets couldn't be loaded
    if openwebmath_dataset is None:
        print("Loading OpenWebMath from HuggingFace hub...")
        openwebmath_dataset = load_dataset(
            "open-web-math/open-web-math",
            split=f"train[:{openwebmath_samples}]"
        )
        
    if fineweb_dataset is None:
        print("Loading FineWeb from HuggingFace hub...")
        fineweb_dataset = load_dataset(
            "HuggingFaceFW/fineweb",
            name=config.get('fineweb_subset', 'CC-MAIN-2024-10'),
            split=f"train[:{fineweb_samples}]"
        )
    
    # Ensure datasets have compatible column formats
    print(f"OpenWebMath columns: {openwebmath_dataset.column_names}")
    print(f"FineWeb columns: {fineweb_dataset.column_names}")
    
    # Make sure both datasets have a 'text' column
    if 'text' not in fineweb_dataset.column_names and 'content' in fineweb_dataset.column_names:
        fineweb_dataset = fineweb_dataset.rename_column('content', 'text')
    
    # Define tokenization function
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=config['max_length'],
            padding="max_length"
        )
    
    # Apply tokenization to both datasets
    print("Tokenizing OpenWebMath dataset...")
    tokenized_openwebmath = openwebmath_dataset.map(
        tokenize_function,
        batched=True,
        batch_size=64,
        remove_columns=openwebmath_dataset.column_names,
        desc="Tokenizing OpenWebMath"
    )
    
    print("Tokenizing FineWeb dataset...")
    tokenized_fineweb = fineweb_dataset.map(
        tokenize_function,
        batched=True,
        batch_size=64,
        remove_columns=fineweb_dataset.column_names,
        desc="Tokenizing FineWeb"
    )
    
    # Combine datasets
    print("Combining datasets...")
    combined_dataset = concatenate_datasets([
        tokenized_openwebmath,
        tokenized_fineweb
    ])
    
    # Shuffle the dataset
    print("Shuffling combined dataset...")
    shuffle_seed = config.get('seed', 42)
    shuffled_dataset = combined_dataset.shuffle(seed=shuffle_seed)
    
    print(f"Created tokenized dataset with {len(shuffled_dataset)} examples")
    print(f"Example features: {list(shuffled_dataset.features.keys())}")
    
    return shuffled_dataset

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
        answer = str(item.get("ans", "")).strip()  # Changed from "answer" to "ans"
        
        text = f"Question: {question}\nAnswer: {answer}"
        problems.append({"text": text})
    
    print(f"Loaded {len(problems)} problems from ParaMAWPS")
    return problems


"""def load_svamp_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    problems = []
    for item in data:
        question = item.get("Body", "") + " " + item.get("Question", "")
        question = question.strip()
        answer = str(item.get("Answer", "")).strip()
        
        # Include equation if available
        if equation:
            text = f"Question: {question}\nAnswer: {answer}"
        else:
            text = f"Question: {question}\nAnswer: {answer}"
            
        problems.append({"text": text})
    
    print(f"Loaded {len(problems)} problems from SVAMP")
    return problems
"""

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
    from utils.data import load_asdiv_data, load_paramawps_data, load_dmath_data
    
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
