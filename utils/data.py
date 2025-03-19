import requests
import os
import json
import logging
from datasets import Dataset as HFDataset
import xml.etree.ElementTree as ET
import sys


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


######## CURRICULUM LEARNING #############
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

def load_aqua_data(file_path):
    """Load and parse AQuA JSON data"""
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
    return problems


def get_cl_learning_data():
    # Add parent directory to path for importing modules
    parent_dir = os.path.abspath(os.path.join(os.getcwd(), '../..'))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    
    # Import data loading functions
    from utils.data import load_asdiv_data, load_paramawps_data, load_svamp_data, load_aqua_data, load_dmath_data
    
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
    def format_aqua(item):
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
        }
    
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
        },
        {
            "name": "AQuA",
            "path": os.path.join(data_root, "data", "curriculum_learning", "5_AQuA", "AQuA_train.json"),
            "loader": load_aqua_data,
            "format": format_aqua
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
