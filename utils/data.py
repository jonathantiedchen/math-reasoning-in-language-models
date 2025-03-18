import requests
import os
import json
import logging
from datasets import Dataset as HFDataset
import xml.etree.ElementTree as ET



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

# Tokenization function for HuggingFace datasets
def tokenize_function(tokenizer):
    def tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt"
        )
    return tokenize

# Function to prepare a dataset for training
def prepare_dataset(data, tokenizer):
    # Convert to HuggingFace dataset
    hf_dataset = HFDataset.from_dict({"text": [item["text"] for item in data]})
    
    # In prepare_dataset:
    tokenized_dataset = hf_dataset.map(
        tokenize_function(tokenizer),
        batched=True,
        remove_columns=["text"]
    )
    
    # Set the format for PyTorch
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    
    return tokenized_dataset