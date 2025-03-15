import requests
import os
import json

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