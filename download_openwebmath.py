#!/usr/bin/env python3
"""
Script to download the OpenWebMath dataset and save it locally
"""

import os
import json
from datasets import load_dataset
from tqdm import tqdm
import pandas as pd

def ensure_dir_exists(directory):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

def save_dataset_to_disk(dataset, output_dir, format="parquet"):
    """
    Save the dataset to disk in the specified format
    Supported formats: parquet, json, csv
    """
    ensure_dir_exists(output_dir)
    
    # Iterate through each split in the dataset
    for split_name, split_data in dataset.items():
        split_dir = os.path.join(output_dir, split_name)
        ensure_dir_exists(split_dir)
        
        # Get the file path for the output
        if format == "parquet":
            file_path = os.path.join(split_dir, f"{split_name}.parquet")
            # Save as parquet file (efficient for large datasets)
            split_data.to_parquet(file_path)
        elif format == "json":
            file_path = os.path.join(split_dir, f"{split_name}.json")
            # Convert to pandas and save as json
            df = split_data.to_pandas()
            df.to_json(file_path, orient="records", lines=True)
        elif format == "csv":
            file_path = os.path.join(split_dir, f"{split_name}.csv")
            # Convert to pandas and save as csv
            df = split_data.to_pandas()
            df.to_csv(file_path, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        print(f"Saved {split_name} split to {file_path}")
        
        # Save a sample file with just a few examples for quick inspection
        sample_file_path = os.path.join(split_dir, f"{split_name}_sample.json")
        with open(sample_file_path, 'w') as f:
            json.dump(split_data[:5], f, indent=2)
        print(f"Saved sample of {split_name} split to {sample_file_path}")
        
        # Save metadata
        metadata_file = os.path.join(split_dir, "metadata.json")
        metadata = {
            "num_examples": len(split_data),
            "column_names": split_data.column_names,
            "features": str(split_data.features),
        }
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved metadata to {metadata_file}")

def main():
    # Define the output directory
    output_dir = "data/openwebmath"
    
    print("Downloading OpenWebMath dataset...")
    # Load the dataset
    try:
        dataset = load_dataset("open-web-math/open-web-math")
        print("Dataset downloaded successfully!")
        
        # Print dataset information
        print("\nDataset Information:")
        for split_name, split_data in dataset.items():
            print(f"Split: {split_name}, Examples: {len(split_data)}")
            print(f"Columns: {split_data.column_names}")
            print(f"First example: {split_data[0]}")
        
        # Save the dataset (in parquet format by default for efficiency)
        print("\nSaving dataset to disk...")
        save_dataset_to_disk(dataset, output_dir, format="parquet")
        
        print("\nDataset saved successfully!")
        print(f"Output directory: {output_dir}")
    except Exception as e:
        print(f"Error downloading or saving the dataset: {e}")

if __name__ == "__main__":
    main()