import os
from datasets import load_dataset
from pathlib import Path

# Define the target directory
target_dir = Path("pre-training")

# Create the directory if it doesn't exist
os.makedirs(target_dir, exist_ok=True)

print(f"Downloading fineweb dataset to {target_dir.absolute()}...")

# Load and save the dataset
ds = load_dataset("HuggingFaceFW/fineweb", "sample-10BT")

# Save the dataset to disk
ds.save_to_disk(target_dir / "fineweb")

print(f"Dataset successfully downloaded and saved to {target_dir / 'fineweb'}")

# Print some statistics about the dataset
print("\nDataset Statistics:")
for split in ds:
    print(f"  - {split} split: {len(ds[split])} examples")
    
# Print a sample example
print("\nSample example:")
print(ds["train"][0])