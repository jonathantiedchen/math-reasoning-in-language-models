"""
Simple script to check available PyTorch devices and select the best one
"""

import torch

def check_devices():
    # Check available devices and select the best one
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch, 'mps') and torch.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple Silicon GPU)")
    else:
        device = torch.device("cpu")
        print("No GPU available, using CPU")
    
    print(f"Using device: {device}")
    return device

if __name__ == "__main__":
    device = check_devices()