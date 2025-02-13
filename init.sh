#!/usr/bin/env bash
# init.sh

# Exit immediately if a command exits with a non-zero status
set -e

# Function to display info messages
function echo_info {
    echo -e "\033[1;34m[INFO]\033[0m $1"
}

function echo_error {
    echo -e "\033[1;31m[ERROR]\033[0m $1"
}

# Detect Operating System
OS="$(uname)"
echo_info "Detected Operating System: $OS"

# Function to install packages on Linux
install_linux_packages() {
    echo_info "Updating package lists..."
    sudo apt-get update

    echo_info "Installing Python3, venv, and pip..."
    sudo apt-get install -y python3 python3-venv python3-pip git
}

# Function to install packages on macOS
install_macos_packages() {
    echo_info "Updating Homebrew..."
    brew update

    echo_info "Installing Python3..."
    brew install python3

    echo_info "Installing Git..."
    brew install git
}

# Install necessary packages based on OS
if [[ "$OS" == "Linux" ]]; then
    install_linux_packages
elif [[ "$OS" == "Darwin" ]]; then
    install_macos_packages
else
    echo_error "Unsupported Operating System: $OS"
    exit 1
fi

echo_info "Checking GPU availability..."
if command -v nvidia-smi > /dev/null 2>&1; then
    echo_info "NVIDIA GPU found:"
    nvidia-smi
else
    echo_error "No GPU detected or driver missing."
fi

# Clone the repository if not already cloned
REPO_DIR="math_reasoning_language_models"  # Replace with your actual repo name
REPO_URL="https://github.com/jonathantiedchen/math_reasoning_language_models.git"  # Replace with your repo URL

if [ ! -d "$REPO_DIR" ]; then
    echo_info "Cloning repository from GitHub..."
    git clone "$REPO_URL"
else
    echo_info "Repository already cloned. Pulling latest changes..."
    cd "$REPO_DIR"
    git reset --hard HEAD
    git pull
fi

# Set up virtual environment
if [ ! -d "venv" ]; then
    echo_info "Setting up virtual environment..."
    python3 -m venv .venv
else
    echo_info "Virtual environment already exists."
fi

# Change to repository directory
# cd "$REPO_DIR"

# Activate virtual environment
echo_info "Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo_info "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo_info "Installing dependencies..."
pip install -r requirements.txt

# Ensuring reproducibility
echo_info "Setting environment variable for reproducibility..."
export TF_DETERMINISTIC_OPS=1
export TF_CUDNN_DETERMINISTIC=1

echo_info "Initialization complete."
