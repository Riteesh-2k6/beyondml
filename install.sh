#!/usr/bin/env bash
set -e

echo -e "\033[1;36m==================================\033[0m"
echo -e "\033[1;36m     Installing BeyondML...       \033[0m"
echo -e "\033[1;36m==================================\033[0m"

# Check for Git
if ! command -v git &> /dev/null; then
    echo "\033[1;31mGit is required to install BeyondML. Please install Git first.\033[0m"
    exit 1
fi

# Check for Python
if ! command -v python3 &> /dev/null; then
    echo "\033[1;31mPython3 is required to install BeyondML. Please install Python first.\033[0m"
    exit 1
fi

INSTALL_DIR="$HOME/BeyondML"

# Clone or pull
if [ ! -d "$INSTALL_DIR" ]; then
    echo "Cloning repository into $INSTALL_DIR..."
    git clone https://github.com/Riteesh-2k6/beyondml.git "$INSTALL_DIR"
else
    echo "Directory $INSTALL_DIR already exists, pulling latest changes..."
    cd "$INSTALL_DIR"
    git pull
fi

cd "$INSTALL_DIR"

# Create venv
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

echo "Activating virtual environment and installing dependencies..."
source .venv/bin/activate
pip install --upgrade pip
pip install -e .

echo -e "\n\033[1;32m✅ BeyondML installed successfully!\033[0m"
echo -e "\033[1;33mTo run the application, just copy and paste these lines:\033[0m"
echo "cd ~/BeyondML"
echo "source .venv/bin/activate"
echo "beyondml run"
