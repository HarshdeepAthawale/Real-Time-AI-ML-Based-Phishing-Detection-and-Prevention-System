#!/bin/bash
# Setup script for training environment

set -e

echo "Setting up training environment..."

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip -q

# Install dependencies
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt -q

echo ""
echo "✓ Setup complete! Activate the environment with:"
echo "  source venv/bin/activate"
echo ""
echo "Then you can run training scripts."
