#!/bin/bash

# Setup script for Stock Prediction project
# This script sets up the Python environment and installs dependencies

echo "================================================"
echo "Stock Prediction Project - Setup Script"
echo "================================================"
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1)
echo "Found: $python_version"
echo ""

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv
echo "✓ Virtual environment created"
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated"
echo ""

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip --quiet
echo "✓ pip upgraded"
echo ""

# Install requirements
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt
echo "✓ Dependencies installed"
echo ""

# Download NLTK data
echo "Downloading NLTK data for sentiment analysis..."
python -c "import nltk; nltk.download('vader_lexicon', quiet=True)"
echo "✓ NLTK data downloaded"
echo ""

# Create data directory if it doesn't exist
echo "Setting up data directory..."
mkdir -p data
echo "✓ Data directory ready"
echo ""

echo "================================================"
echo "Setup Complete!"
echo "================================================"
echo ""
echo "To activate the virtual environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "To start Jupyter Notebook, run:"
echo "  jupyter notebook notebooks/01_baseline_model.ipynb"
echo ""
echo "To run tests, run:"
echo "  pytest tests/"
echo ""
echo "Happy predicting! 📈"
