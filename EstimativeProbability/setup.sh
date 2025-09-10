#!/bin/bash

# Setup script for EstimativeProbability project

echo "Setting up EstimativeProbability project..."

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "Creating .env file from template..."
    cp env.example .env
    echo "Please edit .env file and add your OpenRouter API key"
fi

echo "Setup complete!"
echo "To activate the environment, run: source venv/bin/activate"
