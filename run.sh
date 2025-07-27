#!/bin/bash

# Run script for LLM Knowledge Graph Generation API

echo "🚀 Starting LLM Knowledge Graph Generation API..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install/upgrade dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "⚠️  .env file not found. Please copy .env.example to .env and add your API keys."
    echo "   cp .env.example .env"
    echo "   Then edit .env with your actual API keys."
    exit 1
fi

# Run the application
echo "🎯 Starting the API server..."
python -m src.app.main
