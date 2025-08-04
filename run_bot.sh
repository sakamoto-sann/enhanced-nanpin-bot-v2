#!/bin/bash
# Run nanpin bot with proper environment setup

# Check if virtual environment exists
if [ ! -d "nanpin_env" ]; then
    echo "Creating virtual environment..."
    python3 -m venv nanpin_env
fi

# Activate virtual environment
source nanpin_env/bin/activate

# Install all required dependencies
pip install -q python-dotenv pyyaml pandas aiohttp cryptography websockets requests

# Create logs directory if it doesn't exist
mkdir -p logs

# Run the bot
python launch_nanpin_bot_fixed.py