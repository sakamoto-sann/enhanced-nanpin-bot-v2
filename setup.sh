#!/bin/bash
# 🚀 Nanpin Bot Quick Setup Script

echo "🤖 NANPIN BOT SETUP"
echo "==================="

# Check if Python 3.8+ is installed
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "🐍 Python version: $python_version"

if python3 -c 'import sys; exit(0 if sys.version_info >= (3, 8) else 1)' 2>/dev/null; then
    echo "✅ Python version is compatible"
else
    echo "❌ Python 3.8+ required. Please upgrade Python."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Copy environment file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "📝 Creating .env file from template..."
    cp .env.example .env
    echo "⚠️  Please edit .env file with your Backpack API credentials!"
fi

# Create logs directory if it doesn't exist
mkdir -p logs

echo ""
echo "✅ SETUP COMPLETE!"
echo "==================="
echo "Next steps:"
echo "1. Edit .env file with your Backpack API credentials"
echo "2. Run: python start_nanpin_bot.py"
echo ""
echo "📖 For detailed instructions, see README.md"