#!/bin/bash
# 🏆 Universal Stock Analyzer Installation Script

echo "🏆 Universal Stock Analyzer Installation"
echo "========================================"
echo ""

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" = "$required_version" ]; then
    echo "✅ Python $python_version detected (>= $required_version required)"
else
    echo "❌ Python $required_version or higher required. Found: $python_version"
    exit 1
fi

echo ""
echo "📦 Installing dependencies..."
pip3 install -r requirements.txt

echo ""
echo "🔧 Making scripts executable..."
chmod +x analyze

echo ""
echo "✅ Installation complete!"
echo ""
echo "🚀 Quick Start:"
echo "  Command line: ./analyze AAPL"
echo "  Web interface: ./analyze --web"
echo ""
echo "📚 For more information, see README.md"
echo ""
echo "🎉 Ready to analyze stocks like a pro!"
