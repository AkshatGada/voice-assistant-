#!/bin/bash
# Voice Assistant Startup Script

echo "🎤 Voice Assistant Startup"
echo "=========================="
echo ""

# Check Python version
python3 --version

# Check if dependencies are installed
echo ""
echo "Checking dependencies..."
python3 -c "import flask; import flask_cors; import mlx_whisper; import numpy; import soundfile; from kokoro import KPipeline; print('✅ All core dependencies installed')" 2>&1 | grep -E "(✅|Error|ModuleNotFound)" || echo "⚠️  Some dependencies may be missing"

# Check mlx_lm
python3 -c "from mlx_lm import load; print('✅ mlx_lm available')" 2>&1 | grep -E "(✅|Error|ModuleNotFound)" || echo "⚠️  mlx_lm may need installation"

# Check model path
if [ -d "/Users/agada/.lmstudio/models/mlx-community/gemma-3-4b-it-qat-4bit" ]; then
    echo "✅ Gemma model found"
else
    echo "⚠️  Gemma model not found at expected path"
fi

echo ""
echo "Starting server on http://localhost:3000"
echo "Press Ctrl+C to stop"
echo ""

python3 app.py
