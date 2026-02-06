#!/bin/bash
# Voice Assistant Startup Script (Gemini Flash Edition)

echo "🎤 Voice Assistant Startup (Gemini Flash)"
echo "=========================================="
echo ""

# Check Python version
python3 --version

# Check for API Key
if [ -z "$GOOGLE_API_KEY" ]; then
    echo "⚠️  Error: GOOGLE_API_KEY environment variable is not set."
    echo "Please run: export GOOGLE_API_KEY='your_key_here'"
    # exit 1
fi

# Check if dependencies are installed
echo "Checking dependencies..."
python3 -c "import flask; import google.generativeai; import mlx_whisper; from kokoro import KPipeline; print('✅ Core dependencies installed')" 2>&1 | grep -E "(✅|Error|ModuleNotFound)" || echo "⚠️  Some dependencies may be missing. Run: pip install -r requirements.txt"

echo ""
echo "Starting server on http://localhost:3000"
echo "Press Ctrl+C to stop"
echo ""

python3 app.py
