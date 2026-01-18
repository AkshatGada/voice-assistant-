# Quick Start Guide

## Installation

1. **Install Python dependencies:**
   ```bash
   cd /Users/agada/voice-assistant
   python3 -m pip install -r requirements.txt
   ```

   If you get errors, install individually:
   ```bash
   python3 -m pip install flask flask-cors mlx-whisper mlx-lm kokoro numpy soundfile werkzeug
   ```

2. **Verify system dependencies:**
   ```bash
   brew install ffmpeg espeak-ng  # If not already installed
   ```

## Running the Application

### Option 1: Using the startup script
```bash
./start.sh
```

### Option 2: Direct Python execution
```bash
python3 app.py
```

The server will start on **http://localhost:3000**

## Usage

1. Open your browser and navigate to `http://localhost:3000`
2. Click the **"Press to Record"** button
3. Speak your question clearly
4. Click the button again (or wait) to stop recording
5. Wait for processing:
   - Your speech is transcribed to text
   - The LLM generates a response
   - The response is converted to speech
6. The audio response will play automatically

## Troubleshooting

### Models not loading
- **Whisper**: Downloads automatically on first use (may take a few minutes)
- **Gemma**: Ensure model exists at `/Users/agada/.lmstudio/models/mlx-community/gemma-3-4b-it-qat-4bit`
- **Kokoro**: Downloads voice files automatically on first use

### Audio not recording
- Allow microphone access in browser settings
- Use Chrome, Firefox, or Edge (Safari has limited MediaRecorder support)
- Check browser console for errors

### Server errors
- Check console output for detailed error messages
- Verify all dependencies are installed: `pip list | grep -E "(flask|mlx|kokoro)"`
- Ensure port 3000 is not already in use

### Slow performance
- First run will be slower (models loading)
- Use smaller Whisper model in `config.py`: `WHISPER_MODEL = "mlx-community/whisper-tiny"`
- Reduce `LLM_MAX_TOKENS` in `config.py` for faster responses

## API Testing

You can test individual endpoints:

```bash
# Health check
curl http://localhost:3000/health

# Transcribe audio (requires audio file)
curl -X POST -F "audio=@test.wav" http://localhost:3000/transcribe

# Generate LLM response
curl -X POST -H "Content-Type: application/json" \
  -d '{"text":"Hello, how are you?"}' \
  http://localhost:3000/generate

# Synthesize speech
curl -X POST -H "Content-Type: application/json" \
  -d '{"text":"Hello world"}' \
  http://localhost:3000/synthesize \
  --output response.wav
```

## Configuration

Edit `config.py` to customize:
- Server port (default: 3000)
- Whisper model size
- Kokoro voice selection
- LLM temperature and max tokens

## Architecture

```
Browser (Frontend)
    ↓ [Audio Recording]
Flask Server (Backend)
    ↓ [STT]
MLX Whisper → Text
    ↓ [LLM]
Gemma 3 4B → Response Text
    ↓ [TTS]
Kokoro → Audio WAV
    ↓ [Playback]
Browser Audio Player
```

All processing happens locally - no cloud services required!
