# Voice Assistant

A complete voice assistant that combines:
- **STT**: MLX Whisper for speech-to-text
- **LLM**: Gemma 3 4B for text generation
- **TTS**: Kokoro for text-to-speech

## Features

- 🎤 Press button to record your question
- 🧠 Local LLM generates intelligent responses
- 🔊 Audio response plays automatically
- 🚀 All processing happens locally on your machine

## Setup

### Prerequisites

1. **Python 3.9+**
2. **ffmpeg** (for audio processing):
   ```bash
   brew install ffmpeg
   ```
3. **espeak-ng** (for Kokoro):
   ```bash
   brew install espeak-ng
   ```

### Installation

1. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Ensure the Gemma model exists at:
   ```
   /Users/agada/.lmstudio/models/mlx-community/gemma-3-4b-it-qat-4bit
   ```

3. (Optional) Configure settings in `config.py` or via environment variables:
   ```bash
   export PORT=3000
   export WHISPER_MODEL="mlx-community/whisper-small"
   export KOKORO_VOICE="af_bella"
   ```

## Usage

1. Start the server:
   ```bash
   python app.py
   ```

2. Open your browser to:
   ```
   http://localhost:3000
   ```

3. Click the record button and speak your question

4. Wait for the assistant to process and respond

## API Endpoints

- `GET /` - Frontend UI
- `GET /health` - Health check and model status
- `POST /transcribe` - Transcribe audio to text
- `POST /generate` - Generate LLM response from text
- `POST /synthesize` - Convert text to speech
- `POST /chat` - Complete pipeline (audio → text → LLM → audio)

## Configuration

Edit `config.py` to customize:
- Model paths
- Server port
- LLM temperature and max tokens
- TTS voice selection

## Troubleshooting

### Models not loading
- Check that model paths exist
- Verify you have enough RAM (Gemma 4B needs ~4GB)
- Check console logs for specific errors

### Audio not recording
- Allow microphone access in browser settings
- Use Chrome, Firefox, or Edge (Safari has limited support)

### Slow responses
- Use smaller Whisper model (tiny/base instead of large)
- Reduce LLM max_tokens
- Ensure you're using Apple Silicon for MLX acceleration

## Architecture

```
Browser → Flask Server → MLX Whisper → Gemma LLM → Kokoro TTS → Browser
```

All processing happens locally - no cloud services required!
