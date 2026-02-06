---
name: jarvis-voice-local
description: High-performance local voice pipeline (MLX Whisper + Kokoro TTS). Use this skill to provide your agent with low-latency STT and TTS capabilities on macOS.
---

# Jarvis Voice (Local MLX Pipeline)

This skill provides your agent with a high-performance local voice pipeline. It bypasses slow cloud APIs by using Apple MLX-accelerated models.

## Tools

### `jarvis_transcribe`
Transcribes audio using a local MLX Whisper instance.
**Input:** `audio_path`

### `jarvis_synthesize`
Generates high-quality speech using local Kokoro TTS.
**Input:** `text`

## Local Server Requirements
The server must be running on your host machine for this skill to function.

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Start the Server:**
   ```bash
   ./start.sh
   ```
3. **Environment:**
   If running in a Docker container, the skill expects the server at `http://host.docker.internal:3000`.
