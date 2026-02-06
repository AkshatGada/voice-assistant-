# Agent Integration Guide: OpenClaw + Jarvis Voice

If you are an agent tasked with giving OpenClaw a local voice, follow these steps to integrate this repository.

## 1. System Dependencies
The host Mac must have `ffmpeg` installed to process audio:
```bash
brew install ffmpeg
```

## 2. Server Setup
Initialize the local MLX server. Use Python 3.11 or 3.12 for best compatibility.
```bash
# Install dependencies
pip install -r requirements.txt

# Set Gemini API Key (for LLM logic)
export GOOGLE_API_KEY="your_key"

# Start server
./start.sh
```

## 3. The Bridge Skill
To let your OpenClaw agent talk to this server, create a bridge skill inside the OpenClaw workspace:

**Path:** `~/.openclaw/skills/jarvis-voice/SKILL.md`
```markdown
---
name: jarvis-voice
description: Connects OpenClaw to local Jarvis server.
---
# Jarvis Voice Bridge
(Reference this repository for tool implementation)
```

## 4. Voice Loop Architecture
For low-latency "conversational" feel, follow this pattern:
1. Capture audio via client (Telegram/WhatsApp/Web).
2. Download audio to temp file.
3. Call `POST http://localhost:3000/transcribe` with the file.
4. Process text through your LLM.
5. Call `POST http://localhost:3000/synthesize` with the reply.
6. Send the resulting `.wav` back as a voice message.

## Troubleshooting
- **Network Reachability:** If in a Docker container, use `http://host.docker.internal:3000` instead of `localhost`.
- **Module Errors:** If `mlx_whisper` fails to import, verify you are on Apple Silicon (M1/M2/M3).
