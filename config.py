"""Configuration for Voice Assistant"""

import os

# Server Configuration
SERVER_PORT = int(os.getenv("PORT", 3000))
SERVER_HOST = os.getenv("HOST", "127.0.0.1")
DEBUG = os.getenv("DEBUG", "False").lower() == "true"

# Model Paths
# Using distil-whisper for faster STT (5x faster than tiny)
# Fallback: mlx-community/whisper-tiny if distil-whisper not available
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "distil-whisper/distil-small.en")
GEMMA_MODEL_PATH = os.getenv(
    "GEMMA_MODEL_PATH",
    "/Users/agada/.lmstudio/models/mlx-community/gemma-3-4b-it-qat-4bit"
)
KOKORO_LANG_CODE = os.getenv("KOKORO_LANG", "a")  # 'a' for American English
KOKORO_VOICE = os.getenv("KOKORO_VOICE", "af_heart")
KOKORO_SPEED = float(os.getenv("KOKORO_SPEED", "1.15"))  # Slightly faster speech, still natural
KOKORO_MODEL_PRECISION = os.getenv("KOKORO_PRECISION", "fp16")  # Use FP16 for Neural Engine optimization

# Audio Configuration
WHISPER_SAMPLE_RATE = 16000
KOKORO_SAMPLE_RATE = 24000

# LLM Configuration
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "100"))  # Reduced from 512 for faster voice responses

# Optimized System Prompt for Jarvis (reduced from 1964 to ~150 chars for faster LLM processing)
SYSTEM_PROMPT = """You are Jarvis, a helpful voice assistant. Provide natural, concise responses under 20 words. Use contractions (I'll, I'm) for a conversational feel. Be warm and direct."""

# Filler configuration for psychological latency masking
ENABLE_FILLER_TOKENS = os.getenv("ENABLE_FILLER_TOKENS", "true").lower() == "true"
FILLER_TOKENS = ["Sure,", "Okay,", "Alright,", "Let me see,"]

# WebSocket Configuration
ENABLE_WEBSOCKET_MODE = os.getenv("ENABLE_WS", "true").lower() == "true"

# Temporary file storage
TEMP_DIR = os.path.join(os.path.dirname(__file__), "temp")
os.makedirs(TEMP_DIR, exist_ok=True)
