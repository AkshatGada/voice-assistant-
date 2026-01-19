"""Configuration for Voice Assistant"""

import os

# Server Configuration
SERVER_PORT = int(os.getenv("PORT", 3000))
SERVER_HOST = os.getenv("HOST", "127.0.0.1")
DEBUG = os.getenv("DEBUG", "False").lower() == "true"

# Model Paths
# Using mlx-community/whisper-tiny (default, guaranteed to exist)
# This is the fastest Whisper model and works well for voice commands
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "mlx-community/whisper-tiny")
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
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "200"))  # Increased for more complete responses

# Optimized System Prompt for Jarvis with Tool Calling Support
SYSTEM_PROMPT = """You are Jarvis, a helpful voice assistant. You can speak naturally AND call tools when needed. 

IMPORTANT: When you need to use a tool:
1. Speak naturally to the user first (e.g., "Let me search for that")
2. Then wrap the JSON call in <tool>...</tool> tags
3. Format: Speak naturally. <tool>{"name": "tool_name", "input": "value"}</tool> Continue speaking if needed.

Always keep speech natural and under 20 words when not using tools. Use contractions (I'll, I'm). Be warm and direct.

Available tools: search_files, get_weather, execute_command (mock for now)."""

# Filler configuration for psychological latency masking
ENABLE_FILLER_TOKENS = os.getenv("ENABLE_FILLER_TOKENS", "true").lower() == "true"
FILLER_TOKENS = [
    "Sure,", "Okay,", "Alright,", "Let me see,", "Got it,", 
    "Right,", "Absolutely,", "Of course,", "Well,", "Hmm,",
    "Indeed,", "Certainly,", "I see,", "Interesting,", "Ah,",
    "Yes,", "Understood,", "Fair enough,", "Good question,", "Let me think,"
]

# WebSocket Configuration
ENABLE_WEBSOCKET_MODE = os.getenv("ENABLE_WS", "true").lower() == "true"

# Temporary file storage
TEMP_DIR = os.path.join(os.path.dirname(__file__), "temp")
os.makedirs(TEMP_DIR, exist_ok=True)
