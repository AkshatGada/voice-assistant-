# Voice Assistant Architecture

## System Overview

The Voice Assistant is a complete **STT → LLM → TTS pipeline** that processes voice input locally using MLX-accelerated models. All components run on-device with no cloud dependencies.

**Architecture Evolution**: The system has been optimized for sub-1.5s perceived latency using:
- **WebSocket Streaming**: Real-time bidirectional communication with progressive audio delivery
- **Voice Activity Detection (VAD)**: Browser-side auto-stop on speech end (500ms silence threshold)
- **Psychological Latency Masking**: Filler tokens ("Sure,", "Okay,", etc.) for instant feedback
- **Progressive Audio Playback**: Stream audio chunks as they're generated using Web Audio API
- **Natural Transitions**: 75ms silence padding after filler words for seamless voice continuity

```
┌─────────┐      ┌──────────────┐      ┌──────────┐      ┌──────────┐      ┌─────────┐
│ Browser │─────▶│ Flask Server │─────▶│  Whisper │─────▶│  Gemma   │─────▶│ Kokoro  │
│ (WebM)  │◀────│  (WebSocket) │      │   STT    │      │   LLM    │      │   TTS   │
│ + VAD   │      │              │      │          │      │          │      │         │
└─────────┘      └──────────────┘      └──────────┘      └──────────┘      └─────────┘
     ▲                                       │                  │                 │
     │                                       │                  │                 │
     └─────────────────────────────────────────────────────────────────────────────┘
                        Streaming Audio Chunks (Progressive Playback)
```

## Components

### 1. Speech-to-Text (STT) - Whisper Tiny

**Model**: `mlx-community/whisper-tiny` (lightweight, fast inference)

**Responsibilities**:
- Converts WebM audio recordings to text
- Automatic language detection
- Handles various audio formats via internal ffmpeg conversion

**Technical Details**:
- **Sample Rate**: 16 kHz mono (automatically converted)
- **Model Size**: Whisper-tiny variant (~75MB, lightweight)
- **Library**: `mlx_whisper` (Apple MLX framework)
- **Audio Format**: Accepts WebM/MP3/WAV via ffmpeg (internal conversion)
- **Performance**: Fast inference on Apple Silicon, optimized for real-time use

**Key Features**:
- Uses ffmpeg internally for format conversion
- No manual audio preprocessing required
- Fast inference on Apple Silicon

### 2. Large Language Model (LLM) - Gemma 3 4B

**Model**: `gemma-3-4b-it-qat-4bit` (Quantized 4-bit)

**Responsibilities**:
- Generates contextual responses to transcribed text
- Conversational AI powered by instruction-tuned Gemma
- Maintains Jarvis personality via system prompt

**Technical Details**:
- **Model Size**: 4B parameters, 4-bit quantization (~2-3GB RAM)
- **Framework**: MLX-LM (optimized for Apple Silicon)
- **System Prompt**: Optimized 150-character prompt for speed
- **Max Tokens**: 100 tokens (reduced from 512 for faster responses)
- **Tokenizer**: Gemma tokenizer with chat template support

**Key Features**:
- Quantized 4-bit weights for memory efficiency
- Instruction-tuned for chat/conversation
- Supports streaming token generation
- Greedy decoding for speed optimization

### 3. Text-to-Speech (TTS) - Kokoro Pipeline

**Model**: Kokoro TTS with `af_heart` voice

**Responsibilities**:
- Synthesizes natural speech from LLM text output
- Generates audio waveforms for playback

**Technical Details**:
- **Voice**: `af_heart` (American English)
- **Sample Rate**: 24 kHz
- **Speed**: 1.15x (15% faster for natural conversation pace)
- **Language**: American English (`lang_code="a"`)
- **Precision**: FP16 (optimized for Apple Neural Engine)
- **Filler Synthesis**: Instant synthesis of filler words with 75ms silence padding for seamless transitions

**Key Features**:
- High-quality neural TTS
- Multiple voice options available
- Streaming support for progressive audio generation
- Low-latency synthesis

### 4. Web Frontend (Browser)

**Technology**: Vanilla JavaScript + HTML5 MediaRecorder API

**Responsibilities**:
- Captures microphone input as WebM audio
- Sends audio to backend via FormData
- Displays transcription and responses
- Plays synthesized audio responses
- Tracks end-to-end latency

**Technical Details**:
- **Audio Format**: `audio/webm;codecs=opus`
- **Bitrate**: 128 kbps
- **Recording**: VAD auto-stop (500ms silence) or manual button
- **Playback**: Progressive Web Audio API (streaming chunks as they arrive)
- **VAD**: Browser-side Voice Activity Detection (@ricky0123/vad-web, WASM-based Silero VAD)
- **WebSocket**: Real-time bidirectional communication via Flask-SocketIO
- **Audio Streaming**: Complete audio blob sent after recording stops (not chunk-by-chunk during recording)

### 5. Flask Backend Server

**Framework**: Flask with CORS support

**Responsibilities**:
- Orchestrates the STT → LLM → TTS pipeline
- Manages model loading and caching
- Handles audio file I/O
- Provides RESTful API endpoints

**Endpoints**:
- `GET /` - Frontend HTML
- `GET /health` - Model status check
- `POST /transcribe` - Standalone STT
- `POST /generate` - Standalone LLM
- `POST /synthesize` - Standalone TTS
- `POST /chat` - Complete pipeline (HTTP fallback)
- `GET /audio_stream/<request_id>` - Stream audio from memory

**WebSocket Events**:
- `connect` - Client connection established
- `disconnect` - Client disconnection
- `audio_complete` - Complete audio blob received (after recording stops)
- `transcription` - STT result sent to client
- `audio_chunk` - Progressive audio chunks (filler + sentences)
- `audio_end` - Final audio chunk, streaming complete
- `error` - Error message to client

## Data Flow

### Request Pipeline (Audio → Response)

**WebSocket Mode (Primary)**:
```
1. Browser captures WebM audio (MediaRecorder + VAD)
   ↓
2. VAD detects speech end (500ms silence) → auto-stop
   ↓
3. Complete audio blob sent via WebSocket `audio_complete` event
   ↓
4. Server saves audio to temp file (.webm)
   ↓
5. mlx_whisper.transcribe() → text
   │   - Uses ffmpeg internally for conversion
   │   - Returns: {"text": "...", "language": "en", ...}
   ↓
6. Server emits `transcription` event to client
   ↓
7. process_audio_streaming() → LLM + TTS pipeline
   │   ├─ Stream LLM tokens (sentence-by-sentence)
   │   ├─ Filler token selected and synthesized immediately
   │   ├─ 75ms silence padding after filler
   │   ├─ Detect sentence boundaries (., !, ?)
   │   └─ For each sentence:
   │       └─ Kokoro TTS synthesis → audio chunks
   │   - Emits `audio_chunk` events progressively
   ↓
8. Client receives audio chunks via WebSocket
   ↓
9. Progressive playback via Web Audio API (audio-player.js)
   ↓
10. Server emits `audio_end` when complete
```

**HTTP Fallback Mode**:
```
1. Browser captures WebM audio (MediaRecorder)
   ↓
2. POST /chat with FormData (audio file)
   ↓
3. Save audio to temp file (.webm)
   ↓
4. mlx_whisper.transcribe() → text
   ↓
5. generate_with_streaming_tts(transcribed_text)
   │   ├─ Stream LLM tokens (sentence-by-sentence)
   │   ├─ Filler token synthesized immediately
   │   ├─ 75ms silence padding after filler
   │   └─ TTS synthesis per sentence
   ↓
6. Cache audio in memory (audio_memory_cache)
   ↓
7. Return JSON: {transcribed_text, llm_response, audio_url, timings}
   ↓
8. Browser fetches /audio_stream/<request_id>
   ↓
9. Play audio via HTML5 Audio element
```

### Audio Streaming Architecture

**Memory-Based Audio Delivery**:
- Generated audio stored in `audio_memory_cache` dict
- Key: 8-character UUID
- Value: `(audio_array, sample_rate)` tuple
- Served directly via `Response` with `io.BytesIO`
- Eliminates file I/O overhead for faster delivery

## Optimizations Applied

### 1. Eager Model Loading
**What**: Pre-load all models during server startup  
**Impact**: Eliminates first-request latency (~30-60s saved)  
**Implementation**:
- Models loaded in `if __name__ == "__main__"` block
- Dummy inference calls to "warm up" models
- LLM warmup with test prompt

**Benefits**:
- Consistent response times from first request
- Models ready immediately after server starts

### 5. Parallel LLM + TTS Streaming
**What**: Start TTS synthesis as soon as first sentence is generated  
**Impact**: Reduces perceived latency by ~40-60%  
**Implementation**:
- `generate_with_streaming_tts()` function
- LLM streams tokens sentence-by-sentence
- TTS synthesizes each sentence immediately
- Audio chunks concatenated progressively

**Benefits**:
- User hears response faster (as soon as first sentence ready)
- LLM and TTS run in parallel after first sentence
- Total latency: max(LLM_time, TTS_time) instead of sum

### 6. Greedy Decoding (Streaming)
**What**: Use greedy sampler (temp=0) for streaming LLM responses  
**Impact**: Faster token generation, more deterministic  
**Implementation**:
- Custom `greedy_sampler()` function returns `mx.argmax(logits)`
- Applied to `mlx_stream_generate()` calls
- Skips probabilistic sampling overhead

**Benefits**:
- Faster token generation (no sampling computation)
- More predictable responses
- Lower CPU/GPU utilization

### 7. Audio Memory Cache
**What**: Serve audio directly from memory instead of disk  
**Impact**: Eliminates file I/O, reduces latency by ~10-50ms  
**Implementation**:
- `audio_memory_cache` global dictionary
- Audio stored as numpy array + sample rate
- `/audio_stream/<request_id>` endpoint serves from memory
- WAV conversion done in-memory with `io.BytesIO`

**Benefits**:
- No temporary file creation/deletion overhead
- Faster audio delivery to frontend
- Reduced disk I/O operations

### 8. Optimized System Prompt
**What**: Reduced system prompt from ~1964 to 150 characters  
**Impact**: Faster LLM processing, lower token costs  
**Implementation**:
- Concise Jarvis personality definition
- Removed verbose instructions
- Maintains conversational tone

**Benefits**:
- Fewer tokens to process = faster generation
- Lower memory usage for prompt encoding
- Maintains quality with less overhead

### 9. Reduced LLM Max Tokens
**What**: Limit generation to 100 tokens (from 512)  
**Impact**: Faster LLM inference, more concise responses  
**Implementation**:
- `LLM_MAX_TOKENS = 100` in config
- Applied to both streaming and non-streaming modes
- Balances speed vs. response length

**Benefits**:
- Shorter generation time
- More focused, concise responses (good for voice)
- Less GPU/CPU usage per request

### 10. System Prompt Token Caching
**What**: Pre-tokenize system prompt for potential KV cache reuse  
**Impact**: Future optimization (not fully implemented)  
**Implementation**:
- `gemma_system_prompt_cache` stores tokenized prompt
- Prepared during model loading
- Ready for KV cache integration if mlx_lm supports it

**Benefits**:
- Infrastructure for future KV cache optimization
- Potential to skip recomputing system prompt embeddings

### 11. Faster TTS Speech Rate
**What**: Increase TTS speed to 1.15x (15% faster)  
**Impact**: Slightly faster audio playback  
**Implementation**:
- `KOKORO_SPEED = 1.15` in config
- Applied to all TTS synthesis calls

**Benefits**:
- More natural conversation pace
- Slightly reduced perceived latency
- Still sounds natural (not robotic)

### 12. Condition-on-Previous-Text Disabled

### 13. Psychological Latency Masking with Filler Tokens
**What**: Instant filler word synthesis ("Sure,", "Okay,", etc.) before LLM response  
**Impact**: Perceived latency reduced from ~3.5s to ~0.5s  
**Implementation**:
- `_select_filler()` function chooses filler based on prompt type
- 20 diverse filler tokens in `FILLER_TOKENS` config
- Filler synthesized immediately when detected
- 75ms silence padding after filler for seamless voice transition

**Benefits**:
- User hears instant feedback (feels like system is "thinking")
- Masks actual LLM processing time
- Natural human-like conversation flow
- Seamless voice continuity with silence padding

### 14. FP16 Precision for TTS (Phase 4)
**What**: Use FP16 precision for Kokoro TTS  
**Impact**: ~0.1-0.2s reduction (Neural Engine optimization)  
**Implementation**:
- `KOKORO_MODEL_PRECISION = "fp16"` in config
- Leverages Apple Neural Engine acceleration

**Benefits**:
- Faster TTS synthesis
- Lower memory usage
- Better performance on Apple Silicon
**What**: Disable Whisper's context dependency  
**Impact**: Faster STT processing  
**Implementation**:
- `condition_on_previous_text=False` in transcribe calls
- Reduces computational overhead

**Benefits**:
- Faster transcription (~100-200ms saved)
- Independent processing of each audio chunk
- Good for short voice commands

## Latency Breakdown

### Typical Latency (from production logs)

| Stage | Average Time | Range | Notes |
|-------|--------------|-------|-------|
| **STT (Speech-to-Text)** | 0.6s | 0.4-2.0s | Varies with audio length |
| **LLM (Response Generation)** | 1.7s | 1.4-2.2s | Streamed, sentence-by-sentence |
| **TTS (Text-to-Speech)** | 1.6s | 0.9-2.7s | Parallel with LLM (after first sentence) |
| **Total Backend** | **3.0-4.0s** | 2.9-4.8s | Measured from audio received to audio ready |
| **Network/Other** | ~50-200ms | Variable | Browser upload, audio download |

### End-to-End Latency

**Measurement Point**: User finishes speaking → Jarvis starts speaking

```
Typical: 3.5-4.5 seconds
Best Case: ~3.0 seconds (short query, fast LLM response)
Worst Case: ~5.0 seconds (long query, complex response)
```

### Latency Optimization Impact

| Optimization | Latency Saved | Notes |
|--------------|---------------|-------|
| Eager Model Loading | ~30-60s (first request) | One-time benefit |
| Parallel LLM+TTS | ~1-2s per request | Major improvement |
| Greedy Decoding | ~100-300ms | Faster token generation |
| Audio Memory Cache | ~10-50ms | Faster delivery |
| Reduced Max Tokens | ~200-500ms | Shorter generation |
| Optimized System Prompt | ~50-100ms | Fewer prompt tokens |
| **Total Per-Request Savings** | **~1.5-3.0s** | Without these: ~5-7s |

### Latency Measurement

**Backend Timing** (in `app.py`):
```python
timings = {
    "stt_start": time.time(),
    "stt_end": time.time(),
    "llm_start": time.time(),
    "llm_end": time.time(),
    "tts_start": time.time(),
    "tts_end": time.time(),
    "total_latency": calculated
}
```

**Frontend Timing** (in `app.js`):
- Tracks `window.userFinishedSpeakingTime`
- Measures `audioPlayer.onplay` event
- Calculates end-to-end latency
- Displays breakdown in console and UI

## System Configuration

### Model Paths
```python
WHISPER_MODEL = "mlx-community/whisper-tiny"
GEMMA_MODEL_PATH = "/Users/agada/.lmstudio/models/mlx-community/gemma-3-4b-it-qat-4bit"
KOKORO_VOICE = "af_heart"
ENABLE_FILLER_TOKENS = True
ENABLE_WEBSOCKET_MODE = True
```

### Performance Tuning
```python
LLM_MAX_TOKENS = 100           # Reduced for speed
KOKORO_SPEED = 1.15            # 15% faster speech
SYSTEM_PROMPT = "...optimized" # 150 chars
```

### Hardware Requirements
- **CPU**: Apple Silicon (M1/M2/M3) recommended
- **RAM**: ~4-6 GB (for all models loaded)
- **Storage**: ~5 GB (models + dependencies)

## File Structure

```
voice-assistant/
├── app.py                  # Main Flask server & pipeline (WebSocket + HTTP)
├── config.py               # Configuration settings
├── static/
│   ├── index.html          # Frontend UI
│   ├── app.js              # Browser JavaScript (VAD + WebSocket client)
│   ├── websocket-handler.js # WebSocket connection management
│   ├── audio-player.js     # Progressive audio playback (Web Audio API)
│   └── vad-config.js       # VAD configuration parameters
├── temp/                   # Temporary audio files
├── requirements.txt        # Python dependencies (includes flask-socketio)
├── ARCHITECTURE.md         # This file
└── OPTIMIZATION_SUMMARY.md # Detailed optimization history
```

## Recent Improvements

### Graceful Shutdown Handler
**What**: Signal handlers for SIGINT/SIGTERM to prevent sentencepiece crash  
**Impact**: Clean server shutdown without macOS crash reports  
**Implementation**:
- `signal_handler()` function intercepts Ctrl+C
- Uses `os._exit(0)` to skip C++ extension cleanup
- Prevents known sentencepiece library crash on macOS

**Benefits**:
- No more "Python quit unexpectedly" crash reports
- Clean shutdown experience
- Handles both SIGINT (Ctrl+C) and SIGTERM

## Future Optimization Opportunities

1. **KV Cache for System Prompt**: Reuse pre-computed embeddings
2. **Quantized Whisper Model**: Use 4-bit or 8-bit quantized version
3. **Batch Processing**: Process multiple requests in parallel
4. **Audio Compression**: Use Opus/MP3 for smaller downloads
5. **Model Pruning**: Further reduce model sizes without quality loss
6. **Hardware Acceleration**: Leverage Neural Engine on Apple Silicon
7. **Voice Cloning**: Match filler voice to LLM output voice for perfect continuity

## Architecture Decisions

### Why MLX Framework?
- **Native Apple Silicon Support**: Optimized for M-series chips
- **Efficient Memory Usage**: Better than PyTorch for local inference
- **Fast Inference**: Low-latency model execution

### Why Local Processing?
- **Privacy**: No data leaves the device
- **Latency**: No network round-trips
- **Offline Capability**: Works without internet
- **Cost**: No API fees

### Why Streaming TTS?
- **Perceived Latency**: User hears response faster
- **Better UX**: Feels more conversational
- **Parallelism**: LLM and TTS run concurrently

### Why Greedy Decoding?
- **Speed**: Fastest generation method
- **Determinism**: Reproducible outputs
- **Sufficient Quality**: Works well for conversational AI

## Monitoring & Debugging

### Health Check Endpoint
`GET /health` returns:
```json
{
  "status": "ok",
  "whisper_ready": true,
  "gemma_ready": true,
  "tts_ready": true,
  "all_models_loaded": true
}
```

### Logging
- Model loading status
- Latency breakdown per request
- Transcription text
- Error traces with full stack traces

### Performance Metrics
- STT latency (per request)
- LLM latency (estimated, streaming mode)
- TTS latency (estimated, streaming mode)
- Total pipeline latency
- End-to-end latency (frontend measurement)

## Tool Calling Support (Phase 5: Added 2026-01-19)

### Overview

Added support for **Tool Calling** to enable the assistant to perform actions beyond speech. The system now separates "Speaking" from "Doing":

- **Speech**: Natural language responses that are converted to audio via TTS
- **Tools**: Function calls wrapped in `<tool>...</tool>` tags that are executed but NOT converted to speech

### System Prompt Update

Updated `SYSTEM_PROMPT` in `config.py` to instruct Gemma to use a structured format for tool calls:

```
You are Jarvis, a helpful voice assistant. You can speak naturally AND call tools when needed.

IMPORTANT: When you need to use a tool:
1. Speak naturally to the user first (e.g., "Let me search for that")
2. Then wrap the JSON call in <tool>...</tool> tags
3. Format: Speak naturally. <tool>{"name": "tool_name", "input": "value"}</tool> Continue speaking if needed.

Always keep speech natural and under 20 words when not using tools. Use contractions (I'll, I'm). Be warm and direct.

Available tools: search_files, get_weather, execute_command (mock for now).
```

### StreamFilter Class

Implemented `StreamFilter` class in `app.py` to handle the mixed stream of speech + tool calls:

**Key Features:**
- **Token-by-token processing**: Analyzes each LLM token as it streams
- **Tag detection**: Watches for `<tool>` opening and `</tool>` closing tags
- **Edge case handling**: Correctly handles tokens that split tags (e.g., token 1: `<to`, token 2: `ol>`)
- **Speech buffering**: Accumulates speech outside tags for immediate TTS
- **Tool buffering**: Accumulates JSON inside tags for parsing
- **JSON parsing**: Automatically parses and validates tool JSON

**Methods:**
- `process_token(token)` → `(speech_text, tool_json, is_speech)`
  - Returns speech to send to TTS and detected tool calls
- `flush()` → `(remaining_speech, remaining_tool)`
  - Called at stream end to handle incomplete buffers

**Example token flow:**
```
Token: "Sure,"        → (speech: "Sure,", tool: None, is_speech: True)
Token: " let"         → (speech: " let", tool: None, is_speech: True)
Token: " me search"   → (speech: " me search", tool: None, is_speech: True)
Token: " <tool>"      → (speech: " ", tool: None, is_speech: True) [enters tool mode]
Token: "{\"name\""    → (speech: "", tool: None, is_speech: False) [buffering JSON]
Token: ": \"search\"" → (speech: "", tool: None, is_speech: False) [buffering JSON]
Token: "}"            → (speech: "", tool: None, is_speech: False) [buffering JSON]
Token: "</tool>"      → (speech: "", tool: {parsed_json}, is_speech: False) [tool complete]
Token: " for that"    → (speech: " for that", tool: None, is_speech: True) [back to speech]
```

### Integration Points

#### 1. `process_audio_streaming()` (WebSocket mode)

Updated to use `StreamFilter`:

```python
stream_filter = StreamFilter()

for token, current_text in generate_llm_response(...):
    speech_text, tool_result, is_speech = stream_filter.process_token(token)
    
    if tool_result:
        print(f"🔧 [TOOL CALL DETECTED]: {tool_result}")
        socketio.emit('tool_call', {'tool': tool_result, 'text': "Executing tool..."}, room=session_id)
    
    if is_speech and speech_text.strip():
        # Stream to TTS as before (sentence buffering, filler tokens, etc.)
```

Features:
- Speech still streams to TTS immediately (low latency)
- Tool calls are logged to console and emitted via WebSocket
- No audio generated for tool calls

#### 2. `generate_with_streaming_tts()` (HTTP fallback mode)

Updated identically:
- Uses `StreamFilter` to parse mixed output
- Tools logged to console
- Only speech audio generated

#### 3. New WebSocket Event: `tool_call`

Frontend receives tool execution details:
```json
{
  "tool": {
    "name": "search_files",
    "input": "optimization"
  },
  "text": "Executing tool..."
}
```

### Latency Impact

**Speech latency**: No change
- Speech still streams sentence-by-sentence immediately
- Tool detection adds negligible overhead (<1ms per token)

**Tool latency**: New capability
- Tool detection: ~1-5ms per tool call
- JSON parsing: ~5-10ms per tool call
- Tool execution: Implementation-dependent (currently mock)

### Mock Execution

Tools are currently **mock** (no-op):
- Detected tool calls are logged to console
- Emitted to frontend via WebSocket
- Ready for real implementation

**Example console output:**
```
🔧 [TOOL CALL DETECTED]: {'name': 'search_files', 'input': 'optimization'}
   Tool input: {'name': 'search_files', 'input': 'optimization'}
```

### Available Tools (Framework Ready)

System prompt lists available tools:
- `search_files`: Search for files by query
- `get_weather`: Fetch weather information
- `execute_command`: Run shell commands (mock for now)

Can be extended by:
1. Adding tool to system prompt
2. Implementing tool execution handler
3. Returning results in follow-up LLM call (optional)

### Example Interaction

**User**: "Search my files for optimization details"

**System flow**:
1. Speech-to-text: "Search my files for optimization details"
2. LLM generates: "Sure, let me search. <tool>{"name": "search_files", "input": "optimization"}</tool> Found it!"
3. Stream filter processes:
   - "Sure, let me search. " → speech
   - `{"name": "search_files", "input": "optimization"}` → tool (logged, not spoken)
   - " Found it!" → speech
4. TTS outputs: "Sure, let me search. Found it!"
5. Tool call logged and available for execution

### Implementation Details

**Buffer management:**
- Avoids unbounded growth
- Clears buffers after tag completion
- Handles stream end with `flush()`

**Error handling:**
- Invalid JSON logged as warning
- Malformed tool calls still allow speech to continue
- Incomplete tool tags at stream end logged

**Performance optimizations:**
- Single-pass token processing
- Minimal string operations
- No regex needed for tag detection
- Memory efficient (O(1) per token)

### Future Enhancements

1. **Real Tool Execution**: Implement actual tool functions (file search, API calls, shell commands)
2. **Tool Results in Context**: Feed tool outputs back to LLM for context-aware responses
3. **Multi-tool Calls**: Handle multiple tool calls in single LLM response
4. **Tool Confirmation**: Ask user before executing certain tools
5. **Async Tool Execution**: Run tools in background, continue streaming speech
6. **Tool Timeouts**: Set max execution time for long-running tools
7. **Tool Error Handling**: Graceful fallback if tool execution fails
8. **Tool Logging**: Track all tool calls for debugging/auditing
9. **Tool Parameters Validation**: Validate tool inputs before execution
10. **Tool Results Caching**: Cache results to avoid duplicate executions

### Testing

To test tool calling:

1. **Enable tool usage in prompt**:
   - System prompt already includes tool format instructions

2. **Start server**:
   ```bash
   python3 app.py
   ```

3. **Speak a command that requires a tool**:
   - Example: "Search my files for optimization"
   - Example: "What's the weather in San Francisco?"
   - Example: "List files in my documents"

4. **Observe output**:
   - Console: `🔧 [TOOL CALL DETECTED]: {...}`
   - Browser console: `tool_call` WebSocket event

5. **Verify latency**:
   - Speech still streams with low latency
   - Tool calls don't delay audio output
   - Speech after tool call is included in TTS

### Files Modified

- `config.py`: Updated `SYSTEM_PROMPT` with tool calling instructions
- `app.py`:
  - Added `StreamFilter` class (lines 38-135)
  - Updated `generate_with_streaming_tts()` to use `StreamFilter`
  - Updated `process_audio_streaming()` to use `StreamFilter` and emit `tool_call` events
  - Added imports: `json`, `re`
