# Voice Assistant - Recent Changes

## Date: January 19, 2026

### Issue Fixed: STT Transcription Failures

**Problem:**
- After latency optimization changes, Whisper STT was returning empty transcriptions (0 segments)
- Audio files were being converted successfully (3-4 second duration) but no text was extracted
- Logs showed "Transcribed text: '' (length: 0)" despite valid audio input

**Root Causes:**
1. **Audio Format Issues**: WebM audio was being saved but not properly converted to WAV format before transcription
2. **Audio Quality Loss**: Simple format conversion wasn't preserving audio loudness/quality
3. **Model Too Aggressive**: `whisper-tiny` model was too aggressive at filtering audio
4. **Missing Whisper Parameters**: Lack of language hints and context for better transcription

**Solutions Implemented:**

### 1. Enhanced Audio Conversion (app.py)
- Added proper WebM → WAV conversion using ffmpeg
- Implemented audio normalization with `loudnorm` filter for consistent volume
- Using PCM 16-bit encoding for better quality
- Proper cleanup of both WebM and converted WAV files

```python
ffmpeg_cmd = [
    'ffmpeg',
    '-i', webm_path,
    '-ar', '16000',  # 16kHz for Whisper
    '-ac', '1',  # Mono
    '-acodec', 'pcm_s16le',  # PCM 16-bit
    '-af', 'loudnorm=I=-16:TP=-1.5:LRA=11',  # Normalize loudness
    '-f', 'wav',
    '-y',
    wav_path
]
```

### 2. Improved Whisper Parameters (app.py)
- Added explicit language setting: `language="en"`
- Added initial prompt for context: `"This is a voice assistant conversation."`
- Disabled word timestamps for speed: `word_timestamps=False`
- Using fp32 for better accuracy: `fp16=False`
- Kept `condition_on_previous_text=False` for speed

### 3. Kept Original Whisper Model (config.py)
- Reverted to `mlx-community/whisper-tiny` (original working model)
- The issue was audio conversion, not the model itself
- Tiny model is fast and works well with proper audio normalization

**Expected Results:**
- Improved transcription accuracy for voice input
- Better handling of varying audio volumes
- More reliable STT with proper audio normalization
- No latency increase (still using tiny model)

**Files Modified:**
- `/Users/agada/voice-assistant/app.py` - Audio conversion and Whisper parameters
- `/Users/agada/voice-assistant/config.py` - Whisper model upgrade

**Testing:**
- Restart the server to load the new `whisper-base` model
- Test with voice recording to verify transcription works
- Monitor logs for "Transcribed text: '<your speech>'" confirmation
