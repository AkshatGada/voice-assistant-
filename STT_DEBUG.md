# STT Transcription Debugging

## Current Issue
- Audio files are being saved (1-3 KB, 2-4 seconds duration)
- Whisper returns 0 segments and empty transcription
- Language is detected as "en" but no text is extracted

## How MLX Whisper Works (from mlx-examples/whisper)

### Audio Loading Process
MLX Whisper uses **ffmpeg internally** to handle audio conversion:

```python
# From mlx_whisper/audio.py
cmd = ["ffmpeg", "-nostdin", "-i", file]
cmd.extend([
    "-threads", "0",
    "-f", "s16le",          # signed 16-bit little-endian
    "-ac", "1",             # mono channel
    "-acodec", "pcm_s16le", # PCM codec
    "-ar", str(16000),      # 16kHz sample rate
    "-"                     # output to stdout
])
```

This means:
- **mlx_whisper.transcribe() accepts ANY audio format** (WebM, MP3, WAV, FLAC, etc.)
- It automatically converts to 16kHz mono PCM internally
- No manual conversion needed in our code

### Test Example
From `mlx-examples/whisper/test.py`:
```python
result = mlx_whisper.transcribe(
    TEST_AUDIO,  # Can be .flac, .mp3, .webm, etc.
    path_or_hf_repo=MODEL_PATH,
    fp16=False
)
```

## Possible Causes of Empty Transcription

### 1. **Audio Quality Issues**
- WebM file might have very low volume
- Browser microphone might not be capturing audio properly
- Audio might be corrupted during recording

### 2. **File Size Too Small**
- Current files: 1-3 KB for 2-4 seconds
- This is extremely small (should be ~50-100 KB for 2-4 seconds of audio)
- Suggests very low bitrate or mostly silence

### 3. **WebM Codec Issues**
- Browser is recording with `audio/webm;codecs=opus`
- Opus codec should work fine with ffmpeg
- But low bitrate (128kbps) might be too aggressive

## Debugging Steps

### 1. Enable Verbose Logging
Added `verbose=True` to see Whisper's internal processing:
```python
transcription_result = mlx_whisper.transcribe(
    audio_path,
    path_or_hf_repo=config.WHISPER_MODEL,
    verbose=True,  # See what Whisper is doing
    condition_on_previous_text=False,
)
```

### 2. Check Actual Audio Content
Test if ffmpeg can read the audio:
```bash
ffmpeg -i /path/to/audio.webm -f null -
```

### 3. Manually Test Transcription
Save a test audio file and try:
```python
import mlx_whisper
result = mlx_whisper.transcribe("test.webm", path_or_hf_repo="mlx-community/whisper-tiny")
print(result["text"])
```

## Potential Fixes

### Option 1: Increase Browser Recording Quality
In `static/app.js`:
```javascript
const options = {
    mimeType: 'audio/webm;codecs=opus',
    audioBitsPerSecond: 256000  // Increase from 128000
};
```

### Option 2: Use Different Audio Format
Try recording as WAV instead of WebM (if browser supports it):
```javascript
const options = {
    mimeType: 'audio/wav',  // Or 'audio/webm;codecs=pcm'
};
```

### Option 3: Add Audio Gain/Normalization
Use Web Audio API to boost volume before sending:
```javascript
async function normalizeAudio(audioBlob) {
    const audioContext = new AudioContext();
    const arrayBuffer = await audioBlob.arrayBuffer();
    const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
    
    // Apply gain
    // ... normalize audio ...
    
    return normalizedBlob;
}
```

### Option 4: Test with Known Good Audio
Replace the uploaded audio with a known working audio file to isolate the issue:
- If test audio works → problem is in recording/frontend
- If test audio fails → problem is in backend/Whisper setup

## Next Steps

1. **Check verbose output** - Run server and look for ffmpeg errors
2. **Test file manually** - Download a saved WebM file and test it locally
3. **Try different browser** - Chrome vs Firefox might have different recording quality
4. **Check microphone settings** - System audio input level might be too low
5. **Test with longer recording** - Try 5-10 seconds instead of 2-3 seconds
