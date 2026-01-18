"""Voice Assistant Backend - STT → LLM → TTS Pipeline"""

import os
import tempfile
import time
import traceback

import mlx_whisper
import numpy as np
import soundfile as sf
from flask import Flask, jsonify, request, send_file, Response
from flask_cors import CORS
from kokoro import KPipeline

import config

app = Flask(__name__, static_folder="static", static_url_path="")
CORS(app)

# Global model holders (lazy loading)
whisper_model_loaded = False
gemma_model = None
gemma_tokenizer = None
tts_pipeline = None
gemma_system_prompt_cache = None  # KV cache for system prompt

# Audio memory cache for faster delivery
audio_memory_cache = {}


def load_whisper_model():
    """Load Whisper model (lazy loading)"""
    global whisper_model_loaded
    if not whisper_model_loaded:
        print(f"Loading Whisper model: {config.WHISPER_MODEL}")
        # Model is loaded automatically on first transcribe call
        whisper_model_loaded = True
        print("Whisper model ready")


def load_gemma_model():
    """Load Gemma LLM model and initialize system prompt cache"""
    global gemma_model, gemma_tokenizer, gemma_system_prompt_cache
    
    if gemma_model is not None:
        return
    
    try:
        print(f"Loading Gemma model from: {config.GEMMA_MODEL_PATH}")
        from mlx_lm import load
        
        if not os.path.exists(config.GEMMA_MODEL_PATH):
            raise FileNotFoundError(
                f"Model path not found: {config.GEMMA_MODEL_PATH}"
            )
        
        gemma_model, gemma_tokenizer = load(config.GEMMA_MODEL_PATH)
        print("Gemma model loaded successfully")
        
        # Pre-compute system prompt for potential KV cache optimization
        # Note: Actual KV cache implementation depends on mlx_lm internals
        # This prepares the system prompt tokens for future optimization
        try:
            if hasattr(gemma_tokenizer, "apply_chat_template"):
                messages = [{"role": "system", "content": config.SYSTEM_PROMPT}]
                system_prompt = gemma_tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
                system_tokens = gemma_tokenizer.encode(system_prompt)
                gemma_system_prompt_cache = {
                    "tokens": system_tokens,
                    "prompt_text": system_prompt,
                }
                print(f"System prompt prepared for caching ({len(system_tokens)} tokens)")
            else:
                gemma_system_prompt_cache = {
                    "tokens": gemma_tokenizer.encode(config.SYSTEM_PROMPT),
                    "prompt_text": config.SYSTEM_PROMPT,
                }
                print("System prompt prepared for caching (fallback method)")
        except Exception as e:
            print(f"Warning: Could not prepare system prompt cache: {e}")
            gemma_system_prompt_cache = None
    except Exception as e:
        print(f"Error loading Gemma model: {e}")
        traceback.print_exc()
        raise


def load_tts_pipeline():
    """Load Kokoro TTS pipeline (lazy loading)"""
    global tts_pipeline
    
    if tts_pipeline is None:
        try:
            print(f"Loading Kokoro TTS pipeline (lang={config.KOKORO_LANG_CODE})")
            tts_pipeline = KPipeline(lang_code=config.KOKORO_LANG_CODE)
            print("Kokoro TTS pipeline loaded successfully")
        except Exception as e:
            print(f"Error loading Kokoro TTS: {e}")
            traceback.print_exc()
            raise


def _generate_llm_response_streaming(formatted_prompt: str):
    """Internal generator function for streaming LLM responses"""
    global gemma_model, gemma_tokenizer
    from mlx_lm import stream_generate as mlx_stream_generate
    import mlx.core as mx
    
    # Create a greedy sampler (temp=0) for speed
    def greedy_sampler(logits):
        return mx.argmax(logits, axis=-1)
    
    full_response = formatted_prompt
    for response in mlx_stream_generate(
        gemma_model,
        gemma_tokenizer,
        prompt=formatted_prompt,
        max_tokens=config.LLM_MAX_TOKENS,
        sampler=greedy_sampler,  # Greedy decoding for speed (temp=0)
    ):
        # Extract token text from GenerationResponse
        token = response.text
        full_response += token
        # Remove prompt prefix if present
        response_text = full_response
        if response_text.startswith(formatted_prompt):
            response_text = response_text[len(formatted_prompt):].strip()
        yield token, response_text


def generate_with_streaming_tts(transcribed_text):
    """Generate LLM response with parallel TTS streaming
    
    Starts TTS synthesis as soon as first complete sentence is generated,
    dramatically reducing perceived latency.
    
    Returns:
        tuple: (full_text, audio_array)
    """
    import re
    
    load_tts_pipeline()
    
    sentence_buffer = ""
    full_text = ""
    audio_chunks = []
    
    # Stream LLM tokens
    for token, current_text in generate_llm_response(transcribed_text, stream=True):
        sentence_buffer += token
        full_text = current_text
        
        # Detect sentence boundaries (., !, ? followed by optional whitespace)
        if re.search(r'[.!?]\s*$', sentence_buffer.strip()):
            sentence = sentence_buffer.strip()
            if sentence:
                # Synthesize this sentence immediately
                for result in tts_pipeline(
                    sentence, 
                    voice=config.KOKORO_VOICE, 
                    speed=config.KOKORO_SPEED
                ):
                    if result.audio is not None:
                        audio_chunks.append(result.audio.numpy())
            sentence_buffer = ""
    
    # Handle remaining text (if any)
    if sentence_buffer.strip():
        for result in tts_pipeline(
            sentence_buffer.strip(), 
            voice=config.KOKORO_VOICE, 
            speed=config.KOKORO_SPEED
        ):
            if result.audio is not None:
                audio_chunks.append(result.audio.numpy())
    
    final_audio = np.concatenate(audio_chunks) if audio_chunks else np.array([])
    return full_text, final_audio


def generate_llm_response(prompt: str, stream: bool = False):
    """Generate response using Gemma LLM
    
    Args:
        prompt: User prompt text
        stream: If True, yields tokens as they're generated. If False, returns full response.
    
    Returns:
        If stream=False: Complete response string
        If stream=True: Generator yielding (token, full_text_so_far) tuples
    """
    global gemma_model, gemma_tokenizer
    
    if gemma_model is None:
        load_gemma_model()
    
    try:
        # Format as chat prompt if tokenizer has chat template
        if hasattr(gemma_tokenizer, "apply_chat_template"):
            # Include system prompt for Jarvis Orchestrator
            messages = [
                {"role": "system", "content": config.SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ]
            formatted_prompt = gemma_tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            # Fallback: prepend system prompt if no chat template
            formatted_prompt = f"{config.SYSTEM_PROMPT}\n\nUser: {prompt}\nAssistant:"
        
        if stream:
            # Streaming mode - return generator from separate function
            return _generate_llm_response_streaming(formatted_prompt)
        else:
            # Non-streaming mode - return complete response
            from mlx_lm import generate as mlx_generate
            
            response = mlx_generate(
                gemma_model,
                gemma_tokenizer,
                prompt=formatted_prompt,
                max_tokens=config.LLM_MAX_TOKENS,
                verbose=False,
            )
            
            # Remove the prompt from response if it was included
            if response.startswith(formatted_prompt):
                response = response[len(formatted_prompt):].strip()
            
            # Ensure MLX computation is complete
            import mlx.core as mx
            if isinstance(response, str):
                # Response is already a string, no need to eval
                pass
            else:
                # If response is an MLX array, force evaluation
                mx.eval(response)
            
            return response
    except Exception as e:
        print(f"Error generating LLM response: {e}")
        traceback.print_exc()
        raise


@app.route("/")
def index():
    """Serve the frontend"""
    return send_file("static/index.html")


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint - verifies all models are loaded"""
    return jsonify({
        "status": "ok",
        "whisper_ready": whisper_model_loaded,
        "gemma_ready": gemma_model is not None,
        "tts_ready": tts_pipeline is not None,
        "all_models_loaded": whisper_model_loaded and gemma_model is not None and tts_pipeline is not None,
    })


@app.route("/transcribe", methods=["POST"])
def transcribe():
    """Transcribe audio to text using Whisper"""
    try:
        if "audio" not in request.files:
            return jsonify({"error": "No audio file provided"}), 400
        
        audio_file = request.files["audio"]
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=".wav", dir=config.TEMP_DIR
        ) as tmp_file:
            audio_file.save(tmp_file.name)
            temp_path = tmp_file.name
        
        try:
            load_whisper_model()
            
            # Transcribe using mlx_whisper
            result = mlx_whisper.transcribe(
                temp_path,
                path_or_hf_repo=config.WHISPER_MODEL,
                verbose=False,
                condition_on_previous_text=False,  # Faster processing
            )
            
            return jsonify({
                "text": result["text"].strip(),
                "language": result.get("language", "en"),
            })
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    except Exception as e:
        print(f"Transcription error: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/generate", methods=["POST"])
def generate():
    """Generate LLM response from text"""
    try:
        data = request.get_json()
        if not data or "text" not in data:
            return jsonify({"error": "No text provided"}), 400
        
        prompt = data["text"]
        response = generate_llm_response(prompt, stream=False)
        
        return jsonify({"response": response})
    
    except Exception as e:
        print(f"LLM generation error: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/synthesize", methods=["POST"])
def synthesize():
    """Synthesize text to speech using Kokoro"""
    try:
        data = request.get_json()
        if not data or "text" not in data:
            return jsonify({"error": "No text provided"}), 400
        
        text = data["text"]
        
        if not text.strip():
            return jsonify({"error": "Empty text"}), 400
        
        load_tts_pipeline()
        
        # Generate audio using Kokoro
        audio_chunks = []
        for result in tts_pipeline(text, voice=config.KOKORO_VOICE, speed=config.KOKORO_SPEED):
            if result.audio is not None:
                audio_chunks.append(result.audio.numpy())
        
        if not audio_chunks:
            return jsonify({"error": "No audio generated"}), 500
        
        # Concatenate all audio chunks
        audio_array = np.concatenate(audio_chunks)
        
        # Save to temporary WAV file
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=".wav", dir=config.TEMP_DIR
        ) as tmp_file:
            sf.write(tmp_file.name, audio_array, config.KOKORO_SAMPLE_RATE)
            temp_path = tmp_file.name
        
        return send_file(
            temp_path,
            mimetype="audio/wav",
            as_attachment=True,
            download_name="response.wav",
        )
    
    except Exception as e:
        print(f"TTS synthesis error: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/chat", methods=["POST"])
def chat():
    """Complete pipeline: audio → text → LLM → audio"""
    audio_path = None
    response_path = None
    
    # Start timing - this is when we receive the audio (user finished speaking)
    pipeline_start_time = time.time()
    timings = {
        "audio_received": pipeline_start_time,
        "stt_start": None,
        "stt_end": None,
        "llm_start": None,
        "llm_end": None,
        "tts_start": None,
        "tts_end": None,
        "total_latency": None,
    }
    
    try:
        if "audio" not in request.files:
            return jsonify({"error": "No audio file provided"}), 400
        
        audio_file = request.files["audio"]
        
        if audio_file.filename == '':
            return jsonify({"error": "No audio file selected"}), 400
        
        # Step 1: Save uploaded audio temporarily
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=".webm", dir=config.TEMP_DIR
        ) as tmp_file:
            audio_file.save(tmp_file.name)
            audio_path = tmp_file.name
        
        # Check if audio file has content
        file_size = os.path.getsize(audio_path)
        print(f"Uploaded audio file size: {file_size} bytes")
        
        # Try to get audio duration using soundfile or ffprobe
        audio_duration = None
        try:
            import soundfile as sf_check
            with sf_check.SoundFile(audio_path) as f:
                audio_duration = len(f) / f.samplerate
                print(f"Audio duration: {audio_duration:.2f} seconds")
        except Exception as e:
            print(f"Could not determine audio duration: {e}")
            # Try ffprobe as fallback
            try:
                import subprocess
                result = subprocess.run(
                    ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', audio_path],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    audio_duration = float(result.stdout.strip())
                    print(f"Audio duration (from ffprobe): {audio_duration:.2f} seconds")
            except Exception:
                pass
        
        if file_size < 1000:  # Less than 1KB is likely empty or corrupted
            return jsonify({
                "error": f"Audio file too small ({file_size} bytes). Please record for at least 1 second and try again.",
                "transcribed_text": "",
            }), 400
        
        if audio_duration and audio_duration < 0.5:  # Less than 500ms
            return jsonify({
                "error": f"Recording too short ({audio_duration:.2f}s). Please record for at least 1 second.",
                "transcribed_text": "",
                "debug_info": {
                    "file_size": file_size,
                    "duration": audio_duration,
                }
            }), 400
        
        try:
            # Step 2: Transcribe audio to text
            try:
                timings["stt_start"] = time.time()
                load_whisper_model()
                
                load_whisper_model()
                print(f"Transcribing audio file: {audio_path}")
                
                # mlx_whisper.transcribe() uses ffmpeg internally to convert audio
                # It accepts WebM, MP3, WAV, etc. and handles conversion automatically
                transcription_result = mlx_whisper.transcribe(
                    audio_path,
                    path_or_hf_repo=config.WHISPER_MODEL,
                    verbose=False,  # Disable verbose for cleaner output
                    condition_on_previous_text=False,  # Faster processing, no context dependency
                )
                timings["stt_end"] = time.time()
                transcribed_text = transcription_result.get("text", "").strip()
                stt_latency = timings["stt_end"] - timings["stt_start"]
                print(f"Transcribed text: '{transcribed_text}' (length: {len(transcribed_text)})")
                print(f"⏱️  STT latency: {stt_latency:.3f}s")
            except Exception as e:
                print(f"Transcription error: {e}")
                traceback.print_exc()
                return jsonify({
                    "error": f"Transcription failed: {str(e)}",
                    "transcribed_text": "",
                }), 500
            
            if not transcribed_text:
                # Check if there were any segments
                segments = transcription_result.get("segments", [])
                language = transcription_result.get("language", "unknown")
                no_speech_prob = transcription_result.get("no_speech_prob", None)
                
                print(f"No transcribed text. Segments: {len(segments)}, Language: {language}")
                if no_speech_prob is not None:
                    print(f"No speech probability: {no_speech_prob:.3f}")
                
                # Provide more helpful error message based on what we know
                if audio_duration and audio_duration < 1.0:
                    error_msg = f"Recording too short ({audio_duration:.2f}s). Please record for at least 1-2 seconds of clear speech."
                elif segments:
                    # There were segments but no text - might be silence or noise
                    error_msg = "No speech detected in audio. Please speak clearly and try again."
                elif no_speech_prob is not None and no_speech_prob > 0.8:
                    error_msg = "Audio appears to be silence or background noise. Please speak clearly into your microphone."
                else:
                    # No segments at all - might be audio format issue or silence
                    error_msg = "Could not process audio. Please ensure you're speaking clearly and your microphone is working."
                
                return jsonify({
                    "error": error_msg,
                    "transcribed_text": "",
                    "debug_info": {
                        "file_size": file_size,
                        "duration": audio_duration,
                        "segments_count": len(segments),
                        "language": language,
                        "no_speech_prob": no_speech_prob,
                    }
                }), 400
            
            # Step 3 & 4: Generate LLM response with parallel TTS streaming
            try:
                timings["llm_tts_start"] = time.time()
                llm_response, audio_array = generate_with_streaming_tts(transcribed_text)
                timings["llm_tts_end"] = time.time()
                combined_latency = timings["llm_tts_end"] - timings["llm_tts_start"]
                
                if not llm_response or not llm_response.strip():
                    return jsonify({
                        "error": "LLM generated empty response",
                        "transcribed_text": transcribed_text,
                        "llm_response": "",
                        "timings": timings,
                    }), 500
                
                if len(audio_array) == 0:
                    return jsonify({
                        "error": "Failed to generate audio",
                        "transcribed_text": transcribed_text,
                        "llm_response": llm_response,
                        "timings": timings,
                    }), 500
                
                # Calculate individual latencies (approximate)
                # In streaming mode, LLM and TTS overlap, so we estimate
                llm_latency = combined_latency * 0.6  # LLM takes ~60% of combined time
                tts_latency = combined_latency * 0.4  # TTS takes ~40% (parallel)
                
                timings["llm_start"] = timings["llm_tts_start"]
                timings["llm_end"] = timings["llm_start"] + llm_latency
                timings["tts_start"] = timings["llm_start"] + llm_latency * 0.3  # TTS starts after 30% of LLM
                timings["tts_end"] = timings["llm_tts_end"]
                
                print(f"⏱️  LLM+TTS (streaming) latency: {combined_latency:.3f}s")
                print(f"⏱️  Estimated LLM latency: {llm_latency:.3f}s")
                print(f"⏱️  Estimated TTS latency: {tts_latency:.3f}s (parallel)")
            except Exception as e:
                print(f"LLM+TTS streaming error: {e}")
                traceback.print_exc()
                return jsonify({
                    "error": f"LLM+TTS generation failed: {str(e)}",
                    "transcribed_text": transcribed_text,
                    "llm_response": "",
                    "timings": timings,
                }), 500
            
            # Calculate total latency
            timings["total_latency"] = timings["llm_tts_end"] - timings["audio_received"]
            
            # Calculate individual stage latencies
            stt_latency = timings["stt_end"] - timings["stt_start"]
            llm_latency = timings["llm_end"] - timings["llm_start"]
            tts_latency = timings["tts_end"] - timings["tts_start"]
            
            print(f"\n{'='*60}")
            print(f"⏱️  LATENCY BREAKDOWN:")
            print(f"   STT (Speech-to-Text):  {stt_latency:.3f}s")
            print(f"   LLM (Response Gen):    {llm_latency:.3f}s")
            print(f"   TTS (Text-to-Speech):  {tts_latency:.3f}s")
            print(f"   {'-'*60}")
            print(f"   TOTAL LATENCY:         {timings['total_latency']:.3f}s")
            print(f"{'='*60}\n")
            
            # Cache audio in memory for faster delivery
            import uuid
            request_id = str(uuid.uuid4())[:8]
            audio_memory_cache[request_id] = (audio_array, config.KOKORO_SAMPLE_RATE)
            
            # Also save to temp file for compatibility
            try:
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".wav", dir=config.TEMP_DIR
                ) as response_file:
                    sf.write(
                        response_file.name,
                        audio_array,
                        config.KOKORO_SAMPLE_RATE,
                    )
                    response_path = response_file.name
            except Exception as e:
                print(f"Audio file save error: {e}")
                traceback.print_exc()
                return jsonify({
                    "error": f"Failed to save audio: {str(e)}",
                    "transcribed_text": transcribed_text,
                    "llm_response": llm_response,
                }), 500
            
            # Return both text and audio URL (prefer memory stream, fallback to file)
            return jsonify({
                "transcribed_text": transcribed_text,
                "llm_response": llm_response,
                "audio_url": f"/audio_stream/{request_id}",  # Use memory stream
                "audio_url_fallback": f"/audio/{os.path.basename(response_path)}",  # Fallback
                "timings": {
                    "stt_ms": round(stt_latency * 1000, 1),
                    "llm_ms": round(llm_latency * 1000, 1),
                    "tts_ms": round(tts_latency * 1000, 1),
                    "total_ms": round(timings["total_latency"] * 1000, 1),
                    "stt_s": round(stt_latency, 3),
                    "llm_s": round(llm_latency, 3),
                    "tts_s": round(tts_latency, 3),
                    "total_s": round(timings["total_latency"], 3),
                }
            })
        
        finally:
            # Clean up input audio file
            if audio_path and os.path.exists(audio_path):
                try:
                    os.unlink(audio_path)
                except Exception as e:
                    print(f"Warning: Failed to delete temp audio file: {e}")
    
    except Exception as e:
        print(f"Chat pipeline error: {e}")
        traceback.print_exc()
        return jsonify({"error": f"Pipeline error: {str(e)}"}), 500


@app.route("/audio/<filename>", methods=["GET"])
def serve_audio(filename):
    """Serve generated audio files"""
    audio_path = os.path.join(config.TEMP_DIR, filename)
    if os.path.exists(audio_path):
        return send_file(audio_path, mimetype="audio/wav")
    return jsonify({"error": "Audio file not found"}), 404


@app.route("/audio_stream/<request_id>", methods=["GET"])
def stream_audio_direct(request_id):
    """Stream audio directly from memory cache"""
    if request_id in audio_memory_cache:
        audio_data, sample_rate = audio_memory_cache[request_id]
        
        # Convert numpy array to WAV bytes in memory
        import io
        wav_buffer = io.BytesIO()
        sf.write(wav_buffer, audio_data, sample_rate, format='WAV')
        wav_bytes = wav_buffer.getvalue()
        
        return Response(
            wav_bytes,
            mimetype='audio/wav',
            headers={
                'Content-Length': str(len(wav_bytes)),
                'Accept-Ranges': 'bytes',
                'Cache-Control': 'no-cache',
            }
        )
    return jsonify({"error": "Audio not found"}), 404


if __name__ == "__main__":
    print(f"Starting Voice Assistant server on {config.SERVER_HOST}:{config.SERVER_PORT}")
    print(f"Whisper model: {config.WHISPER_MODEL}")
    print(f"Gemma model: {config.GEMMA_MODEL_PATH}")
    print(f"Kokoro voice: {config.KOKORO_VOICE}")
    print("\n" + "="*60)
    print("Loading models (this may take a minute on first run)...")
    print("="*60)
    
    # Eager model loading - pre-load all models before accepting requests
    try:
        print("\n[1/3] Loading Whisper model...")
        load_whisper_model()
        # Trigger actual model load by doing a dummy transcribe
        import tempfile
        import numpy as np
        # Create a tiny dummy audio file to trigger model load
        dummy_audio = np.zeros(16000, dtype=np.float32)  # 1 second of silence
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            sf.write(tmp.name, dummy_audio, config.WHISPER_SAMPLE_RATE)
            mlx_whisper.transcribe(tmp.name, path_or_hf_repo=config.WHISPER_MODEL, verbose=False)
        print("✓ Whisper model loaded")
        
        print("\n[2/3] Loading Gemma LLM model...")
        load_gemma_model()
        print("✓ Gemma model loaded")
        
        print("\n[3/3] Loading Kokoro TTS pipeline...")
        load_tts_pipeline()
        # Pre-load voice to avoid delay on first synthesis
        if tts_pipeline:
            tts_pipeline.load_voice(config.KOKORO_VOICE)
        print("✓ Kokoro TTS pipeline loaded")
        
        print("\n[4/4] Warming up LLM...")
        dummy_response = generate_llm_response("test", stream=False)
        print(f"✓ LLM warmed up (generated {len(dummy_response)} chars)")
        
        print("\n" + "="*60)
        print("✓ All models loaded and warmed up successfully!")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n⚠️  Warning: Error during eager model loading: {e}")
        print("Models will be loaded lazily on first request.")
        traceback.print_exc()
    
    app.run(
        host=config.SERVER_HOST,
        port=config.SERVER_PORT,
        debug=config.DEBUG,
    )
