"""Voice Assistant Backend - STT → LLM (Gemini) → TTS Pipeline"""

import os
import tempfile
import time
import traceback
import re

import mlx_whisper
import numpy as np
import soundfile as sf
from flask import Flask, jsonify, request, send_file, Response
from flask_cors import CORS
from kokoro import KPipeline
import google.generativeai as genai

import config

app = Flask(__name__, static_folder="static", static_url_path="")
CORS(app)

# Global model holders (lazy loading)
whisper_model_loaded = False
tts_pipeline = None
gemini_model = None

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


def load_gemini_model():
    """Initialize Gemini Flash model"""
    global gemini_model
    
    if gemini_model is not None:
        return
    
    try:
        api_key = config.GOOGLE_API_KEY
        if not api_key:
            print("⚠️ Warning: GOOGLE_API_KEY not set in config.py or env.")
            return

        genai.configure(api_key=api_key)
        gemini_model = genai.GenerativeModel('gemini-1.5-flash')
        print("Gemini Flash model initialized successfully")
    except Exception as e:
        print(f"Error initializing Gemini: {e}")
        traceback.print_exc()


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


def _generate_llm_response_streaming(prompt: str):
    """Internal generator function for streaming Gemini responses"""
    global gemini_model
    
    if gemini_model is None:
        load_gemini_model()
        if gemini_model is None:
            yield "Error", "Gemini API key not configured."
            return

    full_response = ""
    try:
        # Construct the full prompt including system instructions
        combined_prompt = f"{config.SYSTEM_PROMPT}\n\nUser: {prompt}\nAssistant:"
        
        response = gemini_model.generate_content(
            combined_prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=config.LLM_MAX_TOKENS,
                temperature=config.LLM_TEMPERATURE,
            ),
            stream=True
        )
        
        for chunk in response:
            token = chunk.text
            full_response += token
            yield token, full_response.strip()
            
    except Exception as e:
        print(f"Gemini streaming error: {e}")
        yield "Error", f"Failed to generate: {e}"


def generate_with_streaming_tts(transcribed_text):
    """Generate LLM response with parallel TTS streaming"""
    load_tts_pipeline()
    
    sentence_buffer = ""
    full_text = ""
    audio_chunks = []
    
    # Stream Gemini tokens
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
    """Generate response using Gemini Flash"""
    if stream:
        return _generate_llm_response_streaming(prompt)
    
    global gemini_model
    if gemini_model is None:
        load_gemini_model()
        if gemini_model is None:
            return "Gemini API key not configured."

    try:
        combined_prompt = f"{config.SYSTEM_PROMPT}\n\nUser: {prompt}\nAssistant:"
        response = gemini_model.generate_content(
            combined_prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=config.LLM_MAX_TOKENS,
                temperature=config.LLM_TEMPERATURE,
            )
        )
        return response.text.strip()
    except Exception as e:
        print(f"Gemini generation error: {e}")
        return f"Error: {e}"


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
        "gemini_ready": gemini_model is not None,
        "tts_ready": tts_pipeline is not None,
        "all_models_ready": whisper_model_loaded and gemini_model is not None and tts_pipeline is not None,
    })


@app.route("/transcribe", methods=["POST"])
def transcribe():
    """Transcribe audio to text using Whisper"""
    try:
        if "audio" not in request.files:
            return jsonify({"error": "No audio file provided"}), 400
        
        audio_file = request.files["audio"]
        
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=".wav", dir=config.TEMP_DIR
        ) as tmp_file:
            audio_file.save(tmp_file.name)
            temp_path = tmp_file.name
        
        try:
            load_whisper_model()
            result = mlx_whisper.transcribe(
                temp_path,
                path_or_hf_repo=config.WHISPER_MODEL,
                verbose=False,
                condition_on_previous_text=False,
            )
            
            return jsonify({
                "text": result["text"].strip(),
                "language": result.get("language", "en"),
            })
        finally:
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
        
        audio_chunks = []
        for result in tts_pipeline(text, voice=config.KOKORO_VOICE, speed=config.KOKORO_SPEED):
            if result.audio is not None:
                audio_chunks.append(result.audio.numpy())
        
        if not audio_chunks:
            return jsonify({"error": "No audio generated"}), 500
        
        audio_array = np.concatenate(audio_chunks)
        
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
        return jsonify({"error": str(e)}), 500


@app.route("/chat", methods=["POST"])
def chat():
    """Complete pipeline: audio → text → LLM → audio"""
    audio_path = None
    response_path = None
    pipeline_start_time = time.time()
    
    try:
        if "audio" not in request.files:
            return jsonify({"error": "No audio file provided"}), 400
        
        audio_file = request.files["audio"]
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=".webm", dir=config.TEMP_DIR
        ) as tmp_file:
            audio_file.save(tmp_file.name)
            audio_path = tmp_file.name
        
        # Step 2: Transcribe
        load_whisper_model()
        transcription_result = mlx_whisper.transcribe(
            audio_path,
            path_or_hf_repo=config.WHISPER_MODEL,
            verbose=False,
            condition_on_previous_text=False,
        )
        transcribed_text = transcription_result.get("text", "").strip()
        
        if not transcribed_text:
            return jsonify({"error": "No speech detected"}), 400
            
        # Step 3 & 4: Generate + TTS
        llm_response, audio_array = generate_with_streaming_tts(transcribed_text)
        
        # Cache audio
        import uuid
        request_id = str(uuid.uuid4())[:8]
        audio_memory_cache[request_id] = (audio_array, config.KOKORO_SAMPLE_RATE)
        
        return jsonify({
            "transcribed_text": transcribed_text,
            "llm_response": llm_response,
            "audio_url": f"/audio_stream/{request_id}",
            "total_latency": time.time() - pipeline_start_time
        })
        
    except Exception as e:
        print(f"Chat pipeline error: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        if audio_path and os.path.exists(audio_path):
            os.unlink(audio_path)


@app.route("/audio_stream/<request_id>", methods=["GET"])
def stream_audio_direct(request_id):
    """Stream audio directly from memory cache"""
    if request_id in audio_memory_cache:
        audio_data, sample_rate = audio_memory_cache[request_id]
        import io
        wav_buffer = io.BytesIO()
        sf.write(wav_buffer, audio_data, sample_rate, format='WAV')
        return Response(wav_buffer.getvalue(), mimetype='audio/wav')
    return jsonify({"error": "Audio not found"}), 404


if __name__ == "__main__":
    print(f"Starting Voice Assistant server on {config.SERVER_HOST}:{config.SERVER_PORT}")
    
    # Eager model loading
    try:
        print("\n[1/3] Loading Whisper model...")
        load_whisper_model()
        print("\n[2/3] Initializing Gemini Flash...")
        load_gemini_model()
        print("\n[3/3] Loading Kokoro TTS pipeline...")
        load_tts_pipeline()
        print("\n✓ Models ready")
    except Exception as e:
        print(f"\n⚠️ Warning: Error during eager model loading: {e}")
    
    app.run(host=config.SERVER_HOST, port=config.SERVER_PORT, debug=config.DEBUG)
