// Voice Assistant Frontend

const recordButton = document.getElementById('recordButton');
const statusDiv = document.getElementById('status');
const transcribedTextDiv = document.getElementById('transcribedText');
const responseTextDiv = document.getElementById('responseText');
const audioPlayer = document.getElementById('audioPlayer');
const errorDiv = document.getElementById('error');

let mediaRecorder = null;
let audioChunks = [];
let isRecording = false;
let recordingStartTime = null;
const MIN_RECORDING_DURATION = 1000; // Minimum 1 second recording for better transcription

// VAD (Voice Activity Detection) variables
let vad = null;
let vadActive = false;
let vadEnabled = true;

// Check for browser support
if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    showError('Your browser does not support audio recording. Please use Chrome, Firefox, or Edge.');
    recordButton.disabled = true;
}

// Request microphone access
navigator.mediaDevices.getUserMedia({ audio: true })
    .then(stream => {
        console.log('Microphone access granted');
        setupMediaRecorder(stream);
        initVAD(stream);
    })
    .catch(err => {
        console.error('Microphone access denied:', err);
        showError('Microphone access is required. Please allow microphone access and refresh the page.');
        recordButton.disabled = true;
    });

// Initialize VAD (Voice Activity Detection)
async function initVAD(stream) {
    // Wait for MicVAD to be available
    if (typeof window.MicVAD === 'undefined') {
        console.warn('MicVAD not loaded, retrying...');
        setTimeout(() => initVAD(stream), 500);
        return;
    }
    
    try {
        vad = await window.MicVAD.new({
            onSpeechStart: () => {
                console.log("Speech detected - starting recording");
                if (!isRecording) {
                    startRecording();
                }
            },
            onSpeechEnd: (audio) => {
                console.log("Speech ended - auto-stopping");
                if (isRecording && vadEnabled) {
                    // Check minimum duration
                    if (recordingStartTime && (Date.now() - recordingStartTime) >= MIN_RECORDING_DURATION) {
                        stopRecording();
                    }
                }
            },
            positiveSpeechThreshold: 0.8,  // Sensitivity for speech detection
            negativeSpeechThreshold: 0.5,  // Sensitivity for silence detection
            redemptionFrames: 8,  // Wait 8 frames (400ms) of silence before ending
            minSpeechFrames: 3,   // Minimum speech duration (frames)
            workletURL: "https://cdn.jsdelivr.net/npm/@ricky0123/vad-web@0.0.19/dist/mic-vad/vad.worklet.bundle.min.js",
            modelURL: "https://cdn.jsdelivr.net/npm/@ricky0123/vad-web@0.0.19/dist/silero_vad.onnx",
            sampleRate: 16000
        });
        
        vadActive = true;
        console.log('VAD initialized successfully');
    } catch (err) {
        console.error('Failed to initialize VAD:', err);
        // Continue without VAD - manual stop button will still work
        vadActive = false;
    }
}

// VAD toggle handler
const vadToggle = document.getElementById('vadToggle');
if (vadToggle) {
    vadToggle.addEventListener('change', (e) => {
        vadEnabled = e.target.checked;
        console.log('VAD enabled:', vadEnabled);
    });
}

function setupMediaRecorder(stream) {
    // Use MediaRecorder with WAV format if available, otherwise use webm
    const options = {
        mimeType: 'audio/webm;codecs=opus',
        audioBitsPerSecond: 128000
    };
    
    mediaRecorder = new MediaRecorder(stream, options);
    
    mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
            audioChunks.push(event.data);
            
            // Stream chunk via WebSocket if available
            if (typeof sendAudioChunk !== 'undefined' && typeof window.useWebSocket !== 'undefined' && window.useWebSocket) {
                sendAudioChunk(event.data);
            }
        }
    };
    
    mediaRecorder.onstop = () => {
        // If using WebSocket, signal end
        if (typeof sendAudioEnd !== 'undefined' && typeof window.useWebSocket !== 'undefined' && window.useWebSocket && typeof window.socket !== 'undefined' && window.socket && window.socket.connected) {
            sendAudioEnd();
        } else {
            // Fallback to HTTP
            processRecording();
        }
    };
}

recordButton.addEventListener('click', () => {
    if (!isRecording) {
        startRecording();
    } else {
        stopRecording();
    }
});

function startRecording() {
    if (!mediaRecorder) {
        showError('MediaRecorder not initialized');
        return;
    }
    
    audioChunks = [];
    isRecording = true;
    recordButton.classList.add('recording');
    recordButton.innerHTML = '⏹️<br>Stop<br>Recording';
    statusDiv.className = 'status recording';
    statusDiv.textContent = vadEnabled ? 'Recording... (Auto-stop enabled)' : 'Recording...';
    transcribedTextDiv.textContent = '-';
    responseTextDiv.textContent = '-';
    hideError();
    
    try {
        recordingStartTime = Date.now();
        // Start recording with timeslice to ensure data is captured
        mediaRecorder.start(100); // Collect data every 100ms
        
        // Start VAD monitoring if enabled
        if (vad && vadActive && vadEnabled) {
            vad.start();
            console.log('VAD monitoring started');
        }
        
        console.log('Recording started');
    } catch (err) {
        console.error('Error starting recording:', err);
        showError('Failed to start recording: ' + err.message);
        isRecording = false;
        recordButton.classList.remove('recording');
        recordButton.innerHTML = '🎤<br>Press to<br>Record';
        statusDiv.className = 'status idle';
        statusDiv.textContent = 'Ready to record';
    }
}

function stopRecording() {
    if (!mediaRecorder || !isRecording) {
        return;
    }
    
    // Check minimum recording duration
    if (recordingStartTime && (Date.now() - recordingStartTime) < MIN_RECORDING_DURATION) {
        showError(`Please record for at least ${MIN_RECORDING_DURATION/1000} seconds`);
        // Continue recording
        return;
    }
    
    // Stop VAD monitoring
    if (vad && vadActive) {
        vad.pause();
        console.log('VAD monitoring paused');
    }
    
    // Mark when user finished speaking (end of recording)
    const userFinishedSpeakingTime = Date.now();
    
    isRecording = false;
    recordButton.classList.remove('recording');
    recordButton.innerHTML = '🎤<br>Press to<br>Record';
    statusDiv.className = 'status processing';
    statusDiv.textContent = 'Processing...';
    
    try {
        // Request remaining data before stopping
        mediaRecorder.requestData();
        mediaRecorder.stop();
        const duration = recordingStartTime ? (Date.now() - recordingStartTime) : 0;
        console.log(`Recording stopped after ${duration}ms, chunks:`, audioChunks.length);
        console.log(`⏱️  User finished speaking at: ${userFinishedSpeakingTime}`);
        recordingStartTime = null;
        
        // Store the timestamp for latency calculation
        window.userFinishedSpeakingTime = userFinishedSpeakingTime;
    } catch (err) {
        console.error('Error stopping recording:', err);
        showError('Failed to stop recording: ' + err.message);
        resetUI();
    }
}

async function processRecording() {
    try {
        if (audioChunks.length === 0) {
            showError('No audio data recorded. Please try again.');
            resetUI();
            return;
        }
        
        // Convert audio chunks to blob
        const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
        console.log('Audio blob size:', audioBlob.size, 'bytes');
        
        if (audioBlob.size < 1000) {
            showError('Recording too short. Please speak for at least 1 second.');
            resetUI();
            return;
        }
        
        // Convert to WAV format using Web Audio API
        const wavBlob = await convertToWav(audioBlob);
        
        // Send to backend
        await sendToBackend(wavBlob);
        
    } catch (err) {
        console.error('Error processing recording:', err);
        showError('Error processing audio: ' + err.message);
        resetUI();
    }
}

async function convertToWav(audioBlob) {
    // For now, send webm directly - backend will handle conversion if needed
    // In production, you might want to convert to WAV here using Web Audio API
    return audioBlob;
}

async function sendToBackend(audioBlob) {
    const formData = new FormData();
    formData.append('audio', audioBlob, 'recording.webm');
    
    // Check if streaming is enabled (default to non-streaming for now)
    const useStreaming = false; // Can be enabled via config later
    
    if (useStreaming) {
        // Streaming mode - use Server-Sent Events
        await sendToBackendStreaming(audioBlob);
    } else {
        // Non-streaming mode - original implementation
        await sendToBackendBlocking(audioBlob);
    }
}

async function sendToBackendBlocking(audioBlob) {
    const formData = new FormData();
    formData.append('audio', audioBlob, 'recording.webm');
    
    try {
        const response = await fetch('/chat', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || 'Server error');
        }
        
        const contentType = response.headers.get('content-type');
        
        if (contentType && contentType.includes('application/json')) {
        const data = await response.json();
        
        console.log('Server response:', data);
        
        if (data.error) {
            // Show more detailed error if available
            const errorMsg = data.debug_info 
                ? `${data.error} (File size: ${data.debug_info.file_size} bytes, Language: ${data.debug_info.language})`
                : data.error;
            throw new Error(errorMsg);
        }
            
            // Display transcribed text
            transcribedTextDiv.textContent = data.transcribed_text || '-';
            
            // Display LLM response
            responseTextDiv.textContent = data.llm_response || '-';
            
            // Display latency information if available
            if (data.timings) {
                const latencySection = document.getElementById('latencySection');
                const latencyInfo = document.getElementById('latencyInfo');
                if (latencySection && latencyInfo) {
                    latencySection.style.display = 'block';
                    latencyInfo.innerHTML = `
                        <strong>Backend Processing:</strong><br>
                        STT: ${data.timings.stt_ms}ms | 
                        LLM: ${data.timings.llm_ms}ms | 
                        TTS: ${data.timings.tts_ms}ms<br>
                        <strong>Total Backend: ${data.timings.total_ms}ms</strong><br>
                        <em>End-to-end latency will be shown when audio starts playing...</em>
                    `;
                }
            }
            
            // Play audio response if available
            if (data.audio_url) {
                const audioLoadStartTime = Date.now();
                
                // Track when audio actually starts playing (first word of Jarvis)
                audioPlayer.onplay = () => {
                    const jarvisStartsSpeakingTime = Date.now();
                    const endToEndLatency = jarvisStartsSpeakingTime - window.userFinishedSpeakingTime;
                    
                    console.log(`\n${'='.repeat(60)}`);
                    console.log(`⏱️  END-TO-END LATENCY MEASUREMENT`);
                    console.log(`   User finished speaking: ${new Date(window.userFinishedSpeakingTime).toISOString()}`);
                    console.log(`   Jarvis starts speaking: ${new Date(jarvisStartsSpeakingTime).toISOString()}`);
                    console.log(`   ${'-'.repeat(60)}`);
                    console.log(`   🎯 TOTAL LATENCY: ${endToEndLatency}ms (${(endToEndLatency/1000).toFixed(3)}s)`);
                    if (data.timings) {
                        console.log(`   Backend breakdown:`);
                        console.log(`     STT: ${data.timings.stt_ms}ms`);
                        console.log(`     LLM: ${data.timings.llm_ms}ms`);
                        console.log(`     TTS: ${data.timings.tts_ms}ms`);
                        console.log(`     Backend Total: ${data.timings.total_ms}ms`);
                        const networkLatency = endToEndLatency - data.timings.total_ms;
                        console.log(`     Network/Other: ~${networkLatency.toFixed(1)}ms`);
                    }
                    console.log(`${'='.repeat(60)}\n`);
                    
                    // Update latency display in UI
                    const latencyInfo = document.getElementById('latencyInfo');
                    if (latencyInfo && data.timings) {
                        const networkLatency = Math.max(0, endToEndLatency - data.timings.total_ms);
                        latencyInfo.innerHTML = `
                            <strong>🎯 End-to-End Latency: ${endToEndLatency}ms (${(endToEndLatency/1000).toFixed(3)}s)</strong><br>
                            <strong>Backend Processing:</strong><br>
                            STT: ${data.timings.stt_ms}ms | 
                            LLM: ${data.timings.llm_ms}ms | 
                            TTS: ${data.timings.tts_ms}ms<br>
                            Backend Total: ${data.timings.total_ms}ms | 
                            Network/Other: ~${networkLatency.toFixed(0)}ms
                        `;
                    }
                    
                    // Update status with latency info
                    statusDiv.textContent = `Playing response... (Latency: ${(endToEndLatency/1000).toFixed(2)}s)`;
                };
                
                audioPlayer.src = data.audio_url + '?t=' + Date.now(); // Cache bust
                audioPlayer.style.display = 'block';
                statusDiv.className = 'status playing';
                statusDiv.textContent = 'Loading audio...';
                
                audioPlayer.onended = () => {
                    statusDiv.className = 'status idle';
                    statusDiv.textContent = 'Ready to record';
                };
                
                audioPlayer.onerror = (err) => {
                    console.error('Audio playback error:', err);
                    showError('Error playing audio response');
                    statusDiv.className = 'status idle';
                    statusDiv.textContent = 'Ready to record';
                };
                
                try {
                    await audioPlayer.play();
                } catch (playErr) {
                    console.error('Play error:', playErr);
                    showError('Could not play audio. Click the play button manually.');
                    statusDiv.className = 'status idle';
                    statusDiv.textContent = 'Ready to record';
                }
            } else {
                statusDiv.className = 'status idle';
                statusDiv.textContent = 'Ready to record';
            }
        } else {
            throw new Error('Unexpected response format from server');
        }
        
    } catch (err) {
        console.error('Error sending to backend:', err);
        showError('Error communicating with server: ' + err.message);
        resetUI();
    }
}

async function sendToBackendStreaming(audioBlob) {
    // Streaming implementation - placeholder for future enhancement
    // This would use EventSource or fetch with streaming to receive audio chunks
    console.log('Streaming mode not yet fully implemented, falling back to blocking');
    await sendToBackendBlocking(audioBlob);
}

function resetUI() {
    isRecording = false;
    recordButton.classList.remove('recording');
    recordButton.innerHTML = '🎤<br>Press to<br>Record';
    statusDiv.className = 'status idle';
    statusDiv.textContent = 'Ready to record';
}

function showError(message) {
    errorDiv.textContent = message;
    errorDiv.classList.add('show');
}

function hideError() {
    errorDiv.classList.remove('show');
}

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    if (mediaRecorder && isRecording) {
        mediaRecorder.stop();
    }
});
