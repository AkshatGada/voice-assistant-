// WebSocket Handler for Voice Assistant
// Manages WebSocket connection and streaming audio

// Make variables globally accessible
window.socket = null;
window.sessionId = null;
window.useWebSocket = true;  // Feature flag - can fallback to HTTP

let socket = window.socket;
let sessionId = window.sessionId;
let useWebSocket = window.useWebSocket;

function initWebSocket() {
    if (typeof io === 'undefined') {
        console.warn('Socket.IO not loaded, falling back to HTTP');
        useWebSocket = false;
        return;
    }
    
    socket = io('http://localhost:3000');
    window.socket = socket;  // Make globally accessible
    
    socket.on('connect', () => {
        console.log('WebSocket connected');
    });
    
    socket.on('connected', (data) => {
        sessionId = data.session_id;
        window.sessionId = sessionId;  // Make globally accessible
        console.log('Session ID:', sessionId);
    });
    
    socket.on('transcription', (data) => {
        if (typeof transcribedTextDiv !== 'undefined') {
            transcribedTextDiv.textContent = data.text;
        }
        console.log('Transcription:', data.text);
    });
    
    socket.on('audio_chunk', (data) => {
        // Decode and play audio chunk immediately
        const audioData = base64ToArrayBuffer(data.audio);
        playAudioChunk(audioData);
        
        // Update response text progressively
        if (typeof responseTextDiv !== 'undefined') {
            if (data.is_filler) {
                responseTextDiv.textContent = data.text;
            } else {
                responseTextDiv.textContent += data.text;
            }
        }
    });
    
    socket.on('response_complete', (data) => {
        if (typeof statusDiv !== 'undefined') {
            statusDiv.className = 'status idle';
            statusDiv.textContent = 'Complete';
        }
        if (typeof resetUI !== 'undefined') {
            resetUI();
        }
        console.log('Response complete:', data.full_text);
    });
    
    socket.on('error', (data) => {
        console.error('WebSocket error:', data.message);
        if (typeof showError !== 'undefined') {
            showError(data.message);
        }
    });
    
    socket.on('disconnect', () => {
        console.log('WebSocket disconnected');
        useWebSocket = false;  // Fallback to HTTP
        window.useWebSocket = false;  // Update global
    });
}

function base64ToArrayBuffer(base64) {
    const binaryString = atob(base64);
    const bytes = new Uint8Array(binaryString.length);
    for (let i = 0; i < binaryString.length; i++) {
        bytes[i] = binaryString.charCodeAt(i);
    }
    return bytes.buffer;
}

function sendAudioChunk(chunk) {
    if (!socket || !socket.connected || !useWebSocket) {
        return false;
    }
    
    // Convert Blob to base64
    const reader = new FileReader();
    reader.onload = () => {
        const arrayBuffer = reader.result;
        const base64 = btoa(
            new Uint8Array(arrayBuffer)
                .reduce((data, byte) => data + String.fromCharCode(byte), '')
        );
        socket.emit('audio_chunk', { chunk: base64 });
    };
    reader.readAsArrayBuffer(chunk);
    return true;
}

function sendAudioEnd() {
    if (!socket || !socket.connected || !useWebSocket) {
        return false;
    }
    
    socket.emit('audio_end', {});
    return true;
}

// Initialize WebSocket on page load
if (typeof window !== 'undefined') {
    window.addEventListener('load', () => {
        initWebSocket();
    });
}
