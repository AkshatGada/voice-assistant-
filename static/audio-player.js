// Progressive Audio Player
// Handles streaming audio playback for low-latency responses

let audioQueue = [];
let isPlaying = false;
let audioContext = null;
let currentSource = null;

function initAudioContext() {
    if (!audioContext) {
        audioContext = new AudioContext();
    }
    return audioContext;
}

function playAudioChunk(audioData) {
    audioQueue.push(audioData);
    if (!isPlaying) {
        playNextChunk();
    }
}

async function playNextChunk() {
    if (audioQueue.length === 0) {
        isPlaying = false;
        return;
    }
    
    isPlaying = true;
    const chunk = audioQueue.shift();
    
    try {
        const ctx = initAudioContext();
        const audioBuffer = await ctx.decodeAudioData(chunk.slice(0));
        const source = ctx.createBufferSource();
        source.buffer = audioBuffer;
        source.connect(ctx.destination);
        
        currentSource = source;
        
        source.onended = () => {
            currentSource = null;
            playNextChunk();
        };
        
        source.start();
    } catch (err) {
        console.error('Error playing audio chunk:', err);
        isPlaying = false;
        // Try next chunk
        if (audioQueue.length > 0) {
            playNextChunk();
        }
    }
}

function stopAudioPlayback() {
    if (currentSource) {
        try {
            currentSource.stop();
        } catch (e) {
            // Already stopped
        }
        currentSource = null;
    }
    audioQueue = [];
    isPlaying = false;
}

function pauseAudioPlayback() {
    if (currentSource && audioContext && audioContext.state === 'running') {
        audioContext.suspend();
    }
}

function resumeAudioPlayback() {
    if (audioContext && audioContext.state === 'suspended') {
        audioContext.resume();
    }
}
