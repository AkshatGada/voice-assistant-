// VAD Configuration
// Voice Activity Detection parameters for browser-side speech detection

export const VAD_CONFIG = {
    // Speech detection thresholds
    positiveSpeechThreshold: 0.8,  // Higher = less sensitive (requires louder speech)
    negativeSpeechThreshold: 0.5,  // Lower = more sensitive to silence
    
    // Timing parameters
    redemptionFrames: 8,   // Frames of silence before ending (8 frames ≈ 400ms at 20fps)
    minSpeechFrames: 3,   // Minimum speech duration in frames
    
    // Audio processing
    sampleRate: 16000,     // Sample rate for VAD processing
    
    // CDN URLs for VAD resources
    workletURL: "https://cdn.jsdelivr.net/npm/@ricky0123/vad-web@0.0.19/dist/mic-vad/vad.worklet.bundle.min.js",
    modelURL: "https://cdn.jsdelivr.net/npm/@ricky0123/vad-web@0.0.19/dist/silero_vad.onnx"
};
