// vad-processor.js
class VADProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        this.buffer = [];
        this.isSpeaking = false;
        this.silenceFrames = 0;
        this.threshold = 0.05; // Sensibilité du micro
        this.silenceGracePeriod = 50; // ~1.5s à 16kHz
    }

    process(inputs, outputs, parameters) {
        const input = inputs[0][0];
        if (!input) return true;

        const volume = Math.max(...input.map(Math.abs));

        if (volume > this.threshold) {
            if (!this.isSpeaking) {
                this.isSpeaking = true;
                this.port.postMessage('START');
            }
            this.silenceFrames = 0;
        } else if (this.isSpeaking) {
            this.silenceFrames++;
            if (this.silenceFrames > this.silenceGracePeriod) {
                this.isSpeaking = false;
                this.port.postMessage('STOP');
            }
        }
        return true;
    }
}
registerProcessor('vad-processor', VADProcessor);