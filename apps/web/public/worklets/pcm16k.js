// AudioWorklet: microphone -> 16 kHz mono PCM16 frames (20 ms) + RMS level, posted to the main thread.
// Downsampling uses a one-pole low-pass (anti-alias) followed by linear interpolation.
class Pcm16k extends AudioWorkletProcessor {
  constructor() {
    super();
    this.ratio = sampleRate / 16000;
    this.pos = 0;
    this.prev = 0;
    this.lp = 0;
    this.alpha = Math.min(1, (2 * Math.PI * 7000) / sampleRate / (1 + (2 * Math.PI * 7000) / sampleRate));
    this.frame = new Int16Array(320);
    this.n = 0;
    this.sumSq = 0;
    this.muted = false;
    this.port.onmessage = (e) => { if (e.data && typeof e.data.muted === "boolean") this.muted = e.data.muted; };
  }
  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (!ch) return true;
    for (let i = 0; i < ch.length; i++) {
      this.lp += this.alpha * (ch[i] - this.lp);
      const cur = this.lp;
      // emit output samples that fall between prev and cur
      while (this.pos < 1) {
        const v = this.prev + (cur - this.prev) * this.pos;
        const s = this.muted ? 0 : Math.max(-1, Math.min(1, v));
        this.frame[this.n++] = s * 32767;
        this.sumSq += s * s;
        if (this.n === 320) {
          const rms = Math.sqrt(this.sumSq / 320);
          this.port.postMessage({ pcm: this.frame.buffer, rms }, [this.frame.buffer]);
          this.frame = new Int16Array(320);
          this.n = 0;
          this.sumSq = 0;
        }
        this.pos += this.ratio;
      }
      this.pos -= 1;
      this.prev = cur;
    }
    return true;
  }
}
registerProcessor("pcm16k", Pcm16k);
