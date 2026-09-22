// Microphone capture with automatic end-of-speech detection.
//
// The turn ends when the caller stops talking rather than when they release a
// button. That is the natural way to speak to a phone system, and it is only
// workable if two failure modes are handled explicitly.
//
// Ending too early. A naive silence timer fires during the pause between
// clauses. The detector therefore requires MIN_SPEECH_MS of actual speech
// before it will arm at all, and then a continuous SILENCE_MS below threshold -
// not an average, which a single quiet syllable would satisfy.
//
// Never ending. A noisy room keeps the level above threshold forever. The
// threshold is calibrated against that room's own noise floor during the first
// CALIBRATE_MS rather than being a fixed constant, there is a hard MAX_MS cap,
// and the control stays clickable so a person can always send immediately.
//
// The level is reported to the caller on every frame so the interface can show
// what the microphone is actually hearing. A mic that is picking nothing up
// should look like a mic that is picking nothing up.

export const VoiceCapture = (() => {
  const CALIBRATE_MS = 350;
  const MIN_SPEECH_MS = 300;
  const SILENCE_MS = 1200;
  const MAX_MS = 30000;
  const FLOOR_MULTIPLIER = 2.2;
  const ABS_FLOOR = 0.011;

  class Capture {
    constructor({ onLevel, onState, onResult, onError }) {
      this.onLevel = onLevel || (() => {});
      this.onState = onState || (() => {});
      this.onResult = onResult || (() => {});
      this.onError = onError || (() => {});
      this.active = false;
    }

    async start() {
      if (this.active) return;
      let stream;
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
        });
      } catch (err) {
        this.onError(err);
        return;
      }

      this.active = true;
      this.stream = stream;
      this.chunks = [];
      this.startedAt = performance.now();
      this.noiseSum = 0;
      this.noiseCount = 0;
      this.threshold = ABS_FLOOR;
      this.speechMs = 0;
      this.silenceMs = 0;
      this.lastFrame = performance.now();
      this.phase = "calibrating";

      this.ctx = new (window.AudioContext || window.webkitAudioContext)();
      const source = this.ctx.createMediaStreamSource(stream);
      this.analyser = this.ctx.createAnalyser();
      this.analyser.fftSize = 1024;
      this.analyser.smoothingTimeConstant = 0.35;
      source.connect(this.analyser);
      this.buffer = new Float32Array(this.analyser.fftSize);

      const mime = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
        ? "audio/webm;codecs=opus" : "audio/webm";
      this.recorder = new MediaRecorder(stream, { mimeType: mime });
      this.recorder.ondataavailable = (e) => { if (e.data.size) this.chunks.push(e.data); };
      this.recorder.onstop = () => this._finish();
      this.recorder.start();

      this.onState("calibrating");
      this._loop();
    }

    _rms() {
      this.analyser.getFloatTimeDomainData(this.buffer);
      let sum = 0;
      for (let i = 0; i < this.buffer.length; i++) sum += this.buffer[i] * this.buffer[i];
      return Math.sqrt(sum / this.buffer.length);
    }

    _loop() {
      if (!this.active) return;
      const now = performance.now();
      const dt = now - this.lastFrame;
      this.lastFrame = now;

      const level = this._rms();
      this.onLevel(level, this.threshold);

      const elapsed = now - this.startedAt;

      if (this.phase === "calibrating") {
        this.noiseSum += level;
        this.noiseCount += 1;
        if (elapsed >= CALIBRATE_MS) {
          const floor = this.noiseSum / Math.max(this.noiseCount, 1);
          this.threshold = Math.max(floor * FLOOR_MULTIPLIER, ABS_FLOOR);
          this.phase = "waiting";
          this.onState("waiting");
        }
      } else {
        if (level > this.threshold) {
          this.speechMs += dt;
          this.silenceMs = 0;
          if (this.phase === "waiting" && this.speechMs >= MIN_SPEECH_MS) {
            this.phase = "speaking";
            this.onState("speaking");
          }
        } else if (this.phase === "speaking") {
          this.silenceMs += dt;
          if (this.silenceMs >= SILENCE_MS) { this.stop("silence"); return; }
        }
      }

      if (elapsed >= MAX_MS) { this.stop("timeout"); return; }
      this.raf = requestAnimationFrame(() => this._loop());
    }

    stop(reason = "manual") {
      if (!this.active) return;
      this.active = false;
      this.reason = reason;
      if (this.raf) cancelAnimationFrame(this.raf);
      this.onLevel(0, this.threshold);
      if (this.recorder && this.recorder.state !== "inactive") this.recorder.stop();
      else this._finish();
    }

    _finish() {
      if (this.stream) this.stream.getTracks().forEach((t) => t.stop());
      if (this.ctx && this.ctx.state !== "closed") this.ctx.close();
      const blob = new Blob(this.chunks, { type: "audio/webm" });
      // Nothing was said: a mis-click, or the caller never spoke loudly enough
      // for the detector to arm. Reported rather than sent, so the interface can
      // say so instead of the agent answering silence.
      const spoke = this.speechMs >= MIN_SPEECH_MS && blob.size > 1500;
      this.onState("stopped");
      this.onResult({ blob, spoke, reason: this.reason, speechMs: Math.round(this.speechMs) });
    }
  }

  return Capture;
})();
