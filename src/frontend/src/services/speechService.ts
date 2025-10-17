// Simple speech service using Web Speech APIs available in modern browsers.
// Falls back gracefully if APIs are unavailable.

export type SpeechSupport = {
  hasTTS: boolean;
  hasSTT: boolean;
};

export function getSpeechSupport(): SpeechSupport {
  const hasTTS = typeof window !== "undefined" && "speechSynthesis" in window;
  const hasSTT = typeof window !== "undefined" &&
    ("SpeechRecognition" in (window as any) ||
      "webkitSpeechRecognition" in (window as any));
  return { hasTTS, hasSTT };
}

export function speakText(text: string, voiceName?: string): void {
  if (!text) return;
  if (typeof window === "undefined" || !("speechSynthesis" in window)) return;
  const utterance = new SpeechSynthesisUtterance(text);
  if (voiceName) {
    const voice = window.speechSynthesis.getVoices().find(v => v.name === voiceName);
    if (voice) utterance.voice = voice;
  }
  // Reasonable defaults for clarity
  utterance.rate = 1.0;
  utterance.pitch = 1.0;
  utterance.volume = 1.0;
  window.speechSynthesis.cancel();
  window.speechSynthesis.speak(utterance);
}

export type STTOptions = {
  lang?: string;
  interimResults?: boolean;
};

export function startTranscription(
  onResult: (finalText: string, isFinal: boolean) => void,
  onError?: (err: unknown) => void,
  options?: STTOptions
): () => void {
  const RecognitionCtor = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
  if (!RecognitionCtor) {
    onError?.(new Error("SpeechRecognition API not supported"));
    return () => {};
  }
  const recognition = new RecognitionCtor();
  recognition.lang = options?.lang ?? navigator.language ?? "en-US";
  recognition.interimResults = options?.interimResults ?? false;
  recognition.continuous = false;

  recognition.onresult = (event: any) => {
    const result = event.results[event.results.length - 1];
    const transcript = result[0]?.transcript ?? "";
    const isFinal = result.isFinal;
    onResult(transcript, isFinal);
  };

  recognition.onerror = (ev: any) => {
    onError?.(ev?.error ?? ev);
  };

  recognition.onend = () => {
    // no-op; consumer controls lifecycle via returned stop function
  };

  recognition.start();

  return () => {
    try {
      recognition.stop();
      recognition.abort();
    } catch {
      // ignore
    }
  };
}

// --- Voice capture (PCM 16k) and TTS helpers with barge-in coordination ---

let currentAzureSynth: any | null = null;
let micActive = false;
const micSubscribers = new Set<(active: boolean) => void>();

// Playback context for server TTS/audio responses
let playbackCtx: AudioContext | null = null;
let playbackTimeOffset = 0;
let playbackGain: GainNode | null = null;

function notifyMic(active: boolean) {
  micSubscribers.forEach(cb => {
    try { cb(active); } catch { /* ignore */ }
  });
}

export function onMicActiveChange(cb: (active: boolean) => void): () => void {
  micSubscribers.add(cb);
  return () => micSubscribers.delete(cb);
}

export function isMicActive(): boolean { return micActive; }

export function cancelSpeech(): void {
  try { if (typeof window !== "undefined") window.speechSynthesis?.cancel(); } catch {}
  try {
    if (currentAzureSynth) {
      const synth = currentAzureSynth;
      currentAzureSynth = null;
      synth.stopSpeakingAsync?.(() => synth.close?.(), () => synth.close?.());
    }
  } catch {}
}

async function waitWhileMicActive(): Promise<void> {
  if (!micActive) return;
  await new Promise<void>((resolve) => {
    const unsub = onMicActiveChange((active) => {
      if (!active) {
        unsub();
        resolve();
      }
    });
  });
}

// Azure Speech Synthesis via token endpoint, defers if mic is active
export async function speakAzureText(
  text: string,
  voiceName = "en-GB-AdaMultilingualNeural"
): Promise<void> {
  if (!text) return;
  try {
    // If user is speaking, defer TTS until they finish
    await waitWhileMicActive();
    const resp = await fetch("/speech/token", { credentials: "include" });
    if (!resp.ok) throw new Error("Failed to fetch speech token");
    const { token, region } = await resp.json();
    // Dynamically import to avoid bundling if unused
    const sdk = await import("microsoft-cognitiveservices-speech-sdk");
    const speechConfig = sdk.SpeechConfig.fromAuthorizationToken(token, region);
    speechConfig.speechSynthesisVoiceName = voiceName;
    const audioConfig = sdk.AudioConfig.fromDefaultSpeakerOutput();
    const synth = new sdk.SpeechSynthesizer(speechConfig, audioConfig);
    currentAzureSynth = synth;
    await new Promise<void>((resolve, reject) => {
      synth.speakTextAsync(
        text,
        () => {
          synth.close();
          if (currentAzureSynth === synth) currentAzureSynth = null;
          resolve();
        },
        (e: any) => {
          synth.close();
          if (currentAzureSynth === synth) currentAzureSynth = null;
          reject(e);
        }
      );
    });
  } catch (e) {
    // Fallback to browser TTS if Azure Speech fails
    await waitWhileMicActive();
    speakText(text);
  }
}

// Start capturing microphone, resample to 16k PCM, return stop() and final base64 PCM
export async function startVoiceCapture(
  onStop: (audioBase64: string, sampleRate: number) => void
): Promise<() => void> {
  if (typeof navigator === "undefined" || !navigator.mediaDevices?.getUserMedia) {
    throw new Error("Microphone not supported");
  }
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  const audioCtx = new (window.AudioContext || (window as any).webkitAudioContext)();
  const source = audioCtx.createMediaStreamSource(stream);

  const processor = audioCtx.createScriptProcessor(4096, 1, 1);
  const chunks: Float32Array[] = [];
  processor.onaudioprocess = (e: AudioProcessingEvent) => {
    const input = e.inputBuffer.getChannelData(0);
    chunks.push(new Float32Array(input));
  };
  source.connect(processor);
  processor.connect(audioCtx.destination);

  micActive = true; notifyMic(true); cancelSpeech();

  const stop = () => {
    try { processor.disconnect(); } catch {}
    try { source.disconnect(); } catch {}
    try { stream.getTracks().forEach(t => t.stop()); } catch {}
    try { audioCtx.close(); } catch {}

    // Flatten float32 PCM
    const inputSr = audioCtx.sampleRate;
    const totalLength = chunks.reduce((n, a) => n + a.length, 0);
    const floatData = new Float32Array(totalLength);
    let offset = 0;
    for (const c of chunks) { floatData.set(c, offset); offset += c.length; }

    // Resample to 16k using simple linear interpolation
    const targetSr = 16000;
    const ratio = inputSr / targetSr;
    const newLength = Math.floor(floatData.length / ratio);
    const resampled = new Float32Array(newLength);
    for (let i = 0; i < newLength; i++) {
      const idx = i * ratio;
      const i0 = Math.floor(idx);
      const i1 = Math.min(i0 + 1, floatData.length - 1);
      const frac = idx - i0;
      resampled[i] = floatData[i0] * (1 - frac) + floatData[i1] * frac;
    }

    // Convert float32 [-1,1] to 16-bit PCM little-endian
    const pcm = new Int16Array(resampled.length);
    for (let i = 0; i < resampled.length; i++) {
      const s = Math.max(-1, Math.min(1, resampled[i]));
      pcm[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
    }
    const bytes = new Uint8Array(pcm.buffer);
    // Base64 encode
    let binary = "";
    for (let i = 0; i < bytes.byteLength; i++) binary += String.fromCharCode(bytes[i]);
    const b64 = btoa(binary);

    micActive = false; notifyMic(false);
    onStop(b64, targetSr);
  };

  return stop;
}


// Live streaming via WebSocket to backend /voice/live
export function startLiveVoice(
  onPartial: (text: string) => void,
  onFinal: (text: string) => void
): Promise<() => void> {
  return new Promise(async (resolve, reject) => {
    try {
      if (typeof navigator === "undefined" || !navigator.mediaDevices?.getUserMedia) {
        throw new Error("Microphone not supported");
      }
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const audioCtx = new (window.AudioContext || (window as any).webkitAudioContext)();
      const source = audioCtx.createMediaStreamSource(stream);
      const processor = audioCtx.createScriptProcessor(4096, 1, 1);

      const wsProto = location.protocol === "https:" ? "wss" : "ws";
      const ws = new WebSocket(`${wsProto}://${location.host}/voice/live`);

      let finalReceived = false;
      ws.onmessage = (ev) => {
        try {
          const msg = JSON.parse(ev.data);
          if (msg?.type === "partial") onPartial(msg.text || "");
          if (msg?.type === "final") {
            finalReceived = true;
            onFinal(msg.text || "");
            // Close socket shortly after final arrives to allow server flush
            try { setTimeout(() => { try { ws.close(); } catch {} }, 150); } catch {}
          }
          if (msg?.type === "audio" && msg?.data) {
            try { enqueuePcm16Playback(msg.data, 24000); } catch {}
          }
          if (msg?.type === "audio_done") {
            try { if (playbackCtx) playbackTimeOffset = Math.max(playbackCtx.currentTime, playbackTimeOffset); } catch {}
          }
        } catch {}
      };
      ws.onerror = () => {
        // Surface error and stop
        stopAll();
        reject(new Error("Live voice websocket error"));
      };

      function floatTo16LEbuffer(f32: Float32Array): ArrayBuffer {
        const buf = new ArrayBuffer(f32.length * 2);
        const view = new DataView(buf);
        for (let i = 0; i < f32.length; i++) {
          const s = Math.max(-1, Math.min(1, f32[i]));
          view.setInt16(i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
        }
        return buf;
      }

      function resampleTo(f32: Float32Array, inputSr: number, targetSr: number): Float32Array {
        if (inputSr === targetSr) return f32;
        const ratio = inputSr / targetSr;
        const newLength = Math.floor(f32.length / ratio);
        const out = new Float32Array(newLength);
        for (let i = 0; i < newLength; i++) {
          const idx = i * ratio;
          const i0 = Math.floor(idx);
          const i1 = Math.min(i0 + 1, f32.length - 1);
          const frac = idx - i0;
          out[i] = f32[i0] * (1 - frac) + f32[i1] * frac;
        }
        return out;
      }

      processor.onaudioprocess = (e: AudioProcessingEvent) => {
        const input = e.inputBuffer.getChannelData(0);
        // Voice Live expects 24kHz PCM16 mono
        const targetSr = 24000;
        const inputSr = audioCtx.sampleRate;
        const resampled = resampleTo(input, inputSr, targetSr);
        if (ws.readyState === WebSocket.OPEN) {
          ws.send(floatTo16LEbuffer(resampled));
        }
      };

      // Wait until websocket opens before sending audio frames
      ws.onopen = () => {
        source.connect(processor);
        processor.connect(audioCtx.destination);
        micActive = true; notifyMic(true);
        // Stop any ongoing TTS and server audio playback immediately
        cancelSpeech();
        stopServerPlayback();
      };

      let stopped = false;
      const stopAll = () => {
        if (stopped) return; stopped = true;
        try { processor.disconnect(); } catch {}
        try { source.disconnect(); } catch {}
        try { stream.getTracks().forEach(t => t.stop()); } catch {}
        try { if (audioCtx && audioCtx.state !== "closed") audioCtx.close(); } catch {}
        // Ask server to commit and start response; keep socket open for transcripts
        try { if (ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: "stop" })); } catch {}
        // Fallback: if no final arrives, close socket after a short grace period
        try {
          setTimeout(() => {
            try { if (!finalReceived && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) ws.close(); } catch {}
          }, 2000);
        } catch {}
        micActive = false; notifyMic(false);
      };

      resolve(stopAll);
    } catch (e) {
      reject(e);
    }
  });
}

function b64ToUint8Array(b64: string): Uint8Array {
  const binary = atob(b64);
  const len = binary.length;
  const bytes = new Uint8Array(len);
  for (let i = 0; i < len; i++) bytes[i] = binary.charCodeAt(i);
  return bytes;
}

function ensurePlaybackCtx(): AudioContext {
  if (!playbackCtx || playbackCtx.state === "closed") {
    playbackCtx = new (window.AudioContext || (window as any).webkitAudioContext)();
    playbackTimeOffset = playbackCtx.currentTime;
  }
  return playbackCtx;
}

function enqueuePcm16Playback(b64Pcm: string, sampleRate: number): void {
  const ctx = ensurePlaybackCtx();
  const bytes = b64ToUint8Array(b64Pcm);
  const samples = new Int16Array(bytes.buffer, bytes.byteOffset, Math.floor(bytes.byteLength / 2));
  const float = new Float32Array(samples.length);
  for (let i = 0; i < samples.length; i++) {
    const s = samples[i];
    float[i] = s < 0 ? s / 0x8000 : s / 0x7fff;
  }
  const buffer = ctx.createBuffer(1, float.length, sampleRate);
  buffer.getChannelData(0).set(float);
  const source = ctx.createBufferSource();
  source.buffer = buffer;
  source.connect(ctx.destination);
  const startAt = Math.max(ctx.currentTime, playbackTimeOffset);
  try { source.start(startAt); } catch { try { source.start(); } catch {} }
  playbackTimeOffset = startAt + buffer.duration;
}

function stopServerPlayback(): void {
  try {
    if (playbackCtx) {
      // Mute and reset offset; previously scheduled buffers (if any) will be inaudible
      if (playbackGain) playbackGain.gain.value = 0;
      playbackTimeOffset = Math.max(playbackCtx.currentTime, playbackTimeOffset);
    }
  } catch {}
  // Keep context so we can resume playback without requiring a new user gesture
}


