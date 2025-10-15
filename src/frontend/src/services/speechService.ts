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

// Azure Speech Synthesis via token endpoint
export async function speakAzureText(
  text: string,
  voiceName = "en-GB-AdaMultilingualNeural"
): Promise<void> {
  if (!text) return;
  try {
    const resp = await fetch("/speech/token", { credentials: "include" });
    if (!resp.ok) throw new Error("Failed to fetch speech token");
    const { token, region } = await resp.json();
    // Dynamically import to avoid bundling if unused
    const sdk = await import("microsoft-cognitiveservices-speech-sdk");
    const speechConfig = sdk.SpeechConfig.fromAuthorizationToken(token, region);
    speechConfig.speechSynthesisVoiceName = voiceName;
    const audioConfig = sdk.AudioConfig.fromDefaultSpeakerOutput();
    const synth = new sdk.SpeechSynthesizer(speechConfig, audioConfig);
    await new Promise<void>((resolve, reject) => {
      synth.speakTextAsync(
        text,
        () => {
          synth.close();
          resolve();
        },
        (e: any) => {
          synth.close();
          reject(e);
        }
      );
    });
  } catch (e) {
    // Fallback to browser TTS if Azure Speech fails
    speakText(text);
  }
}


