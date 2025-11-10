import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { AgentPreviewChatBot } from "./AgentPreviewChatBot";
import { LiveVoiceAvatar } from "./LiveVoiceAvatar";
import { IChatItem } from "./chatbot/types";
import styles from "./LiveVoiceView.module.css";

const SPEECH_SDK_SCRIPT_ID = "azure-speech-sdk";
const SPEECH_SDK_SRC = "https://aka.ms/csspeech/jsbrowserpackageraw";
const DEFAULT_AVATAR_CHARACTER = "meg";
const DEFAULT_AVATAR_STYLE = "formal";
const DEFAULT_VOICE = "en-US-AvaMultilingualNeural";

function extractSpeechText(content: string): string {
  if (!content) {
    return "";
  }
  // Only extract voice_summary, never use text_detail
  const summaryMatch = content.match(/voice_summary:\s*([\s\S]*?)(?:\n\n|text_detail:|$)/i);
  if (summaryMatch && summaryMatch[1]) {
    const extracted = summaryMatch[1].trim();
    return extracted;
  }
  return "";
}

interface LiveVoiceViewProps {
  agentDetails: any;
  onClose: () => void;
  onMessageListChange: (messages: IChatItem[]) => void;
  messageList: IChatItem[];
  isResponding: boolean;
  onSend: (message: string) => void;
}

export function LiveVoiceView({
  agentDetails,
  onClose,
  onMessageListChange,
  messageList,
  isResponding,
  onSend,
}: LiveVoiceViewProps): JSX.Element {
  const [isConnected, setIsConnected] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [videoStream, setVideoStream] = useState<MediaStream | null>(null);
  const [sdkReady, setSdkReady] = useState(false);
  const [sdkError, setSdkError] = useState<string | null>(null);
  const [sessionError, setSessionError] = useState<string | null>(null);

  const sdkRef = useRef<any>(null);
  const avatarSynthesizerRef = useRef<any>(null);
  const speechRecognizerRef = useRef<any>(null);
  const peerConnectionRef = useRef<RTCPeerConnection | null>(null);
  const spokenTextQueueRef = useRef<string[]>([]);
  const onSendRef = useRef(onSend);
  const combinedStreamRef = useRef<MediaStream | null>(null);
  
  // Keep onSendRef up to date
  useEffect(() => {
    onSendRef.current = onSend;
  }, [onSend]);

  // Load Azure Speech SDK
  useEffect(() => {
    if (typeof window === "undefined") {
      setSdkError("Speech SDK unavailable in this environment");
      return;
    }

    const win = window as any;
    if (win.SpeechSDK) {
      sdkRef.current = win.SpeechSDK;
      setSdkReady(true);
      return;
    }

    let cancelled = false;
    const existingScript = document.getElementById(SPEECH_SDK_SCRIPT_ID) as HTMLScriptElement | null;
    if (existingScript) {
      existingScript.addEventListener("load", () => {
        if (!cancelled && win.SpeechSDK) {
          sdkRef.current = win.SpeechSDK;
          setSdkReady(true);
        }
      });
      existingScript.addEventListener("error", () => {
        if (!cancelled) {
          setSdkError("Failed to load Azure Speech SDK");
        }
      });
      return;
    }

    const script = document.createElement("script");
    script.id = SPEECH_SDK_SCRIPT_ID;
    script.src = SPEECH_SDK_SRC;
    script.async = true;
    script.onload = () => {
      if (cancelled) return;
      if (win.SpeechSDK) {
        sdkRef.current = win.SpeechSDK;
        setSdkReady(true);
      } else {
        setSdkError("Azure Speech SDK loaded but not available");
      }
    };
    script.onerror = (err) => {
      if (!cancelled) {
        console.error("Failed to load Azure Speech SDK", err);
        setSdkError("Failed to load Azure Speech SDK");
      }
    };
    document.head.appendChild(script);

    return () => {
      cancelled = true;
    };
  }, []);

  // Connect to Azure Avatar when SDK is ready
  useEffect(() => {
    if (!sdkReady || sdkError) {
      return;
    }

    const sdk = sdkRef.current;
    if (!sdk) {
      setSdkError("Azure Speech SDK not available");
      return;
    }

    let cancelled = false;
    let micStream: MediaStream | null = null;

    setSessionError(null);

    const connectAvatar = async () => {
      try {
        const tokenResp = await fetch("/speech/token", { credentials: "include" });
        if (!tokenResp.ok) {
          throw new Error("Failed to fetch speech token");
        }
        const { token, region } = await tokenResp.json();

        const relayTokenResp = await fetch("/speech/avatar/relay/token", { credentials: "include" });
        if (!relayTokenResp.ok) {
          const errorText = await relayTokenResp.text();
          let errorDetail = errorText;
          try {
            const errorJson = JSON.parse(errorText);
            errorDetail = errorJson.detail || errorText;
          } catch {
            // Not JSON, use as-is
          }
          throw new Error(`Failed to fetch avatar relay token: ${relayTokenResp.status} ${errorDetail}`);
        }
        const relayTokenData = await relayTokenResp.json();

        if (cancelled) return;

        // Check if the response has null values - this indicates the feature isn't available
        const iceServerUrl = relayTokenData?.Urls?.[0] || relayTokenData?.urls?.[0] || relayTokenData?.Url || relayTokenData?.url;
        const iceServerUsername = relayTokenData?.Username || relayTokenData?.username;
        const iceServerCredential = relayTokenData?.Password || relayTokenData?.password;
        
        if (!iceServerUrl || !iceServerUsername || !iceServerCredential) {
          console.error("Relay token data structure:", relayTokenData);
          throw new Error(
            `Avatar feature not available. ` +
            `Region '${region}' may not support Avatar, or the API key may not have Avatar permissions. ` +
            `Please check Azure Speech Service documentation for supported regions and ensure Avatar feature is enabled on your subscription.`
          );
        }

        const speechConfig = sdk.SpeechConfig.fromAuthorizationToken(token, region);
        const avatarConfig = new sdk.AvatarConfig(DEFAULT_AVATAR_CHARACTER, DEFAULT_AVATAR_STYLE);
        const avatarSynthesizer = new sdk.AvatarSynthesizer(speechConfig, avatarConfig);
        avatarSynthesizerRef.current = avatarSynthesizer;

        avatarSynthesizer.avatarEventReceived = (_s: any, e: any) => {
          const offsetMsg = e.offset === 0 ? "" : `, offset: ${e.offset / 10000}ms`;
        };

        const peerConnection = new RTCPeerConnection({
          iceServers: [
            {
              urls: [iceServerUrl],
              username: iceServerUsername,
              credential: iceServerCredential,
            },
          ],
        });
        peerConnectionRef.current = peerConnection;

        combinedStreamRef.current = new MediaStream();
        
        peerConnection.ontrack = (event) => {
          if (!combinedStreamRef.current) {
            combinedStreamRef.current = new MediaStream();
          }
          
          if (event.track.kind === "video") {
            combinedStreamRef.current.addTrack(event.track);
            setVideoStream(new MediaStream(combinedStreamRef.current.getTracks()));
          } else if (event.track.kind === "audio") {
            combinedStreamRef.current.addTrack(event.track);
            // Update stream with both video and audio
            setVideoStream(new MediaStream(combinedStreamRef.current.getTracks()));
          }
        };

        peerConnection.addEventListener("datachannel", (event) => {
          const dataChannel = event.channel;
          dataChannel.onmessage = (e) => {
            try {
              const webRTCEvent = JSON.parse(e.data);
              console.log("[WebRTC event]", webRTCEvent);
            } catch {
              console.log("[WebRTC event raw]", e.data);
            }
          };
        });
        peerConnection.createDataChannel("eventChannel");

        peerConnection.oniceconnectionstatechange = () => {
          const state = peerConnection.iceConnectionState;
          if (state === "disconnected" || state === "failed" || state === "closed") {
            console.warn("WebRTC connection lost:", state);
            setVideoStream(null);
            setIsConnected(false);
          } else if (state === "connected" || state === "completed") {
            console.log("WebRTC connection established");
            setIsConnected(true);
          }
        };

        peerConnection.addTransceiver("video", { direction: "sendrecv" });
        peerConnection.addTransceiver("audio", { direction: "sendrecv" });

        // Start avatar and wait for it to be ready
        try {
          const startResult = await avatarSynthesizer.startAvatarAsync(peerConnection);
          if (startResult.reason === sdk.ResultReason.SynthesizingAudioCompleted) {
            console.log("Avatar started successfully", startResult.resultId);
            setIsConnected(true);
          } else {
            console.warn("Avatar started with unexpected reason", startResult.reason);
            if (startResult.reason === sdk.ResultReason.Canceled) {
              const cancellationDetails = sdk.CancellationDetails.fromResult(startResult);
              console.error("Avatar canceled:", cancellationDetails.errorDetails);
              setSessionError(`Avatar canceled: ${cancellationDetails.errorDetails}`);
            }
          }
        } catch (error: any) {
          console.error("Avatar failed to start", error);
          setSessionError("Avatar failed to start");
          return;
        }

        if (cancelled) return;

        const speechRecognitionConfig = sdk.SpeechConfig.fromAuthorizationToken(token, region);
        speechRecognitionConfig.setProperty(
          sdk.PropertyId.SpeechServiceConnection_LanguageIdMode,
          "Continuous"
        );
        const autoDetectSourceLanguageConfig = sdk.AutoDetectSourceLanguageConfig.fromLanguages(["en-US"]);

        const speechRecognizer = sdk.SpeechRecognizer.FromConfig(
          speechRecognitionConfig,
          autoDetectSourceLanguageConfig,
          sdk.AudioConfig.fromDefaultMicrophoneInput()
        );
        speechRecognizerRef.current = speechRecognizer;

        speechRecognizer.recognized = (_s: any, e: any) => {
          if (e.result.reason === sdk.ResultReason.RecognizedSpeech) {
            const userQuery = e.result.text.trim();
            if (userQuery) {
              console.log("User speech recognized:", userQuery);
              onSendRef.current(userQuery);
            }
          }
        };

        speechRecognizer.canceled = (_s: any, e: any) => {
          console.warn("Speech recognizer canceled", e);
        };

        if (cancelled) return;

        micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
        speechRecognizer.startContinuousRecognitionAsync(
          () => {
            console.log("Speech recognizer started");
          },
          (err: any) => {
            console.error("Failed to start speech recognition", err);
            setSessionError("Failed to start speech recognition");
          }
        );
      } catch (error) {
        if (!cancelled) {
          console.error("Error connecting avatar:", error);
          setSessionError(error instanceof Error ? error.message : "Failed to start avatar session");
        }
      }
    };

    connectAvatar();

    return () => {
      cancelled = true;
      if (avatarSynthesizerRef.current) {
        avatarSynthesizerRef.current.close();
        avatarSynthesizerRef.current = null;
      }
      if (speechRecognizerRef.current) {
        try {
          speechRecognizerRef.current.stopContinuousRecognitionAsync();
        } catch {}
        speechRecognizerRef.current.close();
        speechRecognizerRef.current = null;
      }
      if (peerConnectionRef.current) {
        peerConnectionRef.current.close();
        peerConnectionRef.current = null;
      }
      if (micStream) {
        micStream.getTracks().forEach((track) => track.stop());
      }
      spokenTextQueueRef.current = [];
      setVideoStream(null);
      setIsConnected(false);
    };
  }, [sdkReady, sdkError]);

  const speakNext = useCallback((text: string) => {
    const avatarSynthesizer = avatarSynthesizerRef.current;
    const sdk = sdkRef.current;
    if (!avatarSynthesizer || !sdk) {
      console.warn("[Lip Sync Debug] speakNext: Avatar synthesizer or SDK not initialized", {
        hasSynthesizer: !!avatarSynthesizer,
        hasSdk: !!sdk
      });
      setIsSpeaking(false);
      return;
    }
    
    // Don't try to speak if not connected
    if (!isConnected) {
      console.warn("[Lip Sync Debug] speakNext: Avatar not connected, cannot speak", {
        isConnected
      });
      setIsSpeaking(false);
      return;
    }

    const ssml = `<speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis'
                     xmlns:mstts='http://www.w3.org/2001/mstts'
                     xml:lang='en-US'>
                  <voice name='${DEFAULT_VOICE}'>
                    <mstts:ttsembedding>
                      <mstts:leadingsilence-exact value='0'/>
                      ${text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")}
                    </mstts:ttsembedding>
                  </voice>
                </speak>`;
    setIsSpeaking(true);

    avatarSynthesizer.speakSsmlAsync(ssml)
      .then((result: any) => {
        if (result.reason === sdk.ResultReason.SynthesizingAudioCompleted) {
          console.log("[Lip Sync Debug] speakNext: Speech synthesized successfully", result.resultId);
        } else {
          console.warn("[Lip Sync Debug] speakNext: Unexpected result reason", {
            reason: result.reason,
            resultId: result.resultId
          });
        }
        if (spokenTextQueueRef.current.length > 0) {
          const next = spokenTextQueueRef.current.shift();
          if (next) speakNext(next);
        } else {
          setIsSpeaking(false);
        }
      })
      .catch((error: any) => {
        console.error("[Lip Sync Debug] speakNext: speakSsmlAsync error", error);
        if (spokenTextQueueRef.current.length > 0) {
          const next = spokenTextQueueRef.current.shift();
          if (next) speakNext(next);
        } else {
          setIsSpeaking(false);
        }
      });
  }, [isConnected]);

  const speak = useCallback((text: string) => {
    if (!text) {
      return;
    }
    if (isSpeaking) {
      spokenTextQueueRef.current.push(text);
      return;
    }
    speakNext(text);
  }, [isSpeaking, speakNext]);

  const lastProcessedMessageIdRef = useRef<string | null>(null);
  
  useEffect(() => {
    // Only process when response is complete (not streaming)
    if (isResponding) {
      return;
    }
    
    if (messageList.length === 0) {
      return;
    }
    
    const lastMessage = messageList[messageList.length - 1];
    
    // Prevent processing the same message multiple times
    if (lastProcessedMessageIdRef.current === lastMessage.id) {
      return;
    }
    
    if (!lastMessage.isAnswer || !lastMessage.content) {
      return;
    }

    const metadata = (lastMessage as any).more as Record<string, unknown> | undefined;
    const lipSyncContent = typeof metadata?.fullContentForLipSync === "string"
      ? (metadata.fullContentForLipSync as string)
      : (typeof lastMessage.content === "string" ? lastMessage.content : "");

    if (!lipSyncContent.includes("voice_summary:")) {
      return;
    }

    const text = extractSpeechText(lipSyncContent);
    if (text) {
      lastProcessedMessageIdRef.current = lastMessage.id; // Mark as processed
      speak(text);
    } else {
      console.log("[Lip Sync Debug] useEffect: No text extracted, not speaking");
    }
  }, [messageList, isResponding, speak]);

  return (
    <div className={styles.liveVoiceContainer}>
      <div className={styles.avatarSection}>
        <LiveVoiceAvatar videoStream={videoStream} isConnected={isConnected} />
        <div className={styles.statusIndicator}>
          {sdkError ? (
            <span className={styles.disconnected}>● {sdkError}</span>
          ) : sessionError ? (
            <span className={styles.disconnected}>● {sessionError}</span>
          ) : isConnected ? (
            <span className={styles.connected}>● Connected</span>
          ) : sdkReady ? (
            <span className={styles.disconnected}>● Connecting…</span>
          ) : (
            <span className={styles.disconnected}>● Loading SDK…</span>
          )}
        </div>
      </div>
      <div className={styles.chatSection}>
        <AgentPreviewChatBot
          agentName={agentDetails.name}
          agentLogo={agentDetails.metadata?.logo}
          chatContext={useMemo(() => ({
            messageList,
            isResponding,
            onSend,
          }), [messageList, isResponding, onSend])}
        />
      </div>
    </div>
  );
}
