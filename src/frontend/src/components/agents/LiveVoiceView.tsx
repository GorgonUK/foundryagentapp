import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { AgentPreviewChatBot } from "./AgentPreviewChatBot";
import { LiveVoiceAvatar } from "./LiveVoiceAvatar";
import { IChatItem } from "./chatbot/types";
import styles from "./LiveVoiceView.module.css";
import { formatAgentName } from "../../utils/formatAgentName";

const SPEECH_SDK_SCRIPT_ID = "azure-speech-sdk";
const SPEECH_SDK_SRC = "https://aka.ms/csspeech/jsbrowserpackageraw";
const LOG_PREFIX = "[LiveVoice]";

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
  audioInputDeviceId?: string;
  avatarCharacter?: string;
  avatarStyle?: string;
  voice?: string;
}

export function LiveVoiceView({
  agentDetails,
  onClose: _onClose,
  onMessageListChange: _onMessageListChange,
  messageList,
  isResponding,
  onSend,
  audioInputDeviceId,
  avatarCharacter = "meg",
  avatarStyle = "business",
  voice = "en-GB-LibbyNeural",
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
  const micStreamRef = useRef<MediaStream | null>(null);
  const isCleaningUpRef = useRef<boolean>(false);
  const isConnectingRef = useRef<boolean>(false);
  const agentDisplayName = useMemo(() => {
    const formatted = formatAgentName(agentDetails?.name);
    return formatted || agentDetails?.name || "";
  }, [agentDetails?.name]);
  
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
      console.log(LOG_PREFIX, "Speech SDK already loaded on window");
      sdkRef.current = win.SpeechSDK;
      setSdkReady(true);
      return;
    }

    let cancelled = false;
    const existingScript = document.getElementById(SPEECH_SDK_SCRIPT_ID) as HTMLScriptElement | null;
    if (existingScript) {
      existingScript.addEventListener("load", () => {
        if (!cancelled && win.SpeechSDK) {
          console.log(LOG_PREFIX, "Speech SDK script load event (existing tag)");
          sdkRef.current = win.SpeechSDK;
          setSdkReady(true);
        }
      });
      existingScript.addEventListener("error", () => {
        if (!cancelled) {
          console.error(LOG_PREFIX, "Speech SDK existing script failed to load");
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
        console.log(LOG_PREFIX, "Speech SDK script loaded successfully");
        sdkRef.current = win.SpeechSDK;
        setSdkReady(true);
      } else {
        console.error(LOG_PREFIX, "Speech SDK script loaded but SpeechSDK not found on window");
        setSdkError("Azure Speech SDK loaded but not available");
      }
    };
    script.onerror = (err) => {
      if (!cancelled) {
        console.error(LOG_PREFIX, "Failed to load Azure Speech SDK", err);
        setSdkError("Failed to load Azure Speech SDK");
      }
    };
    document.head.appendChild(script);

    return () => {
      cancelled = true;
    };
  }, []);

  // Cleanup function to properly close all connections
  const cleanupConnections = useCallback(async (): Promise<void> => {
    if (isCleaningUpRef.current) {
      console.log(LOG_PREFIX, "Cleanup already in progress, waiting...");
      // Wait for existing cleanup to complete
      while (isCleaningUpRef.current) {
        await new Promise((resolve) => setTimeout(resolve, 100));
      }
      return;
    }

    isCleaningUpRef.current = true;
    console.log(LOG_PREFIX, "Starting cleanup of avatar connections...");

    try {
      // Stop speech recognition first
      if (speechRecognizerRef.current) {
        try {
          await new Promise<void>((resolve) => {
            speechRecognizerRef.current.stopContinuousRecognitionAsync(
              () => {
                console.log(LOG_PREFIX, "Speech recognition stopped");
                resolve();
              },
              (err: any) => {
                console.warn(LOG_PREFIX, "Error stopping speech recognition", err);
                resolve(); // Continue cleanup even if stop fails
              }
            );
          });
        } catch (err) {
          console.warn(LOG_PREFIX, "Exception stopping speech recognition", err);
        }
        try {
          speechRecognizerRef.current.close();
        } catch (err) {
          console.warn(LOG_PREFIX, "Exception closing speech recognizer", err);
        }
        speechRecognizerRef.current = null;
      }

      // Close avatar synthesizer
      if (avatarSynthesizerRef.current) {
        try {
          avatarSynthesizerRef.current.close();
          console.log(LOG_PREFIX, "Avatar synthesizer closed");
        } catch (err) {
          console.warn(LOG_PREFIX, "Exception closing avatar synthesizer", err);
        }
        avatarSynthesizerRef.current = null;
      }

      // Close peer connection
      if (peerConnectionRef.current) {
        try {
          peerConnectionRef.current.close();
          console.log(LOG_PREFIX, "Peer connection closed");
        } catch (err) {
          console.warn(LOG_PREFIX, "Exception closing peer connection", err);
        }
        peerConnectionRef.current = null;
      }

      // Stop microphone tracks
      if (micStreamRef.current) {
        micStreamRef.current.getTracks().forEach((track) => {
          track.stop();
          console.log(LOG_PREFIX, "Microphone track stopped:", track.label);
        });
        micStreamRef.current = null;
      }

      // Clear combined stream
      if (combinedStreamRef.current) {
        combinedStreamRef.current.getTracks().forEach((track) => track.stop());
        combinedStreamRef.current = null;
      }

      // Clear queue and reset state
      spokenTextQueueRef.current = [];
      setVideoStream(null);
      setIsConnected(false);

      // Give a small delay to ensure all cleanup is complete
      await new Promise((resolve) => setTimeout(resolve, 500));
      console.log(LOG_PREFIX, "Cleanup completed");
    } finally {
      isCleaningUpRef.current = false;
    }
  }, []);

  // Connect to Azure Avatar when SDK is ready
  useEffect(() => {
    if (!sdkReady || sdkError) {
      return;
    }

    const sdk = sdkRef.current;
    if (!sdk) {
      console.error(LOG_PREFIX, "SDK ready flag set but SpeechSDK ref missing");
      setSdkError("Azure Speech SDK not available");
      return;
    }

    let cancelled = false;

    setSessionError(null);

    const connectAvatar = async () => {
      // Wait for any ongoing cleanup to complete
      if (isCleaningUpRef.current) {
        console.log(LOG_PREFIX, "Waiting for cleanup to complete before connecting...");
        while (isCleaningUpRef.current) {
          await new Promise((resolve) => setTimeout(resolve, 100));
        }
      }

      // Prevent concurrent connection attempts
      if (isConnectingRef.current) {
        console.log(LOG_PREFIX, "Connection already in progress, skipping...");
        return;
      }

      if (cancelled) return;

      isConnectingRef.current = true;

      try {
        console.log(LOG_PREFIX, "Requesting speech token…");
        const tokenResp = await fetch("/speech/token", { credentials: "include" });
        if (!tokenResp.ok) {
          const errorText = await tokenResp.text().catch(() => "");
          console.error(LOG_PREFIX, "Failed to fetch speech token", tokenResp.status, errorText);
          throw new Error(`Failed to fetch speech token (${tokenResp.status})`);
        }
        const { token, region } = await tokenResp.json();
        console.log(LOG_PREFIX, "Obtained speech token for region", region);

        console.log(LOG_PREFIX, "Requesting avatar relay token…");
        const relayTokenResp = await fetch("/speech/avatar/relay/token", { credentials: "include" });
        if (!relayTokenResp.ok) {
          const errorText = await relayTokenResp.text();
          console.error(LOG_PREFIX, "Failed to fetch avatar relay token", relayTokenResp.status, errorText);
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
        console.log(LOG_PREFIX, "Relay token received; top-level keys:", Object.keys(relayTokenData ?? {}));

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
        const avatarConfig = new sdk.AvatarConfig(avatarCharacter, avatarStyle);
        const avatarSynthesizer = new sdk.AvatarSynthesizer(speechConfig, avatarConfig);
        avatarSynthesizerRef.current = avatarSynthesizer;

        avatarSynthesizer.avatarEventReceived = (_s: any, e: any) => {
          const offsetMsg = e.offset === 0 ? "" : `, offset: ${e.offset / 10000}ms`;
          console.log(LOG_PREFIX, "Avatar event received", e?.eventType, offsetMsg);
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
              console.log(LOG_PREFIX, "[WebRTC event]", webRTCEvent);
            } catch {
              console.log(LOG_PREFIX, "[WebRTC event raw]", e.data);
            }
          };
        });
        peerConnection.createDataChannel("eventChannel");

        peerConnection.oniceconnectionstatechange = () => {
          const state = peerConnection.iceConnectionState;
          if (state === "disconnected" || state === "failed" || state === "closed") {
            console.warn(LOG_PREFIX, "WebRTC connection lost:", state);
            setVideoStream(null);
            setIsConnected(false);
          } else if (state === "connected" || state === "completed") {
            console.log(LOG_PREFIX, "WebRTC connection established");
            setIsConnected(true);
          }
        };

        peerConnection.addTransceiver("video", { direction: "sendrecv" });
        peerConnection.addTransceiver("audio", { direction: "sendrecv" });

        // Start avatar and wait for it to be ready
        try {
          const startResult = await avatarSynthesizer.startAvatarAsync(peerConnection);
          if (startResult.reason === sdk.ResultReason.SynthesizingAudioCompleted) {
            console.log(LOG_PREFIX, "Avatar started successfully", startResult.resultId);
            setIsConnected(true);
          } else {
            console.warn(LOG_PREFIX, "Avatar start returned unexpected reason", startResult.reason);
            if (startResult.reason === sdk.ResultReason.Canceled) {
              const cancellationDetails = sdk.CancellationDetails.fromResult(startResult);
              console.error(LOG_PREFIX, "Avatar canceled:", cancellationDetails.errorDetails);
              setSessionError(`Avatar canceled: ${cancellationDetails.errorDetails}`);
            }
          }
        } catch (error: any) {
          console.error(LOG_PREFIX, "Avatar failed to start", error);
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

        const audioConfig = audioInputDeviceId
          ? sdk.AudioConfig.fromMicrophoneInput(audioInputDeviceId)
          : sdk.AudioConfig.fromDefaultMicrophoneInput();
        console.log(LOG_PREFIX, "Creating speech recognizer with audio config", {
          usingSpecificDevice: !!audioInputDeviceId,
        });
        const speechRecognizer = sdk.SpeechRecognizer.FromConfig(
          speechRecognitionConfig,
          autoDetectSourceLanguageConfig,
          audioConfig
        );
        speechRecognizerRef.current = speechRecognizer;

        speechRecognizer.sessionStarted = (_s: any, e: any) => {
          console.log(LOG_PREFIX, "Speech session started", e?.sessionId);
        };
        speechRecognizer.sessionStopped = (_s: any, e: any) => {
          console.log(LOG_PREFIX, "Speech session stopped", e?.sessionId);
        };
        speechRecognizer.speechStartDetected = (_s: any, e: any) => {
          console.log(LOG_PREFIX, "Speech start detected", e?.offset);
        };
        speechRecognizer.speechEndDetected = (_s: any, e: any) => {
          console.log(LOG_PREFIX, "Speech end detected", e?.offset);
        };

        speechRecognizer.recognized = (_s: any, e: any) => {
          if (e.result.reason === sdk.ResultReason.RecognizedSpeech) {
            const userQuery = e.result.text.trim();
            if (userQuery) {
              console.log(LOG_PREFIX, "User speech recognized:", userQuery);
              onSendRef.current(userQuery);
            } else {
              console.log(LOG_PREFIX, "Recognizer returned empty text for recognized speech event");
            }
          } else if (e.result.reason === sdk.ResultReason.NoMatch) {
            console.warn(LOG_PREFIX, "Recognizer received speech but could not match to text");
          }
        };

        speechRecognizer.canceled = (_s: any, e: any) => {
          console.warn(LOG_PREFIX, "Speech recognizer canceled", e);
          if (e?.errorDetails) {
            setSessionError(`Recognizer canceled: ${e.errorDetails}`);
          }
        };

        speechRecognizer.recognizing = (_s: any, e: any) => {
          if (e.result?.text) {
            console.debug(LOG_PREFIX, "Speech interim result:", e.result.text);
          }
        };

        if (cancelled) return;

        console.log(LOG_PREFIX, "Requesting microphone stream…", {
          deviceId: audioInputDeviceId ?? "default",
        });
        try {
          const constraints: MediaStreamConstraints =
            audioInputDeviceId
              ? { audio: { deviceId: { exact: audioInputDeviceId } } }
              : { audio: true };
          const micStream = await navigator.mediaDevices.getUserMedia(constraints);
          micStreamRef.current = micStream;
          console.log(LOG_PREFIX, "Microphone stream granted", micStream.getAudioTracks().map((t) => t.label));
        } catch (micError) {
          console.error(LOG_PREFIX, "User media (microphone) request failed", micError);
          setSessionError("Microphone access denied or unavailable");
          throw micError;
        }
        speechRecognizer.startContinuousRecognitionAsync(
          () => {
            console.log(LOG_PREFIX, "Speech recognizer started");
          },
          (err: any) => {
            console.error(LOG_PREFIX, "Failed to start speech recognition", err);
            setSessionError("Failed to start speech recognition");
          }
        );
      } catch (error) {
        if (!cancelled) {
          console.error(LOG_PREFIX, "Error connecting avatar pipeline", error);
          setSessionError(error instanceof Error ? error.message : "Failed to start avatar session");
        }
      } finally {
        isConnectingRef.current = false;
      }
    };

    // Cleanup first, then connect
    void (async () => {
      await cleanupConnections();
      if (!cancelled) {
        await connectAvatar();
      }
    })();

    return () => {
      cancelled = true;
      isConnectingRef.current = false;
      // Trigger cleanup when dependencies change
      void cleanupConnections();
    };
  }, [sdkReady, sdkError, audioInputDeviceId, avatarCharacter, avatarStyle, voice, cleanupConnections]);

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
                  <voice name='${voice}'>
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
  }, [isConnected, voice]);

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
      console.log(LOG_PREFIX, "[Lip Sync] No text extracted from message content; not speaking");
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
          agentName={agentDisplayName}
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
