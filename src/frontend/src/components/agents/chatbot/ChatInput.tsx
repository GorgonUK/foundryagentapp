import React, { useState, useEffect, useRef } from "react";
import {
  ChatInput as ChatInputFluent,
  ImperativeControlPlugin,
  ImperativeControlPluginRef,
} from "@fluentui-copilot/react-copilot";
import { ChatInputProps } from "./types";
import { MicFilled, MicOffFilled } from "@fluentui/react-icons";
import clsx from "clsx";
import { startAzureSTT } from "../../../services/speechService";
import styles from "./ChatInput.module.css";

export const ChatInput: React.FC<ChatInputProps> = ({
  onSubmit,
  isGenerating,
  currentUserMessage,
}) => {
  const [inputText, setInputText] = useState<string>("");
  const controlRef = useRef<ImperativeControlPluginRef>(null);
  const [listening, setListening] = useState<boolean>(false);
  const voiceStopRef = useRef<() => void>();

  useEffect(() => {
    if (currentUserMessage !== undefined) {
      controlRef.current?.setInputText(currentUserMessage ?? "");
    }
  }, [currentUserMessage]);
  const onMessageSend = (text: string): void => {
    if (text && text.trim() !== "") {
      onSubmit(text.trim());
      setInputText("");
      controlRef.current?.setInputText("");
    }
  };

  const toggleMic = () => {
    if (listening) {
      // Stop recording - this will trigger the final callback
      try { 
        voiceStopRef.current?.(); 
      } catch (e) {
        console.error("Error stopping STT", e);
        setListening(false);
      }
      return;
    }
    setListening(true);
    // Azure Speech-to-Text: use STT with fast transcription
    startAzureSTT(
      (partialText) => {
        // Update textbox in real-time with partial results (fast transcription)
        if (partialText) {
          setInputText(partialText);
          controlRef.current?.setInputText(partialText);
        }
      },
      (finalText) => {
        // When recording stops, submit the final text
        setListening(false);
        if (finalText && finalText.trim()) {
          setInputText(finalText);
          controlRef.current?.setInputText(finalText);
          onMessageSend(finalText.trim());
        }
      }
    ).then((stop) => {
      voiceStopRef.current = stop;
    }).catch((e) => {
      console.error("Azure STT failed", e);
      setListening(false);
    });
  };

  return (
    <ChatInputFluent
      aria-label="Chat Input"
      charactersRemainingMessage={(_value: number) => ``} // needed per fluentui-copilot API
      data-testid="chat-input"
      disableSend={isGenerating}
      history={true}
      isSending={isGenerating}
      actions={
        <span>
          <div className={styles.micControl}>
            <div
              role="button"
              aria-pressed={listening}
              aria-label={listening ? "Stop recording" : "Start recording"}
              title={listening ? "Stop recording" : "Start recording"}
              className={clsx(styles.micButton, listening && styles.micListening)}
              tabIndex={0}
              onClick={toggleMic}
              onKeyDown={(e) => {
                if (e.key === "Enter" || e.key === " ") {
                  e.preventDefault();
                  toggleMic();
                }
              }}
            >
              {listening 
  ? <MicOffFilled width="20" height="20" /> 
  : <MicFilled width="20" height="20" />
}
            </div>
          </div>
        </span>
      }
      onChange={(
        _: React.ChangeEvent<HTMLInputElement>,
        d: { value: string }
      ) => {
        setInputText(d.value);
      }}
      onSubmit={() => {
        onMessageSend(inputText ?? "");
      }}
      placeholderValue="Type your message here..."
    >
      <ImperativeControlPlugin ref={controlRef} />
    </ChatInputFluent>
  );
};

export default ChatInput;
