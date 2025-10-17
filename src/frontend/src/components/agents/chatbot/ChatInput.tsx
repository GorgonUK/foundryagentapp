import React, { useState, useEffect, useRef } from "react";
import {
  ChatInput as ChatInputFluent,
  ImperativeControlPlugin,
  ImperativeControlPluginRef,
} from "@fluentui-copilot/react-copilot";
import { ChatInputProps } from "./types";
import { MicFilled, MicOffFilled } from "@fluentui/react-icons";
import clsx from "clsx";
import { startLiveVoice } from "../../../services/speechService";
import styles from "./ChatInput.module.css";

export const ChatInput: React.FC<ChatInputProps> = ({
  onSubmit,
  isGenerating,
  currentUserMessage,
}) => {
  const [inputText, setInputText] = useState<string>("");
  const controlRef = useRef<ImperativeControlPluginRef>(null);
  const [listening, setListening] = useState<boolean>(false);
  const stopRef = useRef<() => void>();
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
      try { stopRef.current?.(); } catch {}
      try { voiceStopRef.current?.(); } catch {}
      setListening(false);
      return;
    }
    setListening(true);
    // Voice live API: stream mic to backend and handle partial/final results
    startLiveVoice(
      () => {
        // Optional: display partials
      },
      (finalText) => {
        if (!finalText) return;
        onMessageSend(finalText.trim());
      }
    ).then((stop) => {
      voiceStopRef.current = stop;
    }).catch((e) => {
      console.error("live voice failed", e);
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
