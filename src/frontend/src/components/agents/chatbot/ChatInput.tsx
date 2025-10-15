import React, { useState, useEffect, useRef } from "react";
import {
  ChatInput as ChatInputFluent,
  ImperativeControlPlugin,
  ImperativeControlPluginRef,
} from "@fluentui-copilot/react-copilot";
import { ChatInputProps } from "./types";
import { Button } from "@fluentui/react-components";
import { MicFilled, MicOffFilled } from "@fluentui/react-icons";
import { getSpeechSupport, startTranscription } from "../../../services/speechService";

export const ChatInput: React.FC<ChatInputProps> = ({
  onSubmit,
  isGenerating,
  currentUserMessage,
}) => {
  const [inputText, setInputText] = useState<string>("");
  const controlRef = useRef<ImperativeControlPluginRef>(null);
  const [listening, setListening] = useState<boolean>(false);
  const stopRef = useRef<() => void>();

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
      stopRef.current?.();
      setListening(false);
      return;
    }
    const support = getSpeechSupport();
    if (!support.hasSTT) {
      alert("Speech recognition is not supported in this browser.");
      return;
    }
    setListening(true);
    stopRef.current = startTranscription(
      (text, isFinal) => {
        controlRef.current?.setInputText(text);
        setInputText(text);
        if (isFinal) {
          stopRef.current?.();
          setListening(false);
          onMessageSend(text);
        }
      },
      (err) => {
        console.error("STT error", err);
        setListening(false);
      },
      { interimResults: true }
    );
  };

  return (
    <ChatInputFluent
      aria-label="Chat Input"
      charactersRemainingMessage={(_value: number) => ``} // needed per fluentui-copilot API
      data-testid="chat-input"
      disableSend={isGenerating}
      history={true}
      isSending={isGenerating}
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
      <Button
        aria-label={listening ? "Stop recording" : "Start recording"}
        appearance="subtle"
        onClick={toggleMic}
      >
        {listening ? <MicOffFilled /> : <MicFilled />}
      </Button>
    </ChatInputFluent>
  );
};

export default ChatInput;
