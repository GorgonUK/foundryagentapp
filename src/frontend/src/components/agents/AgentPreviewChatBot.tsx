import React, { useState, useMemo, useEffect, useRef } from "react";
import { AssistantMessage } from "./AssistantMessage";
import { UserMessage } from "./UserMessage";
import { ChatInput } from "./chatbot/ChatInput";
import { AgentPreviewChatBotProps } from "./chatbot/types";

import styles from "./AgentPreviewChatBot.module.css";
import clsx from "clsx";

export function AgentPreviewChatBot({
  agentName,
  agentLogo,
  chatContext,
}: AgentPreviewChatBotProps): React.JSX.Element {
  const [currentUserMessage, setCurrentUserMessage] = useState<
    string | undefined
  >();

  const messageListFromChatContext = useMemo(
    () => chatContext.messageList ?? [],
    [chatContext.messageList]
  );

  // Auto-scroll to bottom on new messages or state changes
  const containerRef = useRef<HTMLDivElement | null>(null);
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    el.scrollTop = el.scrollHeight;
  }, [messageListFromChatContext.length, chatContext.isResponding]);

  const onEditMessage = (messageId: string) => {
    const selectedMessage = messageListFromChatContext.find(
      (message) => !message.isAnswer && message.id === messageId
    )?.content;
    setCurrentUserMessage(selectedMessage);
  };

  const isEmpty = messageListFromChatContext.length === 0;

  const lastMsg = messageListFromChatContext[messageListFromChatContext.length - 1];
  const lastIsAssistant = !!lastMsg?.isAnswer;
  const showTypingIndicator = chatContext.isResponding && !lastIsAssistant;
  const typingPlaceholder = useMemo(
    () => ({
      id: "typing-indicator",
      content: "",
      isAnswer: true,
      more: { time: new Date().toISOString() },
    }),
    []
  );

  return (
    <div
      className={clsx(
        styles.chatContainer,
        isEmpty ? styles.emptyChatContainer : undefined
      )}
    >
      {!isEmpty ? (
        <div className={styles.copilotChatContainer} ref={containerRef}>
          {messageListFromChatContext.map((message, index, messageList) =>
            message.isAnswer ? (
              <AssistantMessage
                key={message.id}
                agentLogo={agentLogo}
                agentName={agentName}
                loadingState={
                  index === messageList.length - 1 && chatContext.isResponding
                    ? "loading"
                    : "none"
                }
                message={message}
              />
            ) : (
              <UserMessage
                key={message.id}
                message={message}
                onEditMessage={onEditMessage}
              />
            )
          )}
          {showTypingIndicator && (
            <div className={styles.typingRow}>
              <span className={styles.typingDots} aria-live="polite" aria-label="Assistant is thinking"></span>
            </div>
          )}
        </div>
      ) : (
        // Empty div needed for proper animation when transitioning to non-empty state
        <div />
      )}
      <div className={styles.inputContainer}>
        <ChatInput
          currentUserMessage={currentUserMessage}
          isGenerating={chatContext.isResponding}
          onSubmit={chatContext.onSend}
        />
      </div>
    </div>
  );
}
