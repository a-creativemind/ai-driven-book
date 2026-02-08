/**
 * ChatbotWidget - Main component for the RAG chatbot interface
 */

import React, { useState, useRef, useEffect } from 'react';
import { ChatMessage } from './ChatMessage';
import { ChatInput } from './ChatInput';
import { useChat } from './hooks/useChat';
import { useTextSelection } from './hooks/useTextSelection';
import type { ChatbotWidgetProps } from './types';
import styles from './ChatbotWidget.module.css';

const DEFAULT_API_URL = 'http://localhost:8000';

export function ChatbotWidget({
  apiUrl = DEFAULT_API_URL,
  position = 'bottom-right',
  defaultOpen = false,
}: ChatbotWidgetProps): JSX.Element {
  const [isOpen, setIsOpen] = useState(defaultOpen);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const { messages, isLoading, sendMessage, clearMessages } = useChat({
    apiUrl,
  });

  const { selection, clearSelection } = useTextSelection();

  // Scroll to bottom when new messages arrive
  useEffect(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages]);

  const handleSend = async (message: string) => {
    await sendMessage(message, selection);
    // Clear selection after sending
    if (selection) {
      clearSelection();
    }
  };

  const positionClass =
    position === 'bottom-right' ? styles.bottomRight : styles.bottomLeft;

  return (
    <div className={`${styles.widgetContainer} ${positionClass}`}>
      {isOpen && (
        <div className={styles.chatWindow}>
          <div className={styles.chatHeader}>
            <h3 className={styles.headerTitle}>Textbook Assistant</h3>
            <button
              className={styles.closeButton}
              onClick={() => setIsOpen(false)}
              title="Close chat"
            >
              <CloseIcon />
            </button>
          </div>

          <div className={styles.messagesContainer}>
            {messages.length === 0 ? (
              <WelcomeMessage />
            ) : (
              <>
                {messages.map((message) => (
                  <ChatMessage key={message.id} message={message} />
                ))}
                {isLoading && <TypingIndicator />}
              </>
            )}
            <div ref={messagesEndRef} />
          </div>

          <ChatInput
            onSend={handleSend}
            isLoading={isLoading}
            selection={selection}
            onClearSelection={clearSelection}
          />
        </div>
      )}

      <button
        className={styles.toggleButton}
        onClick={() => setIsOpen(!isOpen)}
        title={isOpen ? 'Close chat' : 'Open chat'}
      >
        {isOpen ? <CloseIcon /> : <ChatIcon />}
      </button>
    </div>
  );
}

function WelcomeMessage(): JSX.Element {
  return (
    <div className={styles.welcomeMessage}>
      <h3>Welcome!</h3>
      <p>
        Ask me anything about Physical AI and Humanoid Robotics.
        You can also select text on the page and ask questions about it.
      </p>
    </div>
  );
}

function TypingIndicator(): JSX.Element {
  return (
    <div className={styles.typingIndicator}>
      <span className={styles.typingDot} />
      <span className={styles.typingDot} />
      <span className={styles.typingDot} />
    </div>
  );
}

function ChatIcon(): JSX.Element {
  return (
    <svg
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
    </svg>
  );
}

function CloseIcon(): JSX.Element {
  return (
    <svg
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <line x1="18" y1="6" x2="6" y2="18" />
      <line x1="6" y1="6" x2="18" y2="18" />
    </svg>
  );
}

export default ChatbotWidget;
