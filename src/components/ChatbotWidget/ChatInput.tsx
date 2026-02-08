/**
 * ChatInput component for message input
 */

import React, { useState, useRef, useEffect } from 'react';
import type { TextSelection } from './types';
import styles from './ChatbotWidget.module.css';

interface ChatInputProps {
  onSend: (message: string) => void;
  isLoading: boolean;
  selection?: TextSelection | null;
  onClearSelection: () => void;
}

export function ChatInput({
  onSend,
  isLoading,
  selection,
  onClearSelection,
}: ChatInputProps): JSX.Element {
  const [input, setInput] = useState('');
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Auto-resize textarea
  useEffect(() => {
    const textarea = textareaRef.current;
    if (textarea) {
      textarea.style.height = 'auto';
      textarea.style.height = `${Math.min(textarea.scrollHeight, 120)}px`;
    }
  }, [input]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (input.trim() && !isLoading) {
      onSend(input.trim());
      setInput('');
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  return (
    <form className={styles.inputContainer} onSubmit={handleSubmit}>
      {selection && (
        <div className={styles.selectionBanner}>
          <div className={styles.selectionPreview}>
            <span className={styles.selectionLabel}>Selected text:</span>
            <span className={styles.selectionText}>
              {truncateText(selection.text, 100)}
            </span>
          </div>
          <button
            type="button"
            className={styles.clearSelection}
            onClick={onClearSelection}
            title="Clear selection"
          >
            ×
          </button>
        </div>
      )}

      <div className={styles.inputRow}>
        <textarea
          ref={textareaRef}
          className={styles.textInput}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={
            selection
              ? 'Ask about the selected text...'
              : 'Ask about the textbook...'
          }
          disabled={isLoading}
          rows={1}
        />
        <button
          type="submit"
          className={styles.sendButton}
          disabled={!input.trim() || isLoading}
          title="Send message"
        >
          {isLoading ? (
            <span className={styles.loadingSpinner} />
          ) : (
            <SendIcon />
          )}
        </button>
      </div>
    </form>
  );
}

function truncateText(text: string, maxLength: number): string {
  if (text.length <= maxLength) return text;
  return text.substring(0, maxLength) + '...';
}

function SendIcon(): JSX.Element {
  return (
    <svg
      width="20"
      height="20"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <line x1="22" y1="2" x2="11" y2="13" />
      <polygon points="22 2 15 22 11 13 2 9 22 2" />
    </svg>
  );
}
