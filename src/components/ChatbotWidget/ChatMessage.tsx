/**
 * ChatMessage component for displaying individual messages
 */

import React from 'react';
import type { ChatMessage as ChatMessageType } from './types';
import { SourceCard } from './SourceCard';
import styles from './ChatbotWidget.module.css';

interface ChatMessageProps {
  message: ChatMessageType;
}

export function ChatMessage({ message }: ChatMessageProps): JSX.Element {
  const isUser = message.role === 'user';

  return (
    <div
      className={`${styles.message} ${
        isUser ? styles.userMessage : styles.assistantMessage
      }`}
    >
      <div className={styles.messageContent}>
        {isUser ? (
          <p>{message.content}</p>
        ) : (
          <div
            className={styles.markdownContent}
            dangerouslySetInnerHTML={{ __html: formatMarkdown(message.content) }}
          />
        )}
      </div>

      {!isUser && message.sources && message.sources.length > 0 && (
        <div className={styles.sourcesContainer}>
          <p className={styles.sourcesLabel}>Sources:</p>
          <div className={styles.sourcesList}>
            {message.sources.map((source, index) => (
              <SourceCard key={source.chunk_id || index} source={source} />
            ))}
          </div>
        </div>
      )}

      <span className={styles.timestamp}>
        {formatTime(message.timestamp)}
      </span>
    </div>
  );
}

function formatTime(date: Date): string {
  return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

function formatMarkdown(text: string): string {
  // Basic markdown formatting
  let html = text
    // Code blocks
    .replace(/```(\w*)\n([\s\S]*?)```/g, '<pre><code class="language-$1">$2</code></pre>')
    // Inline code
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    // Bold
    .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
    // Italic
    .replace(/\*([^*]+)\*/g, '<em>$1</em>')
    // Headers
    .replace(/^### (.+)$/gm, '<h4>$1</h4>')
    .replace(/^## (.+)$/gm, '<h3>$1</h3>')
    .replace(/^# (.+)$/gm, '<h2>$1</h2>')
    // Lists
    .replace(/^- (.+)$/gm, '<li>$1</li>')
    .replace(/(<li>.*<\/li>\n?)+/g, '<ul>$&</ul>')
    // Line breaks
    .replace(/\n\n/g, '</p><p>')
    .replace(/\n/g, '<br/>');

  // Wrap in paragraph if not already wrapped
  if (!html.startsWith('<')) {
    html = `<p>${html}</p>`;
  }

  return html;
}
