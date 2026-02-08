/**
 * Custom hook for managing chat state and API calls
 */

import { useState, useCallback } from 'react';
import type {
  ChatMessage,
  ChatResponse,
  TextSelection,
} from '../types';

interface UseChatOptions {
  apiUrl: string;
}

interface UseChatReturn {
  messages: ChatMessage[];
  isLoading: boolean;
  error: string | null;
  sendMessage: (message: string, selection?: TextSelection | null) => Promise<void>;
  clearMessages: () => void;
  clearError: () => void;
}

export function useChat({ apiUrl }: UseChatOptions): UseChatReturn {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const sendMessage = useCallback(
    async (message: string, selection?: TextSelection | null) => {
      setIsLoading(true);
      setError(null);

      // Create user message
      const userMessage: ChatMessage = {
        id: generateId(),
        role: 'user',
        content: message,
        timestamp: new Date(),
      };

      setMessages((prev) => [...prev, userMessage]);

      try {
        // Build history from previous messages
        const history = messages.slice(-6).map((msg) => ({
          role: msg.role,
          content: msg.content,
        }));

        let response: Response;

        if (selection?.text) {
          // Use selection endpoint
          response = await fetch(`${apiUrl}/api/chat/selection`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              message,
              selected_text: selection.text,
              chapter_id: selection.chapterId || null,
              history,
            }),
          });
        } else {
          // Use regular chat endpoint
          response = await fetch(`${apiUrl}/api/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              message,
              history,
            }),
          });
        }

        if (!response.ok) {
          const errorData = await response.json().catch(() => ({}));
          throw new Error(errorData.detail || `HTTP error: ${response.status}`);
        }

        const data: ChatResponse = await response.json();

        // Create assistant message
        const assistantMessage: ChatMessage = {
          id: generateId(),
          role: 'assistant',
          content: data.answer,
          timestamp: new Date(),
          sources: data.sources,
        };

        setMessages((prev) => [...prev, assistantMessage]);
      } catch (err) {
        const errorMessage =
          err instanceof Error ? err.message : 'An error occurred';
        setError(errorMessage);

        // Add error message to chat
        const errorChatMessage: ChatMessage = {
          id: generateId(),
          role: 'assistant',
          content: `Sorry, I encountered an error: ${errorMessage}. Please try again.`,
          timestamp: new Date(),
        };

        setMessages((prev) => [...prev, errorChatMessage]);
      } finally {
        setIsLoading(false);
      }
    },
    [apiUrl, messages]
  );

  const clearMessages = useCallback(() => {
    setMessages([]);
    setError(null);
  }, []);

  const clearError = useCallback(() => {
    setError(null);
  }, []);

  return {
    messages,
    isLoading,
    error,
    sendMessage,
    clearMessages,
    clearError,
  };
}

function generateId(): string {
  return `${Date.now()}-${Math.random().toString(36).substring(2, 9)}`;
}
