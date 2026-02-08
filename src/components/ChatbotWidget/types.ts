/**
 * Type definitions for the ChatbotWidget
 */

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  sources?: SourceReference[];
}

export interface SourceReference {
  chunk_id: string;
  chapter_id: string;
  chapter_title: string;
  section_title?: string;
  content_preview: string;
  similarity_score: number;
  difficulty: string;
}

export interface ChatRequest {
  message: string;
  history: Array<{
    role: string;
    content: string;
  }>;
}

export interface SelectionChatRequest {
  message: string;
  selected_text: string;
  chapter_id?: string;
  history: Array<{
    role: string;
    content: string;
  }>;
}

export interface ChatResponse {
  answer: string;
  sources: SourceReference[];
  tokens_used?: number;
}

export interface TextSelection {
  text: string;
  chapterId?: string;
}

export interface ChatbotWidgetProps {
  apiUrl?: string;
  position?: 'bottom-right' | 'bottom-left';
  defaultOpen?: boolean;
}
