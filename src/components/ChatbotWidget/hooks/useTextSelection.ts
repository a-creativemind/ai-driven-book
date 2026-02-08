/**
 * Custom hook for capturing text selection on the page
 */

import { useState, useEffect, useCallback } from 'react';
import type { TextSelection } from '../types';

interface UseTextSelectionReturn {
  selection: TextSelection | null;
  clearSelection: () => void;
}

export function useTextSelection(): UseTextSelectionReturn {
  const [selection, setSelection] = useState<TextSelection | null>(null);

  const handleSelectionChange = useCallback(() => {
    const windowSelection = window.getSelection();

    if (!windowSelection || windowSelection.isCollapsed) {
      return;
    }

    const selectedText = windowSelection.toString().trim();

    // Only capture meaningful selections (at least 10 characters)
    if (selectedText.length < 10) {
      return;
    }

    // Try to find chapter ID from the page context
    const chapterId = getChapterIdFromContext();

    setSelection({
      text: selectedText,
      chapterId,
    });
  }, []);

  useEffect(() => {
    // Listen for mouseup to capture selection
    const handleMouseUp = () => {
      // Small delay to ensure selection is complete
      setTimeout(handleSelectionChange, 10);
    };

    // Listen for keyboard selection (Shift+Arrow keys)
    const handleKeyUp = (e: KeyboardEvent) => {
      if (e.shiftKey) {
        setTimeout(handleSelectionChange, 10);
      }
    };

    document.addEventListener('mouseup', handleMouseUp);
    document.addEventListener('keyup', handleKeyUp);

    return () => {
      document.removeEventListener('mouseup', handleMouseUp);
      document.removeEventListener('keyup', handleKeyUp);
    };
  }, [handleSelectionChange]);

  const clearSelection = useCallback(() => {
    setSelection(null);
    // Also clear the browser selection
    window.getSelection()?.removeAllRanges();
  }, []);

  return {
    selection,
    clearSelection,
  };
}

/**
 * Try to extract chapter ID from the current page context
 */
function getChapterIdFromContext(): string | undefined {
  // Try to get from URL path
  const path = window.location.pathname;
  const docsMatch = path.match(/\/docs\/(?:[\w-]+\/)?(\w[\w-]*)/);
  if (docsMatch) {
    return docsMatch[1];
  }

  // Try to get from meta tag
  const metaTag = document.querySelector('meta[name="chapter-id"]');
  if (metaTag) {
    return metaTag.getAttribute('content') || undefined;
  }

  // Try to get from article data attribute
  const article = document.querySelector('article[data-chapter-id]');
  if (article) {
    return article.getAttribute('data-chapter-id') || undefined;
  }

  return undefined;
}
