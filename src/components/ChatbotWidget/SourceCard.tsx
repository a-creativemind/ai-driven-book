/**
 * SourceCard component for displaying citation sources
 */

import React, { useState } from 'react';
import type { SourceReference } from './types';
import styles from './ChatbotWidget.module.css';

interface SourceCardProps {
  source: SourceReference;
}

export function SourceCard({ source }: SourceCardProps): JSX.Element {
  const [expanded, setExpanded] = useState(false);

  const scorePercent = Math.round(source.similarity_score * 100);
  const difficultyColor = getDifficultyColor(source.difficulty);

  return (
    <div className={styles.sourceCard}>
      <div
        className={styles.sourceHeader}
        onClick={() => setExpanded(!expanded)}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => e.key === 'Enter' && setExpanded(!expanded)}
      >
        <div className={styles.sourceTitle}>
          <span className={styles.chapterTitle}>{source.chapter_title}</span>
          {source.section_title && (
            <span className={styles.sectionTitle}> - {source.section_title}</span>
          )}
        </div>
        <div className={styles.sourceMeta}>
          <span
            className={styles.difficultyBadge}
            style={{ backgroundColor: difficultyColor }}
          >
            {source.difficulty}
          </span>
          <span className={styles.scoreIndicator} title={`${scorePercent}% match`}>
            {getScoreIcon(source.similarity_score)}
          </span>
          <span className={styles.expandIcon}>{expanded ? '−' : '+'}</span>
        </div>
      </div>

      {expanded && (
        <div className={styles.sourcePreview}>
          <p>{source.content_preview}</p>
        </div>
      )}
    </div>
  );
}

function getDifficultyColor(difficulty: string): string {
  switch (difficulty.toLowerCase()) {
    case 'beginner':
      return '#28a745';
    case 'intermediate':
      return '#ffc107';
    case 'advanced':
      return '#dc3545';
    default:
      return '#6c757d';
  }
}

function getScoreIcon(score: number): string {
  if (score >= 0.9) return '★★★';
  if (score >= 0.8) return '★★☆';
  if (score >= 0.7) return '★☆☆';
  return '☆☆☆';
}
