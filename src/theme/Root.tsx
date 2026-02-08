/**
 * Root component wrapper for Docusaurus
 *
 * This component wraps the entire application and injects
 * the ChatbotWidget globally on all pages.
 */

import React from 'react';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import { ChatbotWidget } from '../components/ChatbotWidget';

interface Props {
  children: React.ReactNode;
}

export default function Root({ children }: Props): JSX.Element {
  const { siteConfig } = useDocusaurusContext();

  // Get chatbot configuration from docusaurus config
  const customFields = siteConfig.customFields || {};
  const chatbotEnabled = customFields.chatbotEnabled !== false;
  const chatbotApiUrl =
    (customFields.chatbotApiUrl as string) || 'http://localhost:8000';

  return (
    <>
      {children}
      {chatbotEnabled && (
        <ChatbotWidget apiUrl={chatbotApiUrl} position="bottom-right" />
      )}
    </>
  );
}
