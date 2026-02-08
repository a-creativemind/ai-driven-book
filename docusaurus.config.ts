import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

const config: Config = {
  title: 'Physical AI & Humanoid Robotics',
  tagline: 'From Foundations to Embodied Intelligence',
  favicon: 'img/favicon.ico',

  // GitHub Pages deployment configuration
  url: 'https://your-username.github.io',
  baseUrl: '/physical-ai-textbook/',

  // GitHub Pages deployment settings
  organizationName: 'your-username', // GitHub org/user name
  projectName: 'physical-ai-textbook', // Repo name
  deploymentBranch: 'gh-pages',
  trailingSlash: false,

  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',

  i18n: {
    defaultLocale: 'en',
    locales: ['en', 'ur'],
    localeConfigs: {
      en: {
        label: 'English',
        direction: 'ltr',
      },
      ur: {
        label: 'اردو',
        direction: 'rtl',
      },
    },
  },

  // Math rendering support
  stylesheets: [
    {
      href: 'https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css',
      type: 'text/css',
      integrity: 'sha384-n8MVd4RsNIU0tAv4ct0nTaAbDJwPJzDEaqSD1odI+WdtXRGWt2kTvGFasHpSy3SV',
      crossorigin: 'anonymous',
    },
  ],

  presets: [
    [
      'classic',
      {
        docs: {
          sidebarPath: './sidebars.ts',
          editUrl: 'https://github.com/your-username/physical-ai-textbook/tree/main/',
          remarkPlugins: [remarkMath],
          rehypePlugins: [rehypeKatex],
          showLastUpdateTime: false,
          showLastUpdateAuthor: false,
        },
        blog: false, // Disable blog
        theme: {
          customCss: './src/css/custom.css',
        },
      } satisfies Preset.Options,
    ],
  ],

  themeConfig: {
    // Social card image
    image: 'img/social-card.jpg',

    // Announcement bar (optional)
    announcementBar: {
      id: 'ai_generated',
      content: 'This textbook is AI-generated using Claude Code with Spec-Kit Plus validation',
      backgroundColor: '#4a90d9',
      textColor: '#ffffff',
      isCloseable: true,
    },

    // Navigation bar
    navbar: {
      title: 'Physical AI Textbook',
      logo: {
        alt: 'Physical AI Logo',
        src: 'img/logo.svg',
      },
      items: [
        {
          type: 'docSidebar',
          sidebarId: 'textbookSidebar',
          position: 'left',
          label: 'Textbook',
        },
        {
          to: '/docs/labs/ros2',
          label: 'Labs',
          position: 'left',
        },
        {
          type: 'localeDropdown',
          position: 'right',
        },
        {
          href: 'https://github.com/your-username/physical-ai-textbook',
          label: 'GitHub',
          position: 'right',
        },
      ],
    },

    // Footer
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Textbook',
          items: [
            {
              label: 'Introduction',
              to: '/docs/intro',
            },
            {
              label: 'Physical AI Foundations',
              to: '/docs/physical-ai/embodiment',
            },
            {
              label: 'Humanoid Robotics',
              to: '/docs/humanoid-robotics/kinematics',
            },
          ],
        },
        {
          title: 'Learning',
          items: [
            {
              label: 'AI Systems',
              to: '/docs/ai-systems/rl',
            },
            {
              label: 'Labs',
              to: '/docs/labs/ros2',
            },
            {
              label: 'Ethics & Future',
              to: '/docs/ethics-future',
            },
          ],
        },
        {
          title: 'Resources',
          items: [
            {
              label: 'GitHub',
              href: 'https://github.com/your-username/physical-ai-textbook',
            },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} Physical AI Textbook. AI-Generated with Claude Code. Licensed under CC BY-NC-SA 4.0.`,
    },

    // Prism syntax highlighting
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
      additionalLanguages: ['python', 'bash', 'yaml', 'json', 'cpp', 'markup'],
    },

    // Table of contents depth
    tableOfContents: {
      minHeadingLevel: 2,
      maxHeadingLevel: 4,
    },

    // Color mode
    colorMode: {
      defaultMode: 'light',
      disableSwitch: false,
      respectPrefersColorScheme: true,
    },

    // Algolia DocSearch (configure when ready)
    // algolia: {
    //   appId: 'YOUR_APP_ID',
    //   apiKey: 'YOUR_SEARCH_API_KEY',
    //   indexName: 'physical-ai-textbook',
    //   contextualSearch: true,
    // },
  } satisfies Preset.ThemeConfig,

  // Plugins
  plugins: [
    // Add custom plugins here if needed
  ],

  // Custom fields for chatbot integration
  customFields: {
    chatbotEnabled: true,
    chatbotApiUrl: process.env.CHATBOT_API_URL || 'http://localhost:8000',
  },
};

export default config;
