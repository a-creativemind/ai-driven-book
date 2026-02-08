import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

/**
 * Sidebar configuration for Physical AI & Humanoid Robotics Textbook
 * Organized into 5 parts as defined in book.spec.yaml
 */
const sidebars: SidebarsConfig = {
  textbookSidebar: [
    // Introduction
    {
      type: 'doc',
      id: 'intro',
      label: 'Introduction',
    },

    // Part I: Physical AI Foundations
    {
      type: 'category',
      label: 'Part I: Physical AI Foundations',
      collapsed: false,
      items: [
        {
          type: 'doc',
          id: 'physical-ai/embodiment',
          label: '1.1 Embodied Intelligence',
        },
        {
          type: 'doc',
          id: 'physical-ai/sensors-actuators',
          label: '1.2 Sensors & Actuators',
        },
        {
          type: 'doc',
          id: 'physical-ai/control-systems',
          label: '1.3 Control Systems',
        },
        {
          type: 'doc',
          id: 'physical-ai/sim2real',
          label: '1.4 Sim-to-Real Transfer',
        },
      ],
    },

    // Part II: Humanoid Robotics
    {
      type: 'category',
      label: 'Part II: Humanoid Robotics',
      collapsed: true,
      items: [
        {
          type: 'doc',
          id: 'humanoid-robotics/kinematics',
          label: '2.1 Kinematics & Dynamics',
        },
        {
          type: 'doc',
          id: 'humanoid-robotics/locomotion',
          label: '2.2 Bipedal Locomotion',
        },
        {
          type: 'doc',
          id: 'humanoid-robotics/manipulation',
          label: '2.3 Dexterous Manipulation',
        },
        {
          type: 'doc',
          id: 'humanoid-robotics/perception',
          label: '2.4 Robot Perception',
        },
      ],
    },

    // Part III: Learning Systems
    {
      type: 'category',
      label: 'Part III: Learning Systems',
      collapsed: true,
      items: [
        {
          type: 'doc',
          id: 'ai-systems/rl',
          label: '3.1 Reinforcement Learning',
        },
        {
          type: 'doc',
          id: 'ai-systems/imitation-learning',
          label: '3.2 Imitation Learning',
        },
        {
          type: 'doc',
          id: 'ai-systems/foundation-models',
          label: '3.3 Foundation Models for Robotics',
        },
      ],
    },

    // Part IV: Tooling & Labs
    {
      type: 'category',
      label: 'Part IV: Tooling & Labs',
      collapsed: true,
      items: [
        {
          type: 'doc',
          id: 'labs/ros2',
          label: '4.1 ROS 2 Fundamentals',
        },
        {
          type: 'doc',
          id: 'labs/isaac-sim',
          label: '4.2 NVIDIA Isaac Sim',
        },
        {
          type: 'doc',
          id: 'labs/mujoco',
          label: '4.3 MuJoCo Physics Simulation',
        },
      ],
    },

    // Part V: Ethics & Future
    {
      type: 'category',
      label: 'Part V: Ethics & Future',
      collapsed: true,
      items: [
        {
          type: 'doc',
          id: 'ethics-future',
          label: '5.1 Safety, Alignment & HRI',
        },
      ],
    },
  ],
};

export default sidebars;
