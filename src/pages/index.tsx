import React from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import styles from './index.module.css';

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={clsx('hero hero--primary', styles.heroBanner)}>
      <div className="container">
        <h1 className="hero__title">{siteConfig.title}</h1>
        <p className="hero__subtitle">{siteConfig.tagline}</p>
        <div className={styles.buttons}>
          <Link
            className="button button--secondary button--lg"
            to="/docs/intro">
            Start Reading
          </Link>
        </div>
      </div>
    </header>
  );
}

type FeatureItem = {
  title: string;
  description: string;
  icon: string;
};

const FeatureList: FeatureItem[] = [
  {
    title: 'Physical AI Foundations',
    description: 'Understand embodied intelligence, sensors, actuators, and control systems that enable robots to interact with the physical world.',
    icon: '🤖',
  },
  {
    title: 'Humanoid Robotics',
    description: 'Dive deep into kinematics, locomotion, manipulation, and perception for human-like robotic systems.',
    icon: '🚶',
  },
  {
    title: 'Learning Systems',
    description: 'Master reinforcement learning, imitation learning, and foundation models for training intelligent robots.',
    icon: '🧠',
  },
  {
    title: 'Hands-on Labs',
    description: 'Practical experience with ROS 2, Isaac Sim, and MuJoCo through guided lab exercises.',
    icon: '🔬',
  },
];

function Feature({title, description, icon}: FeatureItem) {
  return (
    <div className={clsx('col col--3')}>
      <div className="text--center padding-horiz--md">
        <div style={{fontSize: '3rem', marginBottom: '1rem'}}>{icon}</div>
        <h3>{title}</h3>
        <p>{description}</p>
      </div>
    </div>
  );
}

export default function Home(): JSX.Element {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title={`${siteConfig.title}`}
      description="Comprehensive textbook on Physical AI and Humanoid Robotics">
      <HomepageHeader />
      <main>
        <section className={styles.features}>
          <div className="container">
            <div className="row">
              {FeatureList.map((props, idx) => (
                <Feature key={idx} {...props} />
              ))}
            </div>
          </div>
        </section>
        <section className={styles.cta}>
          <div className="container">
            <h2>AI-Generated with Spec-Driven Development</h2>
            <p>
              This textbook is authored using Claude Code with Spec-Kit Plus validation,
              ensuring consistent structure, measurable learning objectives, and validated code examples.
            </p>
            <div className={styles.buttons}>
              <Link
                className="button button--primary button--lg"
                to="/docs/physical-ai/embodiment">
                Begin with Embodied Intelligence
              </Link>
            </div>
          </div>
        </section>
      </main>
    </Layout>
  );
}
