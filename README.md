------

# SMOF: Surrogate-Assisted Multi-Objective Optimization Framework

### Based on LLM-Enhanced Computation Experiment for Intelligent Marketing

This repository contains the official implementation of **SMOF**, a novel framework designed for intelligent marketing decision-making on digital platforms. The framework addresses the challenges of high computational costs in simulation, conflicting marketing objectives, and the need for semantic-aware data synthesis.

------

## 🔬 Overview

Traditional marketing optimization often struggles with the "curse of dimensionality" and the expensive nature of high-fidelity simulations. **SMOF** introduces an integrated pipeline:

1. **LLM-CE (Large Language Model-Enhanced Computation Experiment):** Utilizes LLMs to extract consumer sentiment and refine noisy platform data into reliable scenario-based training sets.
2. **Heterogeneous Surrogate Integration:** Employs an ensemble of regression models (KNN, SVR, Decision Trees) to approximate complex objective functions, reducing evaluation time from minutes to milliseconds.
3. **DRL-Guided MOO:** Combines Deep Reinforcement Learning (DQN) with Multi-Objective Evolutionary Algorithms (NSGA-III variant) to navigate the Pareto frontier efficiently, guided by a specialized Solution Discriminator.

------

## 🏗 System Architecture

The framework is architected into three discrete layers:

### 1. Data Distillation & Simulation

- **Semantic Cleaning:** Raw comments from e-commerce platforms (e.g., JD.com) are processed via LLM for sentiment polarity and attribute extraction.
- **Scenario Generation:** Rules-based simulation generates synthetic experimental data to cover edge cases in marketing dynamics.

### 2. Surrogate Modeling

- **Objective Mapping:** Training $f: \mathcal{X} \to \mathcal{Y}$ where $\mathcal{X}$ includes pricing, technical investment, and advertising spend, and $\mathcal{Y}$ represents Corporate Profit, Platform Profit, and Consumer Utility.

### 3. Optimization Engine

- **Agent-Environment Interaction:** A DQN agent learns to select optimal evolutionary operators.
- **Knowledge Guidance:** A pre-trained discriminator evaluates the feasibility of solutions, accelerating convergence toward the Pareto front.



------

## 🚀 Execution Guide

### Prerequisites

- Python 3.9+

Bash

```
pip install torch pandas numpy scikit-learn pymoo DrissionPage matplotlib
```

### Workflow

1. **Data Acquisition:**

   Extract real-world marketing data using the automated crawler:

   Bash

   ```
   python get_review/mi_14.py
   ```

2. **Pareto Optimization:**

   Execute the DRL-MOO engine to find the optimal marketing mix:

   Bash

   ```
   python DATA-DQN-EA.py
   ```

------

## 📈 Key Technical Features

- **Asynchronous Crawling:** Utilizes `DrissionPage` for robust, browser-level data extraction, bypassing common anti-scraping mechanisms.
- **Ensemble Surrogates:** Implements a multi-model voting/switching mechanism to ensure prediction stability across different regions of the design space.
- **Deep RL Integration:** The DQN agent optimizes the exploration-exploitation balance of the evolutionary search, significantly outperforming static MOO baselines in Hypervolume (HV) and Inverted Generational Distance (IGD) metrics.

------

## 🛠 Methodology Principles

The framework operates on the principle of **Knowledge-Data Dual Drive**. While data provides the empirical basis via surrogates, the "Knowledge" (distilled via LLMs and the Solution Discriminator) constrains the search space to physically and economically meaningful regions.

**Research Area:** Computational Intelligence & Evolutionary Computation