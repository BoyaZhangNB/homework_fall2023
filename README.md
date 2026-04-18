# CS285 Homework Repository (Fall 2023)

This repository contains coursework implementations for Berkeley CS285: Deep Reinforcement Learning, Decision Making, and Control.

It is organized as a sequence of homework modules that progressively cover imitation learning, policy gradients, value-based and actor-critic methods, model-based control, and offline RL.

## Repository Overview

- Coursework structure: each `hwX/` folder is a self-contained assignment with its own code, configs, scripts, and dataset outputs.
- Common pattern: algorithm implementations live under each homework's `cs285/` package, with runnable entry points in `cs285/scripts/`.
- Experiment artifacts: training runs are saved under homework-specific `data/` directories.

## Homework Index

1. [HW1: Imitation Learning (Behavior Cloning + DAgger)](hw1/README.md)
2. [HW2: Policy Gradient and Advantage Estimation](hw2/README.md)
3. [HW3: DQN and Soft Actor-Critic](hw3/README.md)
4. [HW4: Model-Based RL, MPC, and MBPO](hw4/README.txt)
5. [HW5: Exploration and Offline RL (RND, CQL, AWAC, IQL)](hw5/README.txt)

## Homework Summaries

### HW1
- Focus: imitation learning from expert demonstrations.
- Main methods: Behavior Cloning and DAgger.
- Key output: supervised policy training and iterative dataset aggregation.

### HW2
- Focus: on-policy policy gradient training.
- Main methods: vanilla PG, reward-to-go, baseline critic, GAE, normalized advantages.
- Key output: actor-critic style PG pipeline with trajectory-based updates.

### HW3
- Focus: off-policy deep RL for continuous and discrete tasks.
- Main methods: DQN (with optional Double DQN) and SAC.
- Key output: replay-buffer training, target-network updates, entropy-regularized actor-critic.

### HW4
- Focus: model-based control and model-based policy optimization.
- Main methods: learned ensemble dynamics, MPC random shooting/CEM, optional MBPO with SAC.
- Key output: planning from learned dynamics and synthetic-rollout augmentation.

### HW5
- Focus: exploration data collection and offline RL algorithms.
- Main methods (assignment targets): RND, CQL, AWAC, IQL.
- Key output: scripts and config pipeline for exploration, offline training, and finetuning; some agent logic remains template/TODO depending on file.

## Getting Started

- Start with a homework-specific README from the index above.
- Install dependencies inside each homework folder (`requirements.txt`, `setup.py`).
- Run experiments from that homework's scripts/configs.
