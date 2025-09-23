# CAPO

<img width="1024" height="1024" alt="image" src="https://github.com/user-attachments/assets/69828e58-26ca-4b04-8ea0-6f4786239885" />

# PPO with Dynamic, Switchable Objectives

This repository contains a Python implementation of a Proximal Policy Optimization (PPO) agent enhanced with a dynamic and switchable objective system. This allows the reinforcement learning agent to dynamically change its reward function during training, enabling more complex and adaptive behaviors. The framework is designed to be modular and extensible, allowing for the easy addition of new, custom objectives.

## Table of Contents
- [Core Concepts](#core-concepts)
- [Features](#features)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [How to Use](#how-to-use)
  - [Basic Training Loop](#basic-training-loop)
  - [Configuration](#configuration)
  - [Defining a Custom Objective](#defining-a-custom-objective)
  - [Registering and Using a Custom Objective](#registering-and-using-a-custom-objective)
- [Architectural Overview](#architectural-overview)
- [Included Objective Modules](#included-objective-modules)
- [Contributing](#contributing)

## Core Concepts

The central idea behind this project is to move beyond static reward functions in reinforcement learning. By introducing a modular system for objectives, the PPO agent can be guided by different reward signals at different stages of training. This is managed by three key components:

1.  **ObjectiveModule**: An abstract base class that defines the interface for all reward-calculating modules. Each module encapsulates a specific reward logic (e.g., exploration, exploitation, similarity to a baseline).

2.  **ObjectiveRegistry**: A centralized registry that discovers and manages all available `ObjectiveModule` classes, including custom plugins.

3.  **SwitchableObjectiveManager**: This orchestrator dynamically selects and combines objectives based on a predefined configuration. It can operate in a "single" mode (switching between individual objectives) or a "multi" mode (combining multiple objectives based on priority).

The PPO agent interacts with this system to calculate a composite reward at each step, which is then used for policy updates.

## Features

*   **Modular Objective System**: Easily define and plug in new reward functions without altering the core PPO algorithm.
*   **Dynamic Switching**: Objectives can be switched periodically during training to encourage a curriculum of behaviors.
*   **Multi-Objective Rewards**: Combine multiple objectives, weighted by importance, to create complex reward signals.
*   **Baseline-Driven Objectives**: Several built-in objectives leverage a `BaselineManager` to compute rewards based on the agent's past experiences (e.g., rewarding exploration away from the mean trajectory).
*   **Extensible Plugin Architecture**: The `ObjectiveRegistry` can automatically discover and load custom objective modules from specified directories.
*   **Comprehensive Configuration**: Control all aspects of the agent and the objective system through a single configuration dictionary.

## Getting Started

### Prerequisites

*   Python 3.7+
*   PyTorch
*   NumPy

### Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/your-username/ppo-dynamic-objectives.git
    cd ppo-dynamic-objectives
    ```

2.  Install the required dependencies:
    ```bash
    pip install torch numpy
    ```
## How to Use

### Basic Training Loop

The following example demonstrates how to initialize and train the PPO agent with the dynamic objective system.

```python
import torch
import numpy as np

# Assuming the classes from run.py are in the current scope

if __name__ == "__main__":
    # 1. Define the configuration
    config = {
        "state_dim": 4,
        "action_dim": 2,
        "latent_dim": 64,
        "lr_actor": 0.0003,
        "lr_critic": 0.001,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_epsilon": 0.2,
        "epochs": 10,
        "value_loss_coeff": 0.5,
        "entropy_coeff": 0.01,
        "num_episodes": 1000,
        "max_steps_per_episode": 200,
        "baseline_update_freq": 10,
        "objective_switching_freq": 50,
        "mode": "single",  # "single" or "multi"

        # Objective Configurations
        "exploration": {
            "name": "exploration",
            "enabled": True,
            "weight": 0.1
        },
        "exploitation": {
            "name": "exploitation",
            "enabled": True,
            "weight": 0.5
        },
        "spreading": {
            "name": "spreading",
            "enabled": True,
            "weight": 0.2
        }
    }

    # 2. Initialize the PPO agent
    agent = PPOAgent(config)

    # 3. Start the training process
    agent.train()

```