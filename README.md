# Reinforcement Learning Framework

This repository provides a set of reinforcement learning (RL) environments and algorithms for training and experimentation. It supports various popular RL algorithms and integrates multiple environments to facilitate research and development.

## Installation

To set up the project, follow these steps:

1. **(Optional)** Create a new virtual environment using Conda:
   ```bash
   conda create --name rl_env python=3.10
   conda activate rl_env
   ```

2. Install `poetry` for dependency management:
   ```bash
   pip install poetry
   ```

3. Install dependencies from `pyproject.toml`:
   ```bash
   poetry install
   ```

## Algorithms

The repository includes implementations of the following RL algorithms:

- [x] Proximal Policy Optimization (PPO)
- [ ] Deep Q-Network (DQN)
- [ ] Double Deep Q-Network (DDQN)
- [ ] Advantage Actor-Critic (A2C)
- [ ] Asynchronous Advantage Actor-Critic (A3C)
- [ ] Soft Actor-Critic (SAC)
- [ ] Deep Deterministic Policy Gradient (DDPG)
- [ ] Twin Delayed DDPG (TD3)
- [ ] Trust Region Policy Optimization (TRPO)
- [ ] Monte Carlo Tree Search (MCTS)
- [ ] Rainbow DQN

## Environments

The repository supports multiple environments for RL training:

- [ ] OpenAI Gym
- [ ] Atari Games
- [ ] MuJoCo
- [ ] PyBullet
- [ ] Unity ML-Agents
- [ ] Roboschool
- [ ] DeepMind Control Suite
- [ ] CARLA Simulator
- [ ] StarCraft II Learning Environment (SC2LE)
- [ ] Habitat AI (Embodied AI)

## Roadmap

### Planned Features:
- [ ] Implement TRPO and Rainbow DQN
- [ ] Expand support for real-world robotics simulations
- [ ] Improve training performance with distributed computing
- [ ] Add more benchmark results and comparisons
- [ ] Implement curriculum learning strategies

## Contributing

Contributions are welcome! If you'd like to add a new algorithm or improve existing implementations, please open an issue or submit a pull request.