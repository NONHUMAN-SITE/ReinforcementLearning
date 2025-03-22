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

## Training

To train a model, you'll need to change the cfg/train_config.yaml file to your desired settings and all the .yaml files in env/ and algorithm/ to your desired settings.

Alternatively, you can use the `train.py` script to train a model passing parameters to the script.

You can see some examples in scripts/ppo/train_lunarlander.sh

```bash
python ../../train.py \
    env=lunarlander \
    algorithm=ppo \
    train.total_timesteps=500 \
    algorithm.K_epochs=30 \
    algorithm.T_steps=300 \
    algorithm.N_actors=8 \
    algorithm.eps_clip=0.2 \
    algorithm.entropy_coef=0.01 \
    algorithm.vf_coef=1.0 \
    algorithm.gamma=0.98 \
    algorithm.gae_lambda=0.95 \
    train.save_frequency=10 \
    train.validate_frequency=10 \
    train.validate_episodes=5 \
    train.seed=42 \
    train.save_path=../../output \
    buffer.batch_size=512
```

## Model Evaluation

To evaluate a trained model, you'll need a model directory with the following structure:

```
model_directory/
├── algorithm_config.yaml
├── model.pth
├── env_config.yaml
├── eval_config.yaml
└── trainparameters_config.yaml
```

This structure will be saved by the `train.py` script when the model is trained. You can see some examples in our [HuggingFace repository](https://huggingface.co/collections/NONHUMAN-RESEARCH/reinforcement-learning-67da3666b6f6cfc4a4b2e125).

### Running Evaluations

1. Use `eval.py` to evaluate trained models:

```bash
python eval.py
```

```python
checkpoint_dir = "PATH/TO/CHECKPOINT"
```

⚠️ **Important Notes:**
- Ensure the model architecture matches the original training architecture when loading the state_dict
- Only modify environment visualization parameters in `eval_config.yaml` (e.g., `render_mode: human`)
- Changing core environment parameters may conflict with the trained policy

### Downloading Pre-trained Models

We provide pre-trained models through our HuggingFace repository. Use `download_model.py` to fetch checkpoints:

```bash
python download_model.py
```

```python
repo_id = "PATH/TO/REPO"
```

Access our collection of pre-trained models at:
[NONHUMAN-RESEARCH RL Collection](https://huggingface.co/collections/NONHUMAN-RESEARCH/reinforcement-learning-67da3666b6f6cfc4a4b2e125)

All downloaded models include the necessary configuration files for evaluation.

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

- [x] Gymnasium
   - [x] CartPole
   - [x] BipedalWalker
   - [x] LunarLander
- [x] Atari Games
   - [x] Breakout
- [ ] MuJoCo
- [ ] PyBullet
- [ ] Unity ML-Agents
- [ ] Roboschool
- [ ] DeepMind Control Suite
- [ ] CARLA Simulator
- [ ] StarCraft II Learning Environment (SC2LE)
- [ ] Habitat AI (Embodied AI)


## Contributing

Contributions are welcome! If you'd like to add a new algorithm or improve existing implementations, please open an issue or submit a pull request.