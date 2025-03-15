from dataclasses import dataclass
from typing import Any, Optional, List
from omegaconf import MISSING


# Environment Config
@dataclass
class EnvConfig:
    _target_: str = MISSING
    name: str = MISSING
    max_steps: int = 1000
    render_mode: str = MISSING

@dataclass
class CartPoleEnvConfig(EnvConfig):
    render_mode: str = "human"


@dataclass
class LunarLanderEnvConfig(EnvConfig):
    reward_threshold: float = 200.0
    enable_wind: bool = False
    gravity: float = -10.0

# Training Parameters
@dataclass
class TrainParameters:
    epochs: int = 100
    seed: int = 42
    learning_rate: float = 0.001
    gamma: float = 0.99
    epsilon: float = 1.0
    update_frequency: int = 1000
    target_update_frequency: int = 1000
    max_steps: int = 500

# Buffer Config
@dataclass
class BufferConfig:
    type: str = "basic"
    buffer_size: int = 10000
    batch_size: int = 64

# Algorithm Config
@dataclass
class AlgorithmConfig:
    _target_: str = MISSING
    name: str = MISSING

# PPO Config
@dataclass
class PPOConfig(AlgorithmConfig):
    _target_: str = "rl.config.schemas.PPOConfig"
    name: str = "ppo"
    eps_clip: float = 0.2
    entropy_coef: float = 0.01
    vf_coef: float = 0.5
    K_epochs: int = 5
    N_actors: int = 1
    T_steps: int = 2048


# Train Config
@dataclass
class TrainConfig:
    defaults: List[Any] = MISSING
    env: EnvConfig = MISSING
    algorithm: AlgorithmConfig = MISSING
    train: TrainParameters = TrainParameters()
    buffer: BufferConfig = BufferConfig()
