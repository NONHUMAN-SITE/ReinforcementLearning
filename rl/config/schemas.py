from dataclasses import dataclass
from typing import Any, Optional, List
from omegaconf import MISSING

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


@dataclass
class TrainParameters:
    epochs: int = 100
    seed: int = 42
    buffer_size: int = 10000
    batch_size: int = 64
    learning_rate: float = 0.001
    gamma: float = 0.99
    epsilon: float = 1.0
    update_frequency: int = 1000
    target_update_frequency: int = 1000
    max_steps: int = 500

@dataclass
class BufferConfig:
    type: str = "basic"
    buffer_size: int = 10000

@dataclass
class TrainConfig:
    defaults: List[Any] = MISSING
    env: EnvConfig = MISSING
    train: TrainParameters = TrainParameters()
    buffer: BufferConfig = BufferConfig()
