from dataclasses import dataclass
from typing import Any, Optional, List
from omegaconf import MISSING

@dataclass
class EnvConfig:
    _target_: str = MISSING
    name: str = MISSING
    max_steps: int = 1000

@dataclass
class CartPoleEnvConfig(EnvConfig):
    reward_threshold: float = 195.0
    max_angle: float = 12.0

    def __post_init__(self):
        if self.reward_threshold < 0:
            raise ValueError("reward_threshold must be positive")
        if not 0 < self.max_angle <= 45:
            raise ValueError("max_angle must be between 0 and 45 degrees")

@dataclass
class LunarLanderEnvConfig(EnvConfig):
    reward_threshold: float = 200.0
    enable_wind: bool = False
    gravity: float = -10.0

    def __post_init__(self):
        if self.reward_threshold < 0:
            raise ValueError("reward_threshold must be positive")
        if self.gravity >= 0:
            raise ValueError("gravity must be negative")

@dataclass
class TrainConfig:
    defaults: List[Any] = MISSING
    env: EnvConfig = MISSING
    train: dict = MISSING
    epochs: int = 100
    seed: int = 42
