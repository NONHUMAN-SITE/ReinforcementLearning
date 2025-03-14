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
    render_mode: str = "human"


@dataclass
class LunarLanderEnvConfig(EnvConfig):
    reward_threshold: float = 200.0
    enable_wind: bool = False
    gravity: float = -10.0


@dataclass
class TrainConfig:
    defaults: List[Any] = MISSING
    env: EnvConfig = MISSING
    train: dict = MISSING
    epochs: int = 100
    seed: int = 42
