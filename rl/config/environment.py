from dataclasses import dataclass
from rl.config.schemas import BaseConfig,EnvConfig


@dataclass
class CartPoleEnvConfig(EnvConfig):
    render_mode: str = "human"

@dataclass
class LunarLanderEnvConfig(EnvConfig):
    enable_wind: bool = False
    turbulence_power: float = 0.0
    wind_power: float = 15.0
    gravity: float = -10.0
    render_mode: str = "human"

@dataclass
class BipedalWalkerEnvConfig(EnvConfig):
    render_mode: str = "human"
    hardcore: bool = False
    min_std: float = 0.1 # Is continuous action space
    init_std: float = 0.6 # Is continuous action space

@dataclass
class CarRacingEnvConfig(EnvConfig):
    render_mode: str = "human"
    continuous: bool = True
    min_std: float = 0.1 # Is continuous action space
    init_std: float = 0.6 # Is continuous action space
    lap_complete_percent: float = 0.95
    stack_frames: int = 4

@dataclass
class BreakoutEnvConfig(EnvConfig):
    render_mode: str = "human"
    obs_type: str = "rgb"
    stack_frames: int = 4
    full_action_space: bool = False