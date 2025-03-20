from dataclasses import dataclass
from typing import Any, Optional, List
from omegaconf import MISSING
import yaml
import os


# Environment Config
@dataclass
class BaseConfig:
    def save_config(self, path: str):
        """Save configuration to yaml file"""
        # Create config directory if it doesn't exist
        os.makedirs(path, exist_ok=True)
        
        # Get class name without 'Config' suffix
        if isinstance(self, EnvConfig):
            class_name = "env"
        elif isinstance(self, TrainParameters):
            class_name = "trainparameters"
        elif isinstance(self, BufferConfig):
            class_name = "buffer"
        elif isinstance(self, AlgorithmConfig):
            class_name = "algorithm"
        else:
            class_name = self.__class__.__name__.replace('Config', '').lower()
        config_path = os.path.join(path, f"{class_name}_config.yaml")
        
        # Convert dataclass to dict
        config_dict = {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
        
        # Save to yaml
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)


@dataclass
class EnvConfig(BaseConfig):
    _target_: str = MISSING
    name: str = MISSING
    name_version: str = MISSING
    max_steps: int = 1000
    render_mode: str = MISSING


# Training Parameters
@dataclass
class TrainParameters(BaseConfig):
    epochs: int = 100
    seed: int = 42
    learning_rate: float = 0.001
    gamma: float = 0.99
    epsilon: float = 1.0
    max_steps: int = 500
    total_timesteps: int = 100000
    validate_frequency: int = 100
    validate_episodes: int = 5
    save_frequency: int = 1000
    save_path: str = MISSING

@dataclass
class EvalParameters(BaseConfig):
    episodes: int = 10
    render: bool = False
    save_path: str = MISSING

# Buffer Config
@dataclass
class BufferConfig(BaseConfig):
    type: str = "basic"
    buffer_size: int = 10000
    batch_size: int = 4096

# Algorithm Config
@dataclass
class AlgorithmConfig(BaseConfig):
    _target_: str = MISSING
    name: str = MISSING


# Train Config
@dataclass
class TrainConfig(BaseConfig):
    defaults: List[Any] = MISSING
    env: EnvConfig = MISSING
    algorithm: AlgorithmConfig = MISSING
    train: TrainParameters = TrainParameters()
    buffer: BufferConfig = BufferConfig()

@dataclass
class EvalConfig(BaseConfig):
    defaults: List[Any] = MISSING
    env: EnvConfig = MISSING
    algorithm: AlgorithmConfig = MISSING
    eval: EvalParameters = EvalParameters()
    