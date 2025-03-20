from dataclasses import dataclass
from rl.config.schemas import AlgorithmConfig

@dataclass
class PPOConfig(AlgorithmConfig):
    _target_: str = "rl.config.algorithms.PPOConfig"
    name: str = "ppo"
    eps_clip: float = 0.2
    entropy_coef: float = 0.01
    vf_coef: float = 1.0
    K_epochs: int = 15
    N_actors: int = 4
    T_steps: int = 2048
    gamma: float = 0.99
    gae_lambda: float = 0.95
