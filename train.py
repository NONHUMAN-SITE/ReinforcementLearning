import warnings
warnings.filterwarnings("ignore")

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from typing import Any
from dataclasses import make_dataclass

from rl.config.schemas import TrainConfig, LunarLanderEnvConfig, CartPoleEnvConfig
from rl.utils.environment import load_env, register_env, validate_env

# Registramos las configuraciones
cs = ConfigStore.instance()
cs.store(name="train_schema", node=TrainConfig)
register_env(cs)

@hydra.main(version_base="1.2", config_path="cfg", config_name="train_config")
def main(cfg: TrainConfig) -> None:
    
    validate_env(cfg.env)


if __name__ == "__main__":
    main()
