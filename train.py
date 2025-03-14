import warnings
warnings.filterwarnings(
    "ignore",
)

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from rl.config.schemas import TrainConfig, CartPoleEnvConfig, LunarLanderEnvConfig
from rl.utils.environment import load_env, register_env
# Registramos las configuraciones
cs = ConfigStore.instance()
cs.store(name="train_schema", node=TrainConfig)
register_env(cs)

@hydra.main(version_base="1.2", config_path="cfg", config_name="train_config")
def main(cfg: TrainConfig) -> None:
    # Imprimimos la configuración para verificar
    print("\nConfiguration:")
    print(OmegaConf.to_yaml(cfg))
    
    print(f"\nEnvironment name: {cfg.env.name}")
    print(f"Max steps: {cfg.env.max_steps}")
    print(f"Training epochs: {cfg.epochs}")
    print(f"Random seed: {cfg.seed}")

    env = load_env(cfg.env)
    env.reset()
    env.step(0)
    env.close()

if __name__ == "__main__":
    main()
