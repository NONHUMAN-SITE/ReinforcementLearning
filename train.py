import warnings
warnings.filterwarnings(
    "ignore",
)

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from rl.config.schemas import TrainConfig, CartPoleEnvConfig, LunarLanderEnvConfig

# Registramos las configuraciones
cs = ConfigStore.instance()
cs.store(name="train_schema", node=TrainConfig)
# Modificamos cómo registramos los esquemas para env
cs.store(group="env", name="cartpole", node=CartPoleEnvConfig, package="_target_")
cs.store(group="env", name="lunarlander", node=LunarLanderEnvConfig, package="_target_")

@hydra.main(version_base="1.2", config_path="cfg", config_name="config")
def main(cfg: TrainConfig) -> None:
    # Imprimimos la configuración para verificar
    print("\nConfiguration:")
    print(OmegaConf.to_yaml(cfg))
    
    print(f"\nEnvironment name: {cfg.env.name}")
    print(f"Max steps: {cfg.env.max_steps}")
    print(f"Training epochs: {cfg.epochs}")
    print(f"Random seed: {cfg.seed}")

if __name__ == "__main__":
    main()
