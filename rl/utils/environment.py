from rl.config.schemas import EnvConfig, CartPoleEnvConfig, LunarLanderEnvConfig
from rl.environment.cartpole_env import CartPoleEnv
from hydra.core.config_store import ConfigStore

def load_env(cfg: EnvConfig):
    if cfg.name in CartPoleEnv.all_envs:
        return CartPoleEnv(cfg)
    else:
        raise ValueError(f"Environment {cfg.name} not found")


def register_env(cs: ConfigStore):
    cs.store(group="env", name="cartpole", node=CartPoleEnvConfig, package="_target_")
    cs.store(group="env", name="lunarlander", node=LunarLanderEnvConfig, package="_target_")
