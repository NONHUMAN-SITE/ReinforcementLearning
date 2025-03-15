from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from dataclasses import make_dataclass
from rl.config.schemas import EnvConfig, CartPoleEnvConfig, LunarLanderEnvConfig
from rl.environment.cartpole_env import CartPoleEnv
from rl.environment.lunarlander_env import LunarLanderEnv
from rl.config.schemas import TrainConfig


def load_env(cfg: EnvConfig):
    if cfg.name in CartPoleEnv.all_envs:
        return CartPoleEnv(cfg)
    elif cfg.name in LunarLanderEnv.all_envs:
        return LunarLanderEnv(cfg)
    else:
        raise ValueError(f"Environment {cfg.name} not found")


def register_env(cs: ConfigStore):
    cs.store(group="env", name="cartpole", node=CartPoleEnvConfig, package="_target_")
    cs.store(group="env", name="lunarlander", node=LunarLanderEnvConfig, package="_target_")


def validate_env(cfg: EnvConfig):
    try:
        
        env_type = OmegaConf.select(cfg, "env._target_")
        if env_type is None:
            raise ValueError("No _target_ for the environment")

        env_class_map = {
            "rl.config.schemas.LunarLanderEnvConfig": LunarLanderEnvConfig,
            "rl.config.schemas.CartPoleEnvConfig": CartPoleEnvConfig
        }
        
        env_class = env_class_map.get(env_type)
        if env_class is None:
            raise ValueError(f"Unsupported environment type: {env_type}")

        SpecificTrainConfig = make_dataclass(
            'SpecificTrainConfig',
            [('env', env_class)],
            bases=(TrainConfig,)
        )

        cfg = OmegaConf.create(OmegaConf.to_yaml(cfg))
        schema = OmegaConf.structured(SpecificTrainConfig)
        cfg = OmegaConf.merge(schema, cfg)
        
        OmegaConf.resolve(cfg)
    except Exception as e:
        raise ValueError(f"Error: {str(e)}")