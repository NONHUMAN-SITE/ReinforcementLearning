from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from dataclasses import make_dataclass
from rl.config.schemas import EnvConfig, CartPoleEnvConfig, LunarLanderEnvConfig, BufferConfig, PPOConfig, AlgorithmConfig
from rl.environment.cartpole_env import CartPoleEnv
from rl.environment.lunarlander_env import LunarLanderEnv
from rl.config.schemas import TrainConfig
from rl.buffer.basic_buffer import BasicBuffer
from rl.algorithm.ppo.ppo import PPOAlgorithm


def load_env(cfg: EnvConfig):
    if cfg.name in CartPoleEnv.all_envs:
        return CartPoleEnv(cfg)
    elif cfg.name in LunarLanderEnv.all_envs:
        return LunarLanderEnv(cfg)
    else:
        raise ValueError(f"Environment {cfg.name} not found")

def load_buffer(cfg: BufferConfig):
    if cfg.type == "basic":
        return BasicBuffer(cfg.buffer_size, cfg.batch_size)
    else:
        raise ValueError(f"Buffer type {cfg.type} not found")

def load_algorithm(cfg: AlgorithmConfig):
    if cfg.name == "ppo":
        return PPOAlgorithm(cfg)
    else:
        raise ValueError(f"Algorithm {cfg.name} not found")

def register_env(cs: ConfigStore):
    cs.store(group="env", name="cartpole", node=CartPoleEnvConfig, package="_target_")
    cs.store(group="env", name="lunarlander", node=LunarLanderEnvConfig, package="_target_")
    cs.store(group="algorithm", name="ppo", node=PPOConfig, package="_target_")


def validate_config(cfg: EnvConfig):
    try:
        # Obtenemos el _target_ de la configuración del entorno
        env_type = OmegaConf.select(cfg, "_target_")
        if env_type is None:
            raise ValueError("No _target_ for the environment")

        # Mapeo de _target_ a la clase específica del entorno
        env_class_map = {
            "rl.config.schemas.LunarLanderEnvConfig": LunarLanderEnvConfig,
            "rl.config.schemas.CartPoleEnvConfig": CartPoleEnvConfig
        }
        env_class = env_class_map.get(env_type)
        if env_class is None:
            raise ValueError(f"Unsupported environment type: {env_type}")

        # Creamos un dataclass específico que hereda de TrainConfig, pero con 'env' del tipo adecuado.
        SpecificTrainConfig = make_dataclass(
            'SpecificTrainConfig',
            [('env', env_class)],
            bases=(TrainConfig,)
        )

        # Envolvemos la configuración del entorno bajo la clave "env"
        env_container = {"env": OmegaConf.to_container(cfg, resolve=True)}

        # Creamos el esquema estructurado a partir de SpecificTrainConfig
        schema = OmegaConf.structured(SpecificTrainConfig)

        # Fusionamos el esquema con la configuración anidada
        merged_cfg = OmegaConf.merge(schema, env_container)

        OmegaConf.resolve(merged_cfg)
        return merged_cfg.env
    except Exception as e:
        raise ValueError(f"Error: {str(e)}")