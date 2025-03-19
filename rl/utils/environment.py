from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from dataclasses import make_dataclass
from rl.config.schemas import (EnvConfig,
                               CartPoleEnvConfig,
                               LunarLanderEnvConfig,
                               BufferConfig,
                               PPOConfig,
                               AlgorithmConfig,
                               TrainConfig)
from rl.environment.cartpole_env import CartPoleEnv
from rl.environment.lunarlander_env import LunarLanderEnv
from rl.buffer.basic_buffer import BasicBuffer
from rl.algorithm.ppo.ppo import PPOAlgorithm
from rl.nnetworks.env.cartpole_nn import CartPoleActorCritic
from rl.nnetworks.env.lunarlander_nn import LunarLanderActorCritic
def load_env(cfg: EnvConfig):
    '''
    Load the environment based on the configuration
    '''
    if cfg.name_version in CartPoleEnv.all_envs:
        return CartPoleEnv(cfg)
    elif cfg.name_version in LunarLanderEnv.all_envs:
        return LunarLanderEnv(cfg)
    else:
        raise ValueError(f"Environment version {cfg.name_version} not found")



def load_buffer(cfg: BufferConfig):
    '''
    Load the buffer based on the configuration
    '''
    if cfg.type == "basic":
        return BasicBuffer(cfg.buffer_size, cfg.batch_size)
    else:
        raise ValueError(f"Buffer type {cfg.type} not found")



def load_algorithm(cfg: AlgorithmConfig):
    '''
    Load the algorithm based on the configuration
    '''
    if cfg.name == "ppo":
        return PPOAlgorithm(cfg)
    else:
        raise ValueError(f"Algorithm {cfg.name} not found")



def load_model(env_cfg: EnvConfig, algorithm_cfg: AlgorithmConfig):
    '''
    Load the model based on the configuration
    '''
    if env_cfg.name == "cartpole":
        if algorithm_cfg.name == "ppo":
            return CartPoleActorCritic()
        else:
            raise ValueError(f"Algorithm {algorithm_cfg.name} not found")
    elif env_cfg.name == "lunarlander":
        if algorithm_cfg.name == "ppo":
            return LunarLanderActorCritic()
        else:
            raise ValueError(f"Algorithm {algorithm_cfg.name} not found")
    else:
        raise ValueError(f"Model {env_cfg.name} not found")



def register_env(cs: ConfigStore):
    cs.store(group="env", name="cartpole", node=CartPoleEnvConfig, package="_target_")
    cs.store(group="env", name="lunarlander", node=LunarLanderEnvConfig, package="_target_")
    cs.store(group="algorithm", name="ppo", node=PPOConfig, package="_target_")


def validate_train_config(cfg: TrainConfig):
    """
    Valida que la configuración completa (TrainConfig) cumpla con el esquema esperado,
    validando de forma específica la configuración del entorno (env) y del algoritmo (algorithm)
    basándose en sus respectivos _target_.
    """
    try:
        # Validación y selección de la clase concreta para el entorno.
        env_target = OmegaConf.select(cfg.env, "_target_")
        if env_target is None:
            raise ValueError("No se proporcionó _target_ en la configuración del entorno.")
        env_class_map = {
            "rl.config.schemas.CartPoleEnvConfig": CartPoleEnvConfig,
            "rl.config.schemas.LunarLanderEnvConfig": LunarLanderEnvConfig
        }
        env_class = env_class_map.get(env_target)
        if env_class is None:
            raise ValueError(f"Tipo de entorno no soportado: {env_target}")

        # Validación y selección de la clase concreta para el algoritmo.
        algo_target = OmegaConf.select(cfg.algorithm, "_target_")
        if algo_target is None:
            raise ValueError("No se proporcionó _target_ en la configuración del algoritmo.")
        algo_class_map = {
            "rl.config.schemas.PPOConfig": PPOConfig,
            # Aquí se pueden añadir más algoritmos cuando estén disponibles.
        }
        algo_class = algo_class_map.get(algo_target)
        if algo_class is None:
            raise ValueError(f"Tipo de algoritmo no soportado: {algo_target}")

        # Se crea una dataclase dinámica que extiende de TrainConfig pero
        # con los campos 'env' y 'algorithm' tipados de forma específica.
        SpecificTrainConfig = make_dataclass(
            'SpecificTrainConfig',
            [('env', env_class), ('algorithm', algo_class)],
            bases=(TrainConfig,)
        )

        # Se convierte la configuración completa en un contenedor y se fusiona
        # con el esquema estructurado para validar que cumple con el formato.
        train_container = OmegaConf.to_container(cfg, resolve=True)
        schema = OmegaConf.structured(SpecificTrainConfig)
        merged_cfg = OmegaConf.merge(schema, train_container)
        OmegaConf.resolve(merged_cfg)
        return merged_cfg
    except Exception as e:
        raise ValueError(f"Error validando TrainConfig: {str(e)}")