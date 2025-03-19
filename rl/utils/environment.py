from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from dataclasses import make_dataclass
from rl.config.schemas import (EnvConfig,
                               CartPoleEnvConfig,
                               LunarLanderEnvConfig,
                               BipedalWalkerEnvConfig,
                               BufferConfig,
                               PPOConfig,
                               AlgorithmConfig,
                               TrainConfig)
from rl.environment.cartpole_env import CartPoleEnv
from rl.environment.lunarlander_env import LunarLanderEnv
from rl.environment.bipedalwalker_env import BipedalWalkerEnv
from rl.buffer.basic_buffer import BasicBuffer
from rl.algorithm.ppo.ppo import PPOAlgorithm
from rl.nnetworks.env.cartpole_nn import CartPoleActorCritic
from rl.nnetworks.env.lunarlander_nn import LunarLanderActorCritic
from rl.nnetworks.env.bipedalwalker_nn import BipedalWalkerActorCritic

def load_env(cfg: EnvConfig):
    '''
    Load the environment based on the configuration
    '''
    if cfg.name_version in CartPoleEnv.all_envs:
        return CartPoleEnv(cfg)
    elif cfg.name_version in LunarLanderEnv.all_envs:
        return LunarLanderEnv(cfg)
    elif cfg.name_version in BipedalWalkerEnv.all_envs:
        return BipedalWalkerEnv(cfg)
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
    elif env_cfg.name == "bipedalwalker":
        if algorithm_cfg.name == "ppo":
            return BipedalWalkerActorCritic()
        else:
            raise ValueError(f"Algorithm {algorithm_cfg.name} not found")
    else:
        raise ValueError(f"Model {env_cfg.name} not found")



def register_env(cs: ConfigStore):
    cs.store(group="env", name="cartpole", node=CartPoleEnvConfig, package="_target_")
    cs.store(group="env", name="lunarlander", node=LunarLanderEnvConfig, package="_target_")
    cs.store(group="env", name="bipedalwalker", node=BipedalWalkerEnvConfig, package="_target_")
    cs.store(group="algorithm", name="ppo", node=PPOConfig, package="_target_")


def validate_train_config(cfg: TrainConfig):
    """
    Validates that the complete configuration (TrainConfig) meets the expected schema,
    validating specifically the environment configuration (env) and the algorithm configuration (algorithm)
    based on their respective _target_.
    """
    try:
        # Validation and selection of the concrete class for the environment.
        env_target = OmegaConf.select(cfg.env, "_target_")
        if env_target is None:
            raise ValueError("No provided _target_ in the environment configuration.")
        env_class_map = {
            "rl.config.schemas.CartPoleEnvConfig": CartPoleEnvConfig,
            "rl.config.schemas.LunarLanderEnvConfig": LunarLanderEnvConfig,
            "rl.config.schemas.BipedalWalkerEnvConfig": BipedalWalkerEnvConfig,
        }
        env_class = env_class_map.get(env_target)
        if env_class is None:
            raise ValueError(f"Type of environment not supported: {env_target}")

        # Validation and selection of the concrete class for the algorithm.
        algo_target = OmegaConf.select(cfg.algorithm, "_target_")
        if algo_target is None:
            raise ValueError("No provided _target_ in the algorithm configuration.")
        algo_class_map = {
            "rl.config.schemas.PPOConfig": PPOConfig,
            # Here you can add more algorithms when they are available.
        }
        algo_class = algo_class_map.get(algo_target)
        if algo_class is None:
            raise ValueError(f"Type of algorithm not supported: {algo_target}")

        # A dynamic dataclass is created that extends TrainConfig but
        # with the 'env' and 'algorithm' fields typed specifically.
        SpecificTrainConfig = make_dataclass(
            'SpecificTrainConfig',
            [('env', env_class), ('algorithm', algo_class)],
            bases=(TrainConfig,)
        )

        # The complete configuration is converted into a container and merged
        # with the structured schema to validate that it meets the format.
        train_container = OmegaConf.to_container(cfg, resolve=True)
        schema = OmegaConf.structured(SpecificTrainConfig)
        merged_cfg = OmegaConf.merge(schema, train_container)
        OmegaConf.resolve(merged_cfg)
        return merged_cfg
    except Exception as e:
        raise ValueError(f"Error validating TrainConfig: {str(e)}")