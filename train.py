import warnings
warnings.filterwarnings("ignore")

import hydra
from hydra.core.config_store import ConfigStore
from rl.config.schemas import TrainConfig
from rl.utils.environment import (load_env,
                                  load_buffer,
                                  load_algorithm,
                                  load_model,
                                  register_env,
                                  validate_train_config)
# Registramos las configuraciones
cs = ConfigStore.instance()
cs.store(name="train_schema", node=TrainConfig)
register_env(cs)

@hydra.main(version_base="1.2", config_path="cfg", config_name="train_config")
def main(cfg: TrainConfig) -> None:

    cfg = validate_train_config(cfg)

    print("Train config: ", cfg.train)
    print("Env config: ", cfg.env)
    print("Algorithm config: ", cfg.algorithm)
    print("Buffer config: ", cfg.buffer)

    env = load_env(cfg.env)
    
    buffer = load_buffer(cfg.buffer)
    
    algorithm = load_algorithm(cfg.algorithm)
    
    model = load_model(cfg.env, cfg.algorithm)
    
    algorithm.set_algorithm_params(model, env, buffer, cfg.train)

    algorithm.train()




if __name__ == "__main__":
    main()
