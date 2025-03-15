from rl.config.schemas import LunarLanderEnvConfig


class LunarLanderEnv:
    all_envs = ["LunarLander-v2"]

    def __init__(self, cfg: LunarLanderEnvConfig):
        self.cfg = cfg

    def reset(self):
        pass

    def step(self, action):
        pass

    def close(self):
        pass
