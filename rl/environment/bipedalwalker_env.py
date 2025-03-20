import gymnasium as gym
from rl.environment.base import BaseEnv
from rl.config.environment import BipedalWalkerEnvConfig

class BipedalWalkerEnv(BaseEnv):
    name              = "bipedalwalker"
    all_envs          = ["BipedalWalker-v3"]
    render_modes      = ["human", "rgb_array"]
    observation_space = 24
    action_space      = 4
    max_steps         = 1000
    is_continuous     = True

    def __init__(self, cfg: BipedalWalkerEnvConfig):
        self.env = gym.make(cfg.name_version,
                            render_mode=cfg.render_mode,
                            hardcore=cfg.hardcore)
        self.cfg = cfg
        self.min_std = cfg.min_std
        self.init_std = cfg.init_std

    def reset(self):
        state, _ = self.env.reset()
        return state
    
    def step(self, action):
        return self.env.step(action)
    
    def close(self):
        return self.env.close()
    
    def render(self):
        return self.env.render()

