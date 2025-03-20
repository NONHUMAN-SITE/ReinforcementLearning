import ale_py
import gymnasium as gym
from rl.config.schemas import BreakoutEnvConfig
from rl.environment.base import BaseEnv

class BreakoutEnv(BaseEnv):

    name              = "breakout"
    all_envs          = ["ALE/Breakout-v5"]
    render_modes      = ["human", "rgb_array"]
    observation_space = 128
    action_space      = 4
    is_continuous     = False

    def __init__(self, cfg: BreakoutEnvConfig):
        self.cfg = cfg
        gym.register_envs(ale_py)
        self.env = gym.make(cfg.name_version,   
                            render_mode=cfg.render_mode,
                            obs_type=cfg.obs_type)

    def reset(self):
        state,_ = self.env.reset()
        return state

    def step(self, action):
        return self.env.step(action)

    def close(self):
        return self.env.close()

    def render(self):
        return self.env.render()
