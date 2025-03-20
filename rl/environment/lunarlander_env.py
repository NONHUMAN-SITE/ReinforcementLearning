import gymnasium as gym
from rl.config.environment import LunarLanderEnvConfig
from rl.environment.base import BaseEnv

class LunarLanderEnv(BaseEnv):

    name              = "lunarlander"
    all_envs          = ["LunarLander-v3"]
    render_modes      = ["human", "rgb_array"]
    observation_space = 8
    action_space      = 4
    max_steps         = 1000

    def __init__(self, cfg: LunarLanderEnvConfig):
        
        self.env = gym.make(cfg.name_version,
                            render_mode=cfg.render_mode,
                            enable_wind=cfg.enable_wind,
                            turbulence_power=cfg.turbulence_power,
                            wind_power=cfg.wind_power,
                            gravity=cfg.gravity)
        self.cfg = cfg

    def reset(self):
        state, _ = self.env.reset()
        return state

    def step(self, action):
        return self.env.step(action)

    def close(self):
        return self.env.close()
    
    def render(self):
        return self.env.render()
