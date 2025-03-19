import gymnasium as gym
from rl.config.schemas import CarRacingEnvConfig
from rl.environment.base import BaseEnv

class CarRacingEnv(BaseEnv):

    name              = "carracing"
    all_envs          = ["CarRacing-v3"]
    render_modes      = ["human", "rgb_array"]
    observation_space = 128
    action_space      = 5
    max_steps         = 1000
    is_continuous     = False
    
    def __init__(self, cfg: CarRacingEnvConfig):
        self.env = gym.make(cfg.name_version,
                            render_mode=cfg.render_mode,
                            continuous=cfg.continuous,
                            lap_complete_percent=cfg.lap_complete_percent)
        self.cfg = cfg
        self.is_continuous = cfg.continuous

        if self.is_continuous:
            raise ValueError("CarRacingEnv continuous action space is not implemented")

    def reset(self):
        state,_ = self.env.reset()
        return state
    
    def step(self, action):
        return self.env.step(action)
    
    def close(self):
        return self.env.close()
    
    def render(self):
        return self.env.render()
    
