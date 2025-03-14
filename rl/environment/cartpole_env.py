
import gymnasium as gym

from typing import Tuple, Optional
from rl.config.schemas import CartPoleEnvConfig
#from rl.environment.base import BaseEnv

class CartPoleEnv:

    all_envs          = ["CartPole-v1"]
    render_modes      = ["human", "rgb_array"]
    observation_space = (4,)
    action_space      = (2,)
    max_steps         = 500

    def __init__(self,cfg: CartPoleEnvConfig):
        self.env = gym.make("CartPole-v1",render_mode=cfg.render_mode)

    def reset(self):
        return self.env.reset()
    
    def step(self, action):
        return self.env.step(action)
    
    def close(self):
        return self.env.close()
    
    def render(self):
        return self.env.render()
    
if __name__ == "__main__":
    env = CartPoleEnv()
    obs = env.reset()
    print(env.step(1))

