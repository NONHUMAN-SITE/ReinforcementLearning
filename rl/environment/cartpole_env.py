
import gymnasium as gym
from rl.environment.base import BaseEnv

class CartPoleEnv(BaseEnv):
    def __init__(self, render_mode: str = None):
        self.env = gym.make("CartPole-v1",render_mode=render_mode)

    def reset(self):
        return self.env.reset()
    
    def step(self, action):
        return self.env.step(action)
    
    def close(self):
        return self.env.close()