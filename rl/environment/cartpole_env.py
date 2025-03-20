import gymnasium as gym
from rl.config.environment import CartPoleEnvConfig
from rl.environment.base import BaseEnv

class CartPoleEnv(BaseEnv):

    name              = "cartpole"
    all_envs          = ["CartPole-v1"]
    render_modes      = ["human", "rgb_array"]
    observation_space = 4
    action_space      = 2
    max_steps         = 500

    def __init__(self,cfg: CartPoleEnvConfig):
        self.env = gym.make(cfg.name_version,
                            render_mode=cfg.render_mode)
        self.cfg = cfg

    def reset(self):
        state,_ = self.env.reset()
        return state
    
    def step(self, action):
        return self.env.step(action)
        
    def close(self):
        return self.env.close()
    
    def render(self):
        return self.env.render()
    
if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    obs = env.reset()
    print(env.step(1))

