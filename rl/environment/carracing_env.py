import numpy as np
import gymnasium as gym
from collections import deque
from rl.config.environment import CarRacingEnvConfig
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
        self.stack_frames = cfg.stack_frames
        self.stack_frames_buffer = deque(maxlen=self.stack_frames)
        self.is_continuous = cfg.continuous

        if self.is_continuous:
            raise ValueError("CarRacingEnv continuous action space is not implemented")

    def reset(self):
        state,_ = self.env.reset()
        for _ in range(self.stack_frames):
            self.stack_frames_buffer.append(state)
        return self._process_state(self.stack_frames_buffer)
    
    def step(self, action):
        state, reward, done, truncated, info = self.env.step(action)
        self.stack_frames_buffer.append(state)
        return (self._process_state(self.stack_frames_buffer),
                reward,
                done,
                truncated,
                info)
    
    def close(self):
        return self.env.close()
    
    def render(self):
        return self.env.render()
    
    def _process_state(self, stack_frames_buffer):
        frames = np.concatenate(list(stack_frames_buffer), axis=-1)
        return frames
