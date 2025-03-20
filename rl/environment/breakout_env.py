import ale_py
import numpy as np
import gymnasium as gym
from collections import deque
from rl.config.environment import BreakoutEnvConfig
from rl.environment.base import BaseEnv

class BreakoutEnv(BaseEnv):

    name              = "breakout"
    all_envs          = ["ALE/Breakout-v5"]
    render_modes      = ["human", "rgb_array"]
    observation_space = 128
    is_continuous     = False

    def __init__(self, cfg: BreakoutEnvConfig):
        self.cfg = cfg
        gym.register_envs(ale_py)
        self.env = gym.make(cfg.name_version,   
                            render_mode=cfg.render_mode,
                            obs_type=cfg.obs_type,
                            full_action_space=cfg.full_action_space)
        self.action_space = 18 if cfg.full_action_space else 4
        self.stack_frames = cfg.stack_frames
        self.stack_frames_buffer = deque(maxlen=self.stack_frames)

    def reset(self):
        state,_ = self.env.reset()
        for _ in range(self.stack_frames):
            if self.cfg.obs_type == "grayscale":
                state_to_buffer = state[:,:,np.newaxis]
            else:
                state_to_buffer = state
            self.stack_frames_buffer.append(state_to_buffer)
        return self._process_state(self.stack_frames_buffer)

    def step(self, action):
        state, reward, done, truncated, info = self.env.step(action)
        if self.cfg.obs_type == "grayscale":
            state_to_buffer = state[:,:,np.newaxis]
        else:
            state_to_buffer = state
        self.stack_frames_buffer.append(state_to_buffer)
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