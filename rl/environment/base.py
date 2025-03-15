from abc import ABC, abstractmethod

class BaseEnv(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def reset(self):
        '''
        Returns:
            obs, info
        '''
        pass

    @abstractmethod
    def step(self, action):
        '''
        Returns:
            next_obs, reward, terminated, truncated, info
        '''
        pass
    