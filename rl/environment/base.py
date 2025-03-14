class BaseEnv:
    def __init__(self):
        pass

    def reset(self):
        '''
        Returns:
            obs, info
        '''
        pass

    def step(self, action):
        '''
        Returns:
            obs, reward, terminated, truncated, info
        '''
        pass
    
    def close(self):
        pass
    