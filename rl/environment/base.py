from abc import ABC, abstractmethod

class BaseEnv(ABC):
    def __init__(self):
        pass

    '''
    Falta implementar este método en cada entorno concreto.
    '''
    @staticmethod
    def get_env_creator():
        def create_env():
            return BaseEnv()  # O la implementación específica
        return create_env
    
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
    