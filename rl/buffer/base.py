from abc import ABC, abstractmethod

class BaseBuffer(ABC):

    '''
    The order of the transitions is:
    (state, action, reward, next_state, done)
    '''

    @abstractmethod
    def sample(self, batch_size: int):
        pass

    @abstractmethod
    def add(self, transition: tuple):
        pass
    
    @abstractmethod
    def clear(self):
        pass
    
    @abstractmethod
    def __len__(self):
        pass

    def set_collate_fn(self, collate_fn):
        self.collate_fn = collate_fn

    def set_device(self, device):
        self.device = device
