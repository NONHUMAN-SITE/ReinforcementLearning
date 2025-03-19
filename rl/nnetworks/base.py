from abc import ABC, abstractmethod

class BaseNN(ABC):

    def __init__(self):
        pass
    
    @abstractmethod
    def act(self, state):
        pass
    
    @abstractmethod
    def eval(self, state):
        pass
    
    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def update(self, loss):
        pass

    @abstractmethod
    def set_device(self, device):
        pass

    @abstractmethod
    def save_model(self, path):
        pass

    @abstractmethod
    def load_model(self, path):
        pass



class BaseActorCritic(BaseNN):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def actor_forward(self, state):
        pass

    @abstractmethod
    def critic_forward(self, state):
        pass
