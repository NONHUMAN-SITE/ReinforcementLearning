import torch
from os.path import join
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

    @abstractmethod
    def save_best_model(self, path):
        pass

    @abstractmethod
    def load_best_model(self, path):
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

    def save_model(self, path):
        torch.save(self.actor.state_dict(), join(path, "actor.pth"))
        torch.save(self.critic.state_dict(), join(path, "critic.pth"))

    def load_model(self, path):
        self.actor.load_state_dict(torch.load(join(path, "actor.pth")))
        self.critic.load_state_dict(torch.load(join(path, "critic.pth")))

    def save_best_model(self, path):
        torch.save(self.actor.state_dict(), join(path, "best_actor.pth"))
        torch.save(self.critic.state_dict(), join(path, "best_critic.pth"))

    def load_best_model(self, path):
        self.actor.load_state_dict(torch.load(join(path, "best_actor.pth")))
        self.critic.load_state_dict(torch.load(join(path, "best_critic.pth")))