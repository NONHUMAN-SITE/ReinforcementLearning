import torch
import torch.nn as nn
from rl.nnetworks.base import BaseActorCritic

class CartPoleActorCritic(BaseActorCritic):

    def __init__(self):
        self.actor = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Softmax(dim=1)
        )
        self.critic = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        self.optimizer = torch.optim.Adam([
            {'params': self.actor.parameters(), 'lr': 1e-4},
            {'params': self.critic.parameters(), 'lr': 1e-4}
        ])
    
    def act(self, state):
        action = self.actor(state)
        return action.argmax(dim=1).item(), action

    def actor_forward(self, state):
        return self.actor(state)

    def critic_forward(self, state):
        return self.critic(state)
    
    def update(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def set_device(self, device):
        self.actor.to(device)
        self.critic.to(device)
    
    def save_model(self, path):
        torch.save(self.actor.state_dict(), path + "_actor.pth")
        torch.save(self.critic.state_dict(), path + "_critic.pth")
    
    def load_model(self, path):
        self.actor.load_state_dict(torch.load(path + "_actor.pth"))
        self.critic.load_state_dict(torch.load(path + "_critic.pth"))