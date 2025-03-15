import torch.nn as nn

class CartPoleActorCritic:
    def __init__(self):
        self.actor = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )
        self.critic = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def actor_forward(self, state):
        return self.actor(state)
    
    def critic_forward(self, state):
        return self.critic(state)
    
    