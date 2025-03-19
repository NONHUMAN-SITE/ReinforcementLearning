import os
import torch
import torch.nn as nn
from rl.nnetworks.base import BaseActorCritic

class BipedalWalkerActorCritic(BaseActorCritic):
    def __init__(self):
        super().__init__()
        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1)
                nn.init.constant_(m.bias, 0)

        self.action_space = 4

        self.actor = nn.Sequential(
            nn.Linear(24, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, self.action_space),
            nn.Tanh()
        )
        
        self.critic = nn.Sequential(
            nn.Linear(24, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 1)
        )

        # Aplicar inicialización
        self.actor.apply(init_weights)
        self.critic.apply(init_weights)

        # Learning rates diferentes para actor y critic
        self.optimizer = torch.optim.Adam([
            {'params': self.actor.parameters(), 'lr': 2.5e-4},
            {'params': self.critic.parameters(), 'lr': 2.5e-4}
        ])

        self.action_var = None
    
    def act(self, state, with_value_state=False):
        if self.action_var is None:
            raise ValueError("Action var is not set")
        
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            action_mean = self.actor(state_tensor)
            
            cov_matrix = torch.diag(self.action_var).unsqueeze(0)
            dist = torch.distributions.MultivariateNormal(action_mean, cov_matrix)
            action = dist.sample()
            logprob = dist.log_prob(action)
            if with_value_state:
                value = self.critic(state_tensor)
                return action.squeeze(0).cpu().numpy(), logprob, value
            else:
                return action.squeeze(0).cpu().numpy(), logprob
    
    def actor_forward(self, state):
        return self.actor(state)
    
    def critic_forward(self, state):
        return self.critic(state)

    def train(self):
        self.actor.train()
        self.critic.train()

    def eval(self):
        self.actor.eval()
        self.critic.eval()
    
    def update(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def set_device(self, device):
        self.device = device
        self.actor.to(device)
        self.critic.to(device)
    
    def save_model(self, path):
        torch.save(self.actor.state_dict(), os.path.join(path, "actor.pth"))
        torch.save(self.critic.state_dict(), os.path.join(path, "critic.pth"))
    
    def load_model(self, path):
        self.actor.load_state_dict(torch.load(os.path.join(path, "actor.pth")))
        self.critic.load_state_dict(torch.load(os.path.join(path, "critic.pth")))
    
    def set_std(self, std):
        self.action_var = torch.full((self.action_space,), std ** 2)
        self.action_var = self.action_var.to(self.device)