import os
import torch
import torch.nn as nn
from rl.nnetworks.base import BaseActorCritic
from rl.config.environment import CartPoleEnvConfig


class CartPoleActorCritic(BaseActorCritic):

    def __init__(self, env_cfg: CartPoleEnvConfig = None):
        super().__init__()
        # Inicializar pesos con orthogonal initialization
        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1)
                nn.init.constant_(m.bias, 0)

        # Arquitectura más simple pero efectiva
        self.actor = nn.Sequential(
            nn.Linear(4, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 2),
            nn.Softmax(dim=-1)
        )
        
        self.critic = nn.Sequential(
            nn.Linear(4, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

        # Aplicar inicialización
        self.actor.apply(init_weights)
        self.critic.apply(init_weights)

        # Learning rates diferentes para actor y critic
        self.optimizer = torch.optim.Adam([
            {'params': self.actor.parameters(), 'lr': 2.5e-4},
            {'params': self.critic.parameters(), 'lr': 1e-3}
        ])
    
    def act(self, state, with_value_state=False):
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            action_probs = self.actor(state_tensor)
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
            logprob = dist.log_prob(action)
            if with_value_state:
                value = self.critic(state_tensor)
                return action.item(), logprob, value
            else:
                return action.item(), logprob

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