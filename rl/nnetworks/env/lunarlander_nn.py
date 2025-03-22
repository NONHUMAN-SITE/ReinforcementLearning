import os
import torch
import torch.nn as nn
import torch.optim as optim
from rl.nnetworks.base import BaseActorCritic
from rl.config.environment import LunarLanderEnvConfig
from rl.utils.logger import logger

class LunarLanderActorCritic(BaseActorCritic):

    def __init__(self, env_cfg: LunarLanderEnvConfig = None):
        super().__init__()
        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1)
                nn.init.constant_(m.bias, 0)

        self.actor = nn.Sequential(
            nn.Linear(8, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 4),
            nn.Softmax(dim=-1)
        )

        self.critic = nn.Sequential(
            nn.Linear(8, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        self.optimizer = optim.Adam([
            {'params': self.actor.parameters(), 'lr': 2.5e-4},
            {'params': self.critic.parameters(), 'lr': 2.5e-4}
        ])

        self.actor.apply(init_weights)
        self.critic.apply(init_weights)

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
        #actor_grad_norm = self._get_gradient_norm(self.actor)
        #critic_grad_norm = self._get_gradient_norm(self.critic)
        #logger.info(f"Actor grad norm: {actor_grad_norm}, Critic grad norm: {critic_grad_norm}")
        self.optimizer.step()
    
    def set_device(self, device):
        self.device = device
        self.actor.to(device)
        self.critic.to(device)
    
    def _get_gradient_norm(self, model):
        grad_norm = 0
        for param in model.parameters():
            if param.grad is not None:
                param_norm = param.grad.norm(2).item()
                grad_norm += param_norm**2
        grad_norm = grad_norm**0.5
        return grad_norm

