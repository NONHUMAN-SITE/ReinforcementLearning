from os.path import join
import torch
import torch.nn as nn
from rl.nnetworks.base import BaseActorCritic
from rl.config.environment import BreakoutEnvConfig

class BreakoutImgEncoder(nn.Module):
    def __init__(self,input_channels: int = 4):
        super().__init__()

        self.img_encoder = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Flatten()
        )
    def forward(self, x):
        '''
        Ensure the input is in the correct format and then compute the flattened output
        '''
        out = self.img_encoder(x)
        return out

class BreakoutActorCritic(BaseActorCritic):
    def __init__(self, env_cfg: BreakoutEnvConfig = None):

        self.env_cfg = env_cfg
        if self.env_cfg.full_action_space:
            self.action_space = 18
        else:
            self.action_space = 4

        if self.env_cfg.obs_type == "grayscale":
            self.input_channels = self.env_cfg.stack_frames
        else:
            self.input_channels = self.env_cfg.stack_frames * 3
        

        super().__init__()
        
        self.actor = nn.Sequential(
            BreakoutImgEncoder(self.input_channels),
            nn.Linear(1408, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, self.action_space),
            nn.Softmax(dim=-1)
        )   

        self.critic = nn.Sequential(
            BreakoutImgEncoder(self.input_channels),
            nn.Linear(1408, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

        self.optimizer = torch.optim.Adam(
            [
                {'params': self.actor.parameters(), 'lr': 3e-4},
                {'params': self.critic.parameters(), 'lr': 3e-4}
            ]
        )
    
    def act(self, state, with_value_state=False):
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            state_tensor = state_tensor.permute(0, 3, 1, 2)/255.0 # normalize to [0, 1]
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
        state_tensor = state.permute(0, 3, 1, 2)/255.0 # normalize to [0, 1]
        return self.actor(state_tensor)

    def critic_forward(self, state):
        state_tensor = state.permute(0, 3, 1, 2)/255.0 # normalize to [0, 1]
        return self.critic(state_tensor)
    
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