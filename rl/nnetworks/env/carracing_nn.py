import os
import torch
import torch.nn as nn
import torch.optim as optim
from rl.nnetworks.base import BaseActorCritic

class ImgEncoderCarRacing(nn.Module):
    def __init__(self):
        super().__init__()

        self.img_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
    
    def forward(self, x):
        out = self.img_encoder(x)
        return out

class CarRacingActorCritic(BaseActorCritic):
    def __init__(self):
        super().__init__()

        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1)
                nn.init.constant_(m.bias, 0)

        self.actor = nn.Sequential(
            ImgEncoderCarRacing(),
            nn.Linear(4096, 512),
            nn.ReLU(),
            nn.Linear(512, 5),
            nn.Softmax(dim=-1)
        )

        self.critic = nn.Sequential(
            ImgEncoderCarRacing(),
            nn.Linear(4096, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

        self.optimizer = optim.Adam([
            {'params': self.actor.parameters(), 'lr': 2.5e-4},
            {'params': self.critic.parameters(), 'lr': 1e-3}
        ])

        self.actor.apply(init_weights)
        self.critic.apply(init_weights)

    def _create_img_encoder(self):
        return nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )

    def act(self, state, with_value_state=False):
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            state_tensor = state_tensor.permute(0, 3, 1, 2)
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
        state_tensor = state.permute(0, 3, 1, 2)
        return self.actor(state_tensor)

    def critic_forward(self, state):
        state_tensor = state.permute(0, 3, 1, 2)
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
    
    def save_model(self, path):
        torch.save(self.actor.state_dict(), os.path.join(path, "actor.pth"))
        torch.save(self.critic.state_dict(), os.path.join(path, "critic.pth"))
    
    def load_model(self, path):
        self.actor.load_state_dict(torch.load(os.path.join(path, "actor.pth")))
        self.critic.load_state_dict(torch.load(os.path.join(path, "critic.pth")))

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actor_critic = CarRacingActorCritic()
    actor_critic.set_device(device)
    state = torch.randn(3, 96, 96)
    action, logprob, value = actor_critic.act(state, with_value_state=True)
