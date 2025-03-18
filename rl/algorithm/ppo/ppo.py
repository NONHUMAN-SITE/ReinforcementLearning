import torch
import torch.nn as nn
import wandb
from tqdm import tqdm
from typing import List
from rl.config.schemas import PPOConfig,TrainParameters
from rl.buffer.basic_buffer import BasicBuffer
from rl.environment.base import BaseEnv
from rl.nnetworks.base import BaseActorCritic
from rl.utils.logger import logger

class MetricsPPO:
    '''
    Métricas:
    - Reward por timestep
    - Reward por validacion
    - Reward por episodio
    - Loss por timestep
    '''
    def __init__(self):
        self.reward_mean_timestep = 0
        self.reward_validation_timestep = 0
        self.reward_episode_timestep = 0
        self.loss_timestep = 0


class PPOAlgorithm:

    def __init__(self,
                 cfg: PPOConfig):
        
        self.cfg_algorithm = cfg
        self.mse_loss = nn.MSELoss()
        self.global_step = 0
        self.metrics = MetricsPPO()

    def set_algorithm_params(self,
                             model: BaseActorCritic,
                             env: BaseEnv,
                             buffer: BasicBuffer,
                             cfg_train: TrainParameters):
        
        self.model     = model
        self.env       = env
        self.buffer    = buffer
        self.cfg_train = cfg_train

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"Using device: {self.device}")

        self.model.set_device(self.device)
        self.buffer.set_device(self.device)
        self.buffer.set_collate_fn(self.collate_fn)
    
    def train(self):
        
        logger.success(f"Training: {self.cfg_algorithm.name} in {self.env.name}")

        state = self.env.reset()        

        for step in tqdm(range(1,self.cfg_train.total_timesteps + 1),
                         total=self.cfg_train.total_timesteps):
            
            self.buffer.clear()

            self.parallel_recollect()

            for epoch in range(self.cfg_algorithm.K_epochs):

                sample = self.buffer.sample()

                loss = self.get_loss(sample)

                self.model.update(loss)
            
            if step % self.cfg_train.validate_frequency == 0:
                self.validate()


    def validate(self):

        logger.info("Validating...")

        self.model.eval()

        state = self.env.reset()

        episodes_reward = []

        for episode in range(self.cfg_train.validate_episodes):

            total_reward = 0

            while True:

                action, prob = self.model.act(state)

                next_state, reward, done, truncated, info = self.env.step(action)

                next_state = next_state.to(self.device)

                done = done or truncated

                total_reward += reward

                state = next_state

                if done or truncated:
                    break

            episodes_reward.append(total_reward)

        logger.success(f"Validation: {sum(episodes_reward) / len(episodes_reward)}")


    def parallel_recollect(self):

        '''
        Debemos de optimizar esto lo más posible. Las ideas de optimización son:

        * Acumular estados y calcular A_t en forma tensorial.
        * Recolectar de forma paralela usando N_actors.
        '''

        for _ in range(self.cfg_algorithm.N_actors):

            state = self.env.reset()
            state = state.to(self.device)

            for t in range(self.cfg_algorithm.T_steps):

                action, prob = self.model.act(state)

                logprob = torch.log(prob[:,action])

                next_state, reward, done, truncated, info = self.env.step(action)

                next_state = next_state.to(self.device)

                done = done or truncated

                with torch.no_grad():

                    next_value_state = self.model.critic_forward(next_state)
                
                    value_state = self.model.critic_forward(state)
                    
                    delta_t = reward + self.cfg_algorithm.gamma * (1 - done) * next_value_state - value_state
                
                A_t = 0

                for l in range(self.cfg_algorithm.T_steps - t + 1):

                    A_t += (self.cfg_algorithm.gamma ** l) * delta_t
                
                V_targ = A_t + value_state

                self.buffer.add((state, logprob, A_t, V_targ))

                state = next_state

                if done:
                    state = self.env.reset()
                    state = state.to(self.device)

    def get_loss(self, sample): 

        # Esto ya esta en forma tensorial.
        old_states, old_logprobs, old_A_t, old_V_targ = sample

        '''
        N: numero de acciones.
        B: batch size.

        Logprobs es un tensor de shape (B,N)
        '''

        probs = self.model.actor_forward(old_states)  # shape (B,N)
        log_probs = torch.log(probs)  # shape (B,N)

        action = torch.argmax(probs, dim=1)  # shape (B,)
        log_probs_selected = torch.gather(log_probs, 1, action.unsqueeze(1))  # shape (B,1)

        # Entropy: -sum(p_i * log(p_i))
        # Multiply by probs to make it pointwise.
        entropy = -torch.sum(probs * log_probs, dim=1)  # shape (B,)

        ratio = torch.exp(log_probs_selected - old_logprobs.detach())  # shape (B,1)

        surr_obj_1 = ratio * old_A_t

        surr_obj_2 = torch.clamp(ratio, 1 - self.cfg_algorithm.eps_clip, 1 + self.cfg_algorithm.eps_clip) * old_A_t

        surr_obj = torch.min(surr_obj_1, surr_obj_2)

        #VF LOSS
        vf_loss = self.mse_loss(self.model.critic_forward(old_states), old_V_targ)

        loss_total = surr_obj - self.cfg_algorithm.vf_coef * vf_loss + self.cfg_algorithm.entropy_coef * entropy

        #Negative loss because we want to maximize the loss.
        loss = -loss_total.mean()

        return loss
        
    def collate_fn(self, transitions: List[tuple]):

        transitions  = list(zip(*transitions))

        old_states   = torch.stack(transitions[0], dim=0).squeeze(1)
        old_logprobs = torch.stack(transitions[1], dim=0).squeeze(1)
        old_A_t      = torch.stack(transitions[2], dim=0).squeeze(1)
        old_V_targ   = torch.stack(transitions[3], dim=0).squeeze(1)

        return old_states, old_logprobs, old_A_t, old_V_targ


