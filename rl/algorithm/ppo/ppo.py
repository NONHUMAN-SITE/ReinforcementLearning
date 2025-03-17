import torch
import torch.nn as nn
import torch.multiprocessing as mp
import wandb
from tqdm import tqdm
from typing import List
from rl.config.schemas import TrainConfig, PPOConfig,TrainParameters
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
        self.reward_timestep = 0
        self.reward_validation = 0
        self.reward_episode = 0
        self.loss_timestep = 0


class PPOAlgorithm:


    def __init__(self,
                 cfg: PPOConfig):
        
        self.cfg_algorithm = cfg
        self.mse_loss = nn.MSELoss()
        self.global_step = 0

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

    def train(self):
        
        logger.success(f"Training: {self.cfg_algorithm.name} in {self.env.name}")

        state = self.env.reset()        

        for step in tqdm(range(self.cfg_train.total_timesteps),
                         total=self.cfg_train.total_timesteps):
            
            self.buffer.clear()

            self.parallel_recollect()

            for epoch in tqdm(range(self.cfg_algorithm.K_epochs),
                              desc=f"Epochs: {self.cfg_algorithm.K_epochs}",
                              total=self.cfg_algorithm.K_epochs):

                sample = self.buffer.sample()

                loss = self.get_loss(sample)

                self.model.update(loss)


    def parallel_recollect(self):

        '''
        Debemos de optimizar esto lo más posible. Las ideas de optimización son:

        * Acumular estados y calcular A_t en forma tensorial.
        * Recolectar de forma paralela usando N_actors.
        '''

        for _ in range(self.cfg_algorithm.N_actors):

            state = self.env.reset()

            for t in range(self.cfg_algorithm.T_steps):

                action, logprob = self.model.act(state)

                next_state, reward, done, info = self.env.step(action)

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
    
    def get_loss(self, sample): 

        # Esto ya esta en forma tensorial.
        old_states, old_logprobs, old_A_t, old_V_targ = sample

        logprobs = self.model.actor_forward(old_states)

        ratio = torch.exp(logprobs - old_logprobs.detach())

        surr_obj_1 = ratio * old_A_t

        surr_obj_2 = torch.clamp(ratio, 1 - self.cfg_algorithm.eps_clip, 1 + self.cfg_algorithm.eps_clip) * old_A_t

        #Esta multiplicación de ratio por A_t tiene que ser pointwise.
        surr_obj = torch.min(surr_obj_1, surr_obj_2)

        entropy = -torch.sum(logprobs * torch.exp(logprobs), dim=1)

        vf_loss = self.mse_loss(self.model.critic_forward(old_states), old_V_targ)

        loss_total = surr_obj - self.cfg_algorithm.vf_coef * vf_loss + self.cfg_algorithm.entropy_coef * entropy

        #Negative loss because we want to maximize the loss.
        loss = -loss_total.mean()

        return loss
        
    def collate_fn(self, transitions: List[tuple]):

        # Transpose the list of tuples to a list of tensors.
        transitions = list(zip(*transitions))

        print(transitions[0])
        print(transitions[1])
        print(transitions[2])
        print(transitions[3])

        old_states = torch.tensor(transitions[0], dtype=torch.float32)
        old_logprobs = torch.tensor(transitions[1], dtype=torch.float32)
        old_A_t = torch.tensor(transitions[2], dtype=torch.float32)
        old_V_targ = torch.tensor(transitions[3], dtype=torch.float32)

        return old_states, old_logprobs, old_A_t, old_V_targ


