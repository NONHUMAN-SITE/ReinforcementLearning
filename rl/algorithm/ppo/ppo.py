import os
import torch
import torch.nn as nn
import wandb
from tqdm import tqdm
from typing import List
from rl.config.algorithms import PPOConfig
from rl.config.schemas import TrainParameters
from rl.buffer.basic_buffer import BasicBuffer
from rl.environment.base import BaseEnv
from rl.nnetworks.base import BaseActorCritic
from rl.utils.logger import logger
from omegaconf import OmegaConf

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
        self.mse_loss = nn.MSELoss(reduction="none")
        self.global_step = 0
        self.metrics = MetricsPPO()
        self.gae_lambda = cfg.gae_lambda
        self.best_reward = -float("inf")
        
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
        #wandb.init(project="PPO", name=f"{self.env.name}-{self.cfg_algorithm.name}")
        path = os.path.join(self.cfg_train.save_path, f"ppo-{self.env.name}") 
        os.makedirs(path, exist_ok=True)

        if self.env.is_continuous:
            self.model.set_std(self.env.init_std)
            logger.info(f"Continuous action space: Initializing std: {self.env.init_std}")
        else:
            logger.info(f"Discrete action space")
        
        for step in tqdm(range(1, self.cfg_train.total_timesteps + 1),
                         total=self.cfg_train.total_timesteps,
                         desc="Training"):
            
            self.buffer.clear()
            self.parallel_recollect()

            for epoch in range(self.cfg_algorithm.K_epochs):
                for sample in self.buffer:
                    loss = self.get_loss(sample)
                    self.model.update(loss)
            
            if step % self.cfg_train.validate_frequency == 0:
                average_reward =self.validate()
                if average_reward > self.best_reward:
                    self.best_reward = average_reward
                    self.model.save_best_model(path)
            
            if step % self.cfg_train.save_frequency == 0:
                
                # Save model
                self.model.save_model(path)
                self.save_config(path)
            
            if self.env.is_continuous:
                #Linear decay of std
                std = max(self.env.init_std * (1 - step / self.cfg_train.total_timesteps),
                          self.env.min_std)
                if std < 0:
                    raise ValueError("Std is less than 0")
                self.model.set_std(std)
                logger.info(f"Continuous action space: Updating std: {std}")

    def validate(self):

        logger.info("Validating...")

        self.model.eval()

        episodes_reward = []

        for episode in range(self.cfg_train.validate_episodes):

            state = self.env.reset()
            
            total_reward = 0

            while True:

                action, logprob = self.model.act(state)

                self.env.render()

                next_state, reward, done, truncated, info = self.env.step(action)

                done = done or truncated

                total_reward += reward

                state = next_state

                if done or truncated:
                    break

            episodes_reward.append(total_reward)

        average_reward = sum(episodes_reward) / len(episodes_reward)

        logger.success(f"Validation: {average_reward}")

        wandb.log({"reward_validation": average_reward},step=self.global_step)

        self.global_step += 1

        self.model.train()

        return average_reward

    def parallel_recollect(self):
        '''
        Implementación de GAE (Generalized Advantage Estimation)
        '''
        for _ in range(self.cfg_algorithm.N_actors):

            state = self.env.reset()

            states   = []
            logprobs = []
            rewards  = []
            dones    = []
            values   = []
            actions  = []
            
            for t in range(self.cfg_algorithm.T_steps):

                action, logprob, value = self.model.act(state, with_value_state=True)
                
                next_state, reward, done, truncated, info = self.env.step(action)

                done = done or truncated
                
                states.append(state)
                logprobs.append(logprob)
                rewards.append(reward)
                dones.append(done)
                values.append(value)
                actions.append(action)
                state = next_state
                
                if done:
                    state = self.env.reset()
                        
            # Calculating GAE
            advantages = []
            last_advantage = 0
            last_value = values[-1]
            
            for t in reversed(range(len(rewards))):
                if dones[t]:
                    delta = rewards[t] - values[t]
                    last_advantage = delta
                else:
                    delta = rewards[t] + self.cfg_algorithm.gamma * last_value - values[t]
                    last_advantage = delta + self.cfg_algorithm.gamma * self.gae_lambda * last_advantage
                
                advantages.insert(0, last_advantage)
                last_value = values[t]
            
            advantages = torch.tensor(advantages, device=self.device)
            values = torch.tensor(values, device=self.device)

            V_targ = advantages + values
            
            for t in range(self.cfg_algorithm.T_steps):
                self.buffer.add((states[t],actions[t], logprobs[t], advantages[t], V_targ[t]))

    def get_loss(self, sample):

        old_states, old_actions, old_logprobs, old_A_t, old_V_targ = sample

        #CLIP LOSS
        if self.env.is_continuous:
            action_mean = self.model.actor_forward(old_states)
            cov_matrix = torch.diag(self.model.action_var).unsqueeze(0)
            dist = torch.distributions.MultivariateNormal(action_mean, cov_matrix)
            new_logprobs = dist.log_prob(old_actions)
        else:
            probs = self.model.actor_forward(old_states)
            dist = torch.distributions.Categorical(probs)
            new_logprobs = dist.log_prob(old_actions)

        ratio = torch.exp(new_logprobs - old_logprobs.detach())
        clip_ratio = torch.clamp(ratio, 1 - self.cfg_algorithm.eps_clip, 1 + self.cfg_algorithm.eps_clip)

        clip_loss = torch.min(ratio * old_A_t, clip_ratio * old_A_t)


        #ENTROPY LOSS
        entropy_loss = dist.entropy()

        #VALUE LOSS
        value_states = self.model.critic_forward(old_states).squeeze()
        value_loss = self.mse_loss(old_V_targ, value_states)
        value_loss = value_loss

        #TOTAL LOSS
        loss = clip_loss - self.cfg_algorithm.vf_coef * value_loss + self.cfg_algorithm.entropy_coef * entropy_loss
        loss = -loss.mean()
        
        return loss

    def collate_fn(self, transitions: List[tuple]):

        transitions  = list(zip(*transitions))

        old_states   = torch.stack([torch.tensor(transition, dtype=torch.float32) for transition in transitions[0]], dim=0).squeeze()
        old_actions  = torch.stack([torch.tensor(transition, dtype=torch.float32) for transition in transitions[1]], dim=0).squeeze()
        old_logprobs = torch.stack(transitions[2], dim=0).squeeze()
        old_A_t      = torch.stack(transitions[3], dim=0).squeeze()
        old_V_targ   = torch.stack(transitions[4], dim=0).squeeze()

        return (old_states.to(self.device),
                old_actions.to(self.device),
                old_logprobs.to(self.device),
                old_A_t.to(self.device),
                old_V_targ.to(self.device))

    def save_config(self, path: str):
        '''
        Save the configurations to yaml files
        '''
        train_cfg_obj = OmegaConf.to_object(self.cfg_train)
        algorithm_cfg_obj = OmegaConf.to_object(self.cfg_algorithm)
        env_cfg_obj = OmegaConf.to_object(self.env.cfg)
        
        eval_config = {
            'env': env_cfg_obj,
            'algorithm': algorithm_cfg_obj,
            'model': self.model.__class__.__name__
        }
        
        eval_config_path = os.path.join(path, 'eval_config.yaml')
        OmegaConf.save(config=OmegaConf.create(eval_config), f=eval_config_path)
        logger.info(f"Evaluation config saved to {eval_config_path}")
        
        # Guardar las configuraciones originales
        train_cfg_obj.save_config(path)
        algorithm_cfg_obj.save_config(path)
        env_cfg_obj.save_config(path)
