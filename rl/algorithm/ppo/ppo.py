import torch.optim as optim
from rl.config.schemas import TrainConfig

class PPOAlgorithm:
    def __init__(self,
                 model,
                 env,
                 buffer,    
                 cfg: TrainConfig):

        self.model = model
        self.env = env
        self.buffer = buffer
        self.cfg = cfg


    
    