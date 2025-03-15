import random
from rl.buffer.base import BaseBuffer

class BasicBuffer(BaseBuffer):
    def __init__(self, size: int, batch_size: int):
        self.size = size
        self.batch_size = batch_size
        self.buffer = []

    def add(self, transition: tuple):
        self.buffer.append(transition)

    def sample(self, batch_size: int):
        return random.sample(self.buffer, batch_size)
    
    def clear(self):
        self.buffer = []
    
    def __len__(self):
        return len(self.buffer)