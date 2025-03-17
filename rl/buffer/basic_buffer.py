import random
from typing import List,Callable
from rl.buffer.base import BaseBuffer

class BasicBuffer(BaseBuffer):
    def __init__(self, size: int, batch_size: int):
        self.size = size
        self.batch_size = batch_size
        self.buffer = []

    def set_collate_fn(self, collate_fn: Callable):
        self.collate_fn = collate_fn

    def add(self, transition: tuple):
        self.buffer.append(transition)

    def sample(self, batch_size: int):

        transitions_sample = random.sample(self.buffer, batch_size)

        return self.collate_fn(transitions_sample)

    def clear(self):
        self.buffer = []
    
    def __len__(self):
        return len(self.buffer)
    