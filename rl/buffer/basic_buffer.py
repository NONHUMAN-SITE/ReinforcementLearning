import random
from typing import Callable, Iterator
from rl.buffer.base import BaseBuffer

class BasicBuffer(BaseBuffer):
    def __init__(self, size: int, batch_size: int):
        self.size = size
        self.batch_size = batch_size
        self.buffer = []
        self.collate_fn = None
        self.device = None
        self.current_idx = 0  # Índice para tracking
        self.indices = []     # Lista de índices para shuffle

    def set_collate_fn(self, collate_fn: Callable):
        self.collate_fn = collate_fn

    def add(self, transition: tuple):
        if len(self.buffer) >= self.size:
            self.buffer.pop(0)
        self.buffer.append(transition)

    def set_device(self, device):
        self.device = device

    def __iter__(self) -> Iterator:
        self.indices = list(range(len(self.buffer)))
        random.shuffle(self.indices)
        self.current_idx = 0
        return self

    def __next__(self):
        if self.current_idx >= len(self.buffer):
            raise StopIteration

        # Tomar el siguiente batch de índices
        batch_indices = self.indices[self.current_idx:self.current_idx + self.batch_size]
        self.current_idx += self.batch_size

        # Si el último batch es más pequeño que batch_size, completar con elementos del inicio
        if len(batch_indices) < self.batch_size:
            remaining = self.batch_size - len(batch_indices)
            batch_indices.extend(self.indices[:remaining])
            
        # Obtener las transiciones correspondientes a los índices
        transitions_sample = [self.buffer[i] for i in batch_indices]
        
        return self.collate_fn(transitions_sample)

    def shuffle(self):
        """Mezcla los índices del buffer"""
        self.indices = list(range(len(self.buffer)))
        random.shuffle(self.indices)
        self.current_idx = 0

    def clear(self):
        self.buffer = []
    
    def __len__(self):
        return len(self.buffer)
    