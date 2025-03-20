import random
from typing import Callable
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

    def sample(self):
        # Si no hay índices o ya se usaron todos, crear nueva permutación
        if not self.indices:
            self.indices = list(range(len(self.buffer)))
            random.shuffle(self.indices)
            self.current_idx = 0
        
        # Tomar el siguiente batch de índices
        batch_indices = self.indices[self.current_idx:self.current_idx + self.batch_size]
        self.current_idx += self.batch_size
        
        # Si no hay suficientes elementos para un batch completo
        if len(batch_indices) < self.batch_size:
            # Opción 1: Completar con elementos del inicio
            remaining = self.batch_size - len(batch_indices)
            batch_indices.extend(self.indices[:remaining])
            self.current_idx = remaining
            
        # Obtener las transiciones correspondientes a los índices
        transitions_sample = [self.buffer[i] for i in batch_indices]
        
        return self.collate_fn(transitions_sample)

    def clear(self):
        self.buffer = []
    
    def __len__(self):
        return len(self.buffer)
    