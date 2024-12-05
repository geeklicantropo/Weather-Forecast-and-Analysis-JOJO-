#src/utils/gpu_manager.pyimport torch
import os
from typing import Optional
import torch

class GPUManager:
    def __init__(self):
        self.gpu_available = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.gpu_available else "cpu")
        
    def get_device(self) -> torch.device:
        return self.device
    
    def get_memory_info(self) -> Optional[dict]:
        if not self.gpu_available:
            return None
        
        memory_allocated = torch.cuda.memory_allocated(0)
        memory_cached = torch.cuda.memory_reserved(0)
        return {
            "allocated": memory_allocated / 1024**2,  # MB
            "cached": memory_cached / 1024**2,
            "free": torch.cuda.get_device_properties(0).total_memory / 1024**2 - memory_allocated / 1024**2
        }
    
    def clear_memory(self):
        if self.gpu_available:
            torch.cuda.empty_cache()
            
    def get_optimal_batch_size(self, base_batch_size: int = 32) -> int:
        if not self.gpu_available:
            return base_batch_size
            
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        if total_memory >= 16:
            return base_batch_size * 4
        elif total_memory >= 8:
            return base_batch_size * 2
        return base_batch_size

gpu_manager = GPUManager()  # Singleton instance