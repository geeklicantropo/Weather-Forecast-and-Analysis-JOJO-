import torch
import os
from typing import Optional
import torch
from contextlib import contextmanager

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
            
         # Get total and available memory on the GPU
        device_properties = torch.cuda.get_device_properties(0)
        total_memory = device_properties.total_memory / 1024**3  # Total memory in GB
        reserved_memory = torch.cuda.memory_reserved(0) / 1024**3  # Reserved memory in GB
        allocated_memory = torch.cuda.memory_allocated(0) / 1024**3  # Allocated memory in GB
        available_memory = total_memory - reserved_memory - allocated_memory  # Free memory in GB

        # Scale batch size based on available memory
        if total_memory >= 23:
            if available_memory >= 20:
                return base_batch_size * 8
            elif available_memory >= 16:
                return base_batch_size * 6
            elif available_memory >= 12:
                return base_batch_size * 4
            elif available_memory >= 8:
                return base_batch_size * 2
        elif total_memory >= 16:
            if available_memory >= 12:
                return base_batch_size * 4
            elif available_memory >= 8:
                return base_batch_size * 2

        return base_batch_size
        
    @contextmanager
    def memory_monitor(self):
        """Context manager to monitor GPU memory usage."""
        if not self.gpu_available:
            yield
            return
        
        try:
            start_memory = self.get_memory_info()
            yield
            end_memory = self.get_memory_info()
            
            memory_diff = {
                key: end_memory[key] - start_memory[key]
                for key in start_memory
            }
            
            print(f"GPU Memory Usage:")
            print(f"  Allocated: {memory_diff['allocated']:.2f} MB")
            print(f"  Cached: {memory_diff['cached']:.2f} MB")
            print(f"  Free: {memory_diff['free']:.2f} MB")
            
        except Exception as e:
            print(f"Error monitoring GPU memory: {str(e)}")
            raise

gpu_manager = GPUManager()  # Singleton instance