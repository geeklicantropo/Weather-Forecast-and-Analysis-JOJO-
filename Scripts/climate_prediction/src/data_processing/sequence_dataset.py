import numpy as np
import torch
from torch.utils.data import IterableDataset
import os
from typing import List, Tuple, Optional
import logging
from pathlib import Path
from tqdm import tqdm

class SequenceDataset(IterableDataset):
    def __init__(self, 
                 data_dir: str,
                 sequence_length: int,
                 batch_size: int = 32,
                 shuffle: bool = True,
                 device: Optional[torch.device] = None):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = device or torch.device('cpu')
        
        # Get list of all .npz files
        self.files = list(self.data_dir.glob('*.npz'))
        if not self.files:
            raise RuntimeError(f"No .npz files found in {data_dir}")
            
        # Load metadata from first file to get dimensions
        sample = np.load(self.files[0])
        self.n_features = sample['X'].shape[-1]
        
        # Calculate total size
        self.total_sequences = self._count_total_sequences()
        self._setup_memmap()
        
    def _count_total_sequences(self) -> int:
        """Count total sequences across all files."""
        total = 0
        for f in self.files:
            data = np.load(f)
            total += len(data['X'])
        return total
        
    def memory_usage(self, deep: bool = True) -> int:
        """Get memory usage of dataset in bytes."""
        return (self.total_sequences * self.sequence_length * 
                self.n_features * np.dtype('float32').itemsize)
    
    def _setup_memmap(self):
        """Setup memory-mapped arrays for sequences with progress bar."""
        memmap_path = self.data_dir / 'sequences.mmap'
        if not memmap_path.exists():
            # Create memory-mapped array
            self.sequences = np.memmap(memmap_path,
                                     dtype='float32',
                                     mode='w+',
                                     shape=(self.total_sequences,
                                           self.sequence_length,
                                           self.n_features))
            
            # Fill memory map with progress bar
            current_idx = 0
            with tqdm(total=len(self.files), desc="Loading sequences") as pbar:
                for f in self.files:
                    data = np.load(f)
                    batch_size = len(data['X'])
                    self.sequences[current_idx:current_idx + batch_size] = data['X']
                    current_idx += batch_size
                    pbar.update(1)
            
            self.sequences.flush()
        else:
            # Load existing memory map
            self.sequences = np.memmap(memmap_path,
                                     dtype='float32',
                                     mode='r',
                                     shape=(self.total_sequences,
                                           self.sequence_length,
                                           self.n_features))
    
    def __iter__(self):
        # Generate indices
        indices = np.arange(len(self.sequences))
        if self.shuffle:
            np.random.shuffle(indices)
        
        # Yield batches
        for start_idx in range(0, len(indices), self.batch_size):
            batch_indices = indices[start_idx:start_idx + self.batch_size]
            
            # Load batch from memory map
            X_batch = torch.FloatTensor(
                self.sequences[batch_indices].copy()
            ).to(self.device)
            
            # For LSTM input, we need input and target
            # Last timestep is target, rest are input
            yield (X_batch[:, :-1, :], X_batch[:, -1, :])
    
    def __len__(self):
        return len(self.sequences) // self.batch_size