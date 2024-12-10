import numpy as np
import torch
from torch.utils.data import IterableDataset
import os
from typing import List, Tuple, Optional
import logging
from pathlib import Path
from tqdm import tqdm

class SequenceDataset(IterableDataset):
    def __init__(self, temp_dir: str, sequence_length: int, batch_size: int = 32,
                 shuffle: bool = True, device: Optional[torch.device] = None):
        super().__init__()
        self.temp_dir = Path(temp_dir)
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = device or torch.device('cpu')
        
        # Setup directory
        self.chunk_dir = self.temp_dir / 'sequence_chunks'
        self.chunk_dir.mkdir(parents=True, exist_ok=True)
        
        self.sequence_files = sorted(self.chunk_dir.glob('chunk_*.npz'))
        if not self.sequence_files:
            raise RuntimeError(f"No sequence chunks found in {self.chunk_dir}")
        
        self._initialize_stream()

    def _initialize_stream(self):
        """Initialize streaming parameters."""
        self.current_chunk = None
        self.current_chunk_idx = 0
        self.chunks_loaded = 0
        
        # Get total size without loading everything
        self.total_sequences = 0
        for f in self.sequence_files:
            with np.load(f) as data:
                self.total_sequences += len(data['X'])
                if self.current_chunk is None:
                    # Get feature dimension from first file
                    self.n_features = data['X'].shape[-1]
    
    def _load_next_chunk(self):
        """Load next chunk into memory."""
        if self.current_chunk_idx >= len(self.sequence_files):
            if self.shuffle:
                np.random.shuffle(self.sequence_files)
            self.current_chunk_idx = 0
            self.chunks_loaded = 0
            
        chunk_path = self.sequence_files[self.current_chunk_idx]
        with np.load(chunk_path) as data:
            # Keep data on CPU initially
            self.current_chunk = torch.from_numpy(data['X']).float()
        
        self.current_chunk_idx += 1
        self.chunks_loaded += 1
        
    def _count_total_sequences(self) -> int:
        """Count total sequences across all files."""
        total = 0
        for f in self.files:
            data = np.load(f)
            total += len(data['X'])
        return total
    
    def _setup_memmap(self):
        """Setup memory-mapped array for efficient data streaming."""
        memmap_path = self.data_dir / 'sequences.mmap'
        shape = (self.total_sequences, self.sequence_length, self.n_features)
        
        if not memmap_path.exists():
            self.sequences = np.memmap(memmap_path,
                                     dtype='float32',
                                     mode='w+',
                                     shape=shape)
            
            current_idx = 0
            with tqdm(total=len(self.files), desc="Loading sequences") as pbar:
                for f in self.files:
                    data = np.load(f)
                    batch_size = len(data['X'])
                    
                    # Process data in CPU memory first
                    if isinstance(data['X'], torch.Tensor):
                        batch_data = data['X'].cpu().numpy()
                    else:
                        batch_data = data['X']
                    
                    self.sequences[current_idx:current_idx + batch_size] = batch_data
                    current_idx += batch_size
                    pbar.update(1)
            
            self.sequences.flush()
        else:
            self.sequences = np.memmap(memmap_path,
                                     dtype='float32',
                                     mode='r',
                                     shape=shape)
    
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            # Split files among workers
            per_worker = int(np.ceil(len(self.sequence_files) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            self.sequence_files = self.sequence_files[worker_id * per_worker:
                                                    (worker_id + 1) * per_worker]
        
        while True:
            if self.current_chunk is None or self.chunks_loaded % len(self.sequence_files) == 0:
                self._load_next_chunk()
            
            indices = torch.randperm(len(self.current_chunk)) if self.shuffle else torch.arange(len(self.current_chunk))
            
            for start_idx in range(0, len(indices), self.batch_size):
                end_idx = min(start_idx + self.batch_size, len(indices))
                batch_indices = indices[start_idx:end_idx]
                
                # Get sequence and target
                batch = self.current_chunk[batch_indices]
                X = batch[:, :-1, :]
                y = batch[:, -1, :]
                
                # Keep tensors on CPU for pin_memory
                yield X, y
            
            # Free memory after processing chunk
            del self.current_chunk
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            self.current_chunk = None
    
    def __len__(self):
        return self.total_sequences // self.batch_size
    
    def cleanup(self):
        """Clean up temporary files."""
        if hasattr(self, 'current_chunk'):
            del self.current_chunk
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        try:
            for f in self.chunk_dir.glob('chunk_*.npz'):
                f.unlink()
            self.chunk_dir.rmdir()
        except Exception as e:
            logging.warning(f"Failed to clean up temp files: {e}")