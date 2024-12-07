import os
import pandas as pd
import numpy as np
from datetime import datetime
import gc
import logging
from tqdm import tqdm
from pathlib import Path
import psutil
from typing import Optional

class DataSplitter:
    def __init__(self, input_path: str, output_dir: str, chunk_size: int = 20000):
        self.input_path = input_path
        self.output_dir = output_dir
        self.chunk_size = chunk_size
        self.logger = self._setup_logging()
        self._setup_directories()
        
    def _setup_logging(self) -> logging.Logger:
        logger = logging.getLogger("DataSplitter")
        logger.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # Create file handler
        log_dir = os.path.join(self.output_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(
            os.path.join(log_dir, f"split_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        return logger
    
    def _setup_directories(self):
        os.makedirs(self.output_dir, exist_ok=True)
        
    def _process_chunk(self, chunk: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split chunk into train and test based on year."""
        test_mask = chunk['YEAR'] == 2024
        return chunk[~test_mask], chunk[test_mask]
    
    def _save_chunk(self, chunk: pd.DataFrame, filename: str, mode: str = 'a'):
        """Save chunk to file with proper compression."""
        if mode == 'w' or not os.path.exists(filename):
            chunk.to_csv(filename, compression='gzip', mode='w', index=False)
        else:
            chunk.to_csv(filename, compression='gzip', mode='a', header=False, index=False)
    
    def _cleanup_memory(self):
        """Force garbage collection and log memory usage."""
        gc.collect()
        memory_usage = psutil.Process().memory_info().rss / 1024 / 1024
        self.logger.debug(f"Current memory usage: {memory_usage:.2f} MB")
    
    def split_data(self):
        """Split data into train and test sets."""
        train_path = os.path.join(self.output_dir, 'train_full_concatenated.csv.gz')
        test_path = os.path.join(self.output_dir, 'test_full_concatenated.csv.gz')
        
        self.logger.info("Starting data splitting process")
        
        try:
            # Get total rows for progress bar
            total_rows = sum(1 for _ in pd.read_csv(self.input_path, chunksize=self.chunk_size))
            
            with tqdm(total=total_rows, desc="Processing chunks") as pbar:
                for i, chunk in enumerate(pd.read_csv(self.input_path, chunksize=self.chunk_size)):
                    # Process chunk
                    train_chunk, test_chunk = self._process_chunk(chunk)
                    
                    # Save chunks
                    mode = 'w' if i == 0 else 'a'
                    if not train_chunk.empty:
                        self._save_chunk(train_chunk, train_path, mode)
                    if not test_chunk.empty:
                        self._save_chunk(test_chunk, test_path, mode)
                    
                    # Update progress and cleanup
                    pbar.update(len(chunk))
                    self._cleanup_memory()
            
            self.logger.info("Data splitting completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error during data splitting: {str(e)}")
            raise

def main():
    # Setup paths
    input_path = "Scripts/all_data/csvs_concatenated/concatenated_full/full_concatenated.csv.gz"
    output_dir = "Scripts/climate_prediction/outputs/data"
    
    # Initialize and run splitter
    splitter = DataSplitter(input_path, output_dir)
    splitter.split_data()

if __name__ == "__main__":
    main()