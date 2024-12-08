# src/data_processing/feature_engineering.py
import pandas as pd
import numpy as np
from typing import Dict, List
import gc
from tqdm import tqdm
import psutil
import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

class FeatureEngineer:
    def __init__(self, target_variable: str, logger, chunk_size: int = 500000):
        self.target_variable = target_variable
        self.logger = logger
        self.chunk_size = chunk_size
        self.temp_col = target_variable
        
    def process_file(self, input_path: str, output_path: str) -> None:
        """Process and engineer features for a file."""
        try:
            if not os.path.exists(input_path):
                self.logger.log_error(f"Input file not found: {input_path}")
                return None
                
            total_rows = sum(1 for _ in pd.read_csv(input_path, chunksize=self.chunk_size))
            buffer = pd.DataFrame()
            overlap = 24  # One day overlap for rolling features
            
            with tqdm(total=total_rows, desc="Engineering features") as pbar:
                for i, chunk in enumerate(pd.read_csv(input_path, chunksize=self.chunk_size)):
                    try:
                        # Combine with buffer
                        current_data = pd.concat([buffer, chunk]) if not buffer.empty else chunk
                        
                        # Process features
                        processed_chunk = self._process_chunk(current_data)
                        
                        # Save non-overlapping part
                        if i > 0:  # Not first chunk
                            processed_chunk = processed_chunk.iloc[overlap:]
                        
                        # Save chunk
                        mode = 'w' if i == 0 else 'a'
                        header = i == 0
                        processed_chunk.to_csv(output_path, compression='gzip',
                                            mode=mode, header=header, index=True)
                        
                        # Update buffer
                        buffer = chunk.iloc[-overlap:].copy() if i < total_rows - 1 else pd.DataFrame()
                        
                        # Update progress and cleanup
                        pbar.update(len(chunk))
                        self._cleanup_memory()
                        
                    except Exception as e:
                        self.logger.error(f"Error processing chunk {i}: {str(e)}")
                        continue
                        
        except Exception as e:
            self.logger.error(f"Feature engineering failed: {str(e)}")
            raise
            
    def _process_chunk(self, chunk: pd.DataFrame) -> pd.DataFrame:
        """Process a single chunk of data with essential features only."""
        df = chunk.copy()
        
        # Ensure datetime index
        df.index = pd.to_datetime(df.index)
        
        # Add basic temporal features
        df['hour'] = df.index.hour
        df['day'] = df.index.day
        df['month'] = df.index.month
        df['day_of_week'] = df.index.dayofweek
        
        # Add cyclic encoding for temporal features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Add essential rolling features (24-hour window only)
        if self.temp_col in df.columns:
            rolling = df[self.temp_col].rolling(window=24, min_periods=1)
            df['temp_mean_24h'] = rolling.mean()
            df['temp_std_24h'] = rolling.std()
        
        # Basic weather features
        if 'PRECIPITACÃO TOTAL HORÁRIO MM' in df.columns:
            df['is_raining'] = (df['PRECIPITACÃO TOTAL HORÁRIO MM'] > 0).astype(int)
        
        if 'UMIDADE RELATIVA DO AR HORARIA %' in df.columns:
            df['is_humid'] = (df['UMIDADE RELATIVA DO AR HORARIA %'] > 70).astype(int)
        
        return df
        
    def _cleanup_memory(self):
        """Clean up memory after chunk processing."""
        gc.collect()
        memory_usage = psutil.Process().memory_info().rss / 1024 / 1024
        if memory_usage > 4096:
            self.logger.warning(f"High memory usage: {memory_usage:.2f} MB")

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    engineer = FeatureEngineer(
        target_variable="TEMPERATURA DO AR - BULBO SECO HORARIA °C",
        logger=logger
    )
    
    input_files = [
        "Scripts/climate_prediction/outputs/data/train_preprocessed.csv.gz",
        "Scripts/climate_prediction/outputs/data/test_preprocessed.csv.gz"
    ]
    output_files = [
        "Scripts/climate_prediction/outputs/data/train_final.csv.gz",
        "Scripts/climate_prediction/outputs/data/test_final.csv.gz"
    ]
    
    for input_path, output_path in zip(input_files, output_files):
        engineer.process_file(input_path, output_path)