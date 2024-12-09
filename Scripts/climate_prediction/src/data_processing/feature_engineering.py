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
                        
                        # Drop any Unnamed columns
                        processed_chunk = processed_chunk.loc[:, ~processed_chunk.columns.str.contains('^Unnamed')]
                        
                        #Save non-overlapping part
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
        
        # Add refined temporal features (excluding hour)
        df['day'] = df.index.day
        df['month'] = df.index.month
        df['day_of_week'] = df.index.dayofweek
        
        # Add cyclic encoding for temporal features
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # More aggressive missing value handling
        if self.temp_col in df.columns:
            # Fill missing temperatures with seasonal averages
            df[self.temp_col] = df[self.temp_col].groupby([df.index.month, df.index.day]).transform(
                lambda x: x.fillna(x.mean())
            )
            
            # Calculate rolling features with proper handling of NaN
            df['temp_mean_24'] = df[self.temp_col].rolling(
                window=24, min_periods=1, center=True
            ).mean()
            df['temp_std_24'] = df[self.temp_col].rolling(
                window=24, min_periods=1, center=True
            ).std()
        
        # Enhanced weather features with better NaN handling
        if 'PRECIPITACÃO TOTAL HORÁRIO MM' in df.columns:
            df['is_raining'] = df['PRECIPITACÃO TOTAL HORÁRIO MM'].fillna(0).gt(0).astype(int)
            
        if 'UMIDADE RELATIVA DO AR HORARIA %' in df.columns:
            df['UMIDADE RELATIVA DO AR HORARIA %'] = df['UMIDADE RELATIVA DO AR HORARIA %'].fillna(
                df['UMIDADE RELATIVA DO AR HORARIA %'].mean()
            )
            df['is_humid'] = (df['UMIDADE RELATIVA DO AR HORARIA %'] > 70).astype(int)
        
        # Comprehensive NaN handling
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # Fill NaN with forward fill, then backward fill, then column mean
        df[numeric_cols] = df[numeric_cols].fillna(method='ffill', limit=24)
        df[numeric_cols] = df[numeric_cols].fillna(method='bfill', limit=24)
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        
        # Drop any Unnamed columns
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        
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