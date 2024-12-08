import pandas as pd
import numpy as np
from datetime import datetime
import os
from typing import Dict, List, Tuple
from tqdm import tqdm
import gc
import psutil

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

class DataProcessor:
    def __init__(self, chunk_size: int = 20000, logger=None):
        self.chunk_size = chunk_size
        self.logger = logger
        self.required_columns = [
            'DATA YYYY-MM-DD', 'HORA UTC',
            'PRECIPITACÃO TOTAL HORÁRIO MM',
            'PRESSAO ATMOSFERICA AO NIVEL DA ESTACAO HORARIA MB',
            'RADIACAO GLOBAL KJ/M²',
            'TEMPERATURA DO AR - BULBO SECO HORARIA °C',
            'TEMPERATURA DO PONTO DE ORVALHO °C',
            'UMIDADE RELATIVA DO AR HORARIA %',
            'VENTO VELOCIDADE HORARIA M/S',
            'LATITUDE', 'LONGITUDE', 'ALTITUDE'
        ]
        self.dtypes = {
            'PRECIPITACÃO TOTAL HORÁRIO MM': 'float32',
            'PRESSAO ATMOSFERICA AO NIVEL DA ESTACAO HORARIA MB': 'float32',
            'RADIACAO GLOBAL KJ/M²': 'float32',
            'TEMPERATURA DO AR - BULBO SECO HORARIA °C': 'float32',
            'TEMPERATURA DO PONTO DE ORVALHO °C': 'float32',
            'UMIDADE RELATIVA DO AR HORARIA %': 'float32',
            'VENTO VELOCIDADE HORARIA M/S': 'float32',
            'LATITUDE': 'float32',
            'LONGITUDE': 'float32',
            'ALTITUDE': 'float32'
        }
        
    def process_file(self, input_path: str, output_path: str) -> None:
        """Process large CSV file in chunks"""
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Get total rows for progress bar
            total_rows = sum(1 for _ in pd.read_csv(input_path, chunksize=self.chunk_size))
            
            # Process in chunks with progress tracking
            with tqdm(total=total_rows, desc="Processing data") as pbar:
                for chunk_idx, chunk in enumerate(pd.read_csv(input_path, chunksize=self.chunk_size)):
                    try:
                        # Process chunk
                        processed_chunk = self._process_chunk(chunk)
                        
                        # Save chunk
                        mode = 'w' if chunk_idx == 0 else 'a'
                        header = chunk_idx == 0
                        processed_chunk.to_csv(
                            output_path, 
                            compression='gzip',
                            mode=mode,
                            header=header,
                            index=True
                        )
                        
                        # Update progress and cleanup
                        pbar.update(len(chunk))
                        self._cleanup_memory()
                        
                    except Exception as e:
                        self.logger.error(f"Error processing chunk {chunk_idx}: {str(e)}")
                        continue
                        
            self.logger.info("Data processing completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error in file processing: {str(e)}")
            raise

    def _process_chunk(self, chunk: pd.DataFrame) -> pd.DataFrame:
        """Process a single chunk of data"""
        # Validate columns
        self._validate_columns(chunk)
        
        # Convert datatypes
        chunk = self._convert_datatypes(chunk)
        
        # Create datetime index
        chunk = self._create_datetime_index(chunk)
        
        # Handle missing values
        chunk = self._handle_missing_values(chunk)
        
        # Handle outliers
        chunk = self._handle_outliers(chunk)
        
        return chunk

    def _validate_columns(self, df: pd.DataFrame) -> None:
        """Validate required columns are present"""
        missing_cols = set(self.required_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

    def _convert_datatypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert columns to appropriate datatypes"""
        df = df.copy()
        for col, dtype in self.dtypes.items():
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').astype(dtype)
        return df

    def _create_datetime_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create datetime index from date and time columns"""
        df = df.copy()

        def standardize_time(time_str):
            time_str = str(time_str)
            # Handle common formats
            if ':' in time_str:  # Already has minutes
                if len(time_str) == 4:  # Format: H:MM
                    return f"0{time_str}"
                return time_str
            else:  # Only hours
                time_str = str(time_str).zfill(2)  # Pad with zeros
                return f"{time_str}:00"

        # Convert date and standardize time format
        df['DATA YYYY-MM-DD'] = pd.to_datetime(df['DATA YYYY-MM-DD']).dt.strftime('%Y-%m-%d')
        df['HORA UTC'] = df['HORA UTC'].apply(standardize_time)
        
        # Create datetime
        df['DATETIME'] = pd.to_datetime(
            df['DATA YYYY-MM-DD'] + ' ' + df['HORA UTC'],
            format='%Y-%m-%d %H:%M'
        )
        return df.set_index('DATETIME')

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset"""
        df = df.copy()
        
        # Replace invalid values with NaN
        invalid_values = [-9999.0, -999.0, -99.0, 9999.0]
        for col in df.select_dtypes(include=[np.number]).columns:
            df[col] = df[col].replace(invalid_values, np.nan)
        
        # Forward fill short gaps (≤ 6 hours)
        df = df.ffill(limit=6)
        
        # Interpolate medium gaps (≤ 24 hours)
        df = df.infer_objects()
        df = df.interpolate(method='linear', limit=24, limit_direction='both')
        
        return df

    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle outliers using IQR method with rolling windows"""
        df = df.copy()
        
        for col in df.select_dtypes(include=[np.number]).columns:
            # Calculate rolling statistics
            rolling_window = min(24, len(df))  # 24 hours or length of chunk
            rolling_median = df[col].rolling(window=rolling_window, center=True).median()
            rolling_iqr = df[col].rolling(window=rolling_window, center=True).quantile(0.75) - \
                        df[col].rolling(window=rolling_window, center=True).quantile(0.25)
            
            # Define bounds
            lower_bound = rolling_median - 3 * rolling_iqr
            upper_bound = rolling_median + 3 * rolling_iqr
            
            # Replace outliers with rolling median, ensuring dtype consistency
            mask = (df[col] < lower_bound) | (df[col] > upper_bound)
            df.loc[mask, col] = rolling_median[mask].astype(df[col].dtype)
        
        return df

    def _cleanup_memory(self):
        """Force garbage collection and monitor memory usage"""
        gc.collect()
        memory_usage = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        if memory_usage > 1024:  # If usage exceeds 1GB
            self.logger.warning(f"High memory usage detected: {memory_usage:.2f} MB")

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    processor = DataProcessor(logger=logger)
    
    # Process train and test separately
    input_files = [
        "Scripts/climate_prediction/outputs/data/train_full_concatenated.csv.gz",
        "Scripts/climate_prediction/outputs/data/test_full_concatenated.csv.gz"
    ]
    output_files = [
        "Scripts/climate_prediction/outputs/data/train_processed.csv.gz",
        "Scripts/climate_prediction/outputs/data/test_processed.csv.gz"
    ]
    
    for input_path, output_path in zip(input_files, output_files):
        processor.process_file(input_path, output_path)