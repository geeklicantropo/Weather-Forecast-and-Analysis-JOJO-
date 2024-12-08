import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Tuple, List
import logging
from datetime import datetime
import gc
import psutil
from tqdm import tqdm

class ClimateDataPreprocessor:
    VALID_RANGES = {
        'TEMPERATURA DO AR - BULBO SECO HORARIA °C': (-40, 50),
        'PRECIPITACÃO TOTAL HORÁRIO MM': (0, 500),
        'PRESSAO ATMOSFERICA AO NIVEL DA ESTACAO HORARIA MB': (800, 1100),
        'RADIACAO GLOBAL KJ/M²': (0, 5000),
        'UMIDADE RELATIVA DO AR HORARIA %': (0, 100),
        'VENTO VELOCIDADE HORARIA M/S': (0, 100)
    }
    
    INVALID_VALUES = [-9999.0, -999.0, -99.0, 9999.0]
    
    def __init__(self, target_variable: str, logger):
        self.target_variable = target_variable
        self.logger = logger
        self.chunk_size = self._calculate_optimal_chunk_size()
        
    def _calculate_optimal_chunk_size(self) -> int:
        """Calculate optimal chunk size based on available memory."""
        available_memory = psutil.virtual_memory().available
        target_memory_per_chunk = available_memory * 0.1  # Use 10% of available memory
        estimated_row_size = 1024  # bytes per row
        return max(10000, min(int(target_memory_per_chunk / estimated_row_size), 50000))
        
    def preprocess(self, input_path: str, output_path: str) -> None:
        """Process data in chunks with memory management."""
        try:
            # Count total rows for progress tracking
            total_rows = sum(1 for _ in pd.read_csv(input_path, chunksize=self.chunk_size))
            
            with tqdm(total=total_rows, desc="Preprocessing data") as pbar:
                for i, chunk in enumerate(pd.read_csv(input_path, chunksize=self.chunk_size)):
                    try:
                        processed_chunk = self._process_chunk(chunk)
                        
                        # Save processed chunk
                        mode = 'w' if i == 0 else 'a'
                        header = i == 0
                        processed_chunk.to_csv(output_path, 
                                            compression='gzip',
                                            mode=mode,
                                            header=header,
                                            index=True)
                        
                        pbar.update(len(chunk))
                        self._cleanup_memory()
                        
                    except Exception as e:
                        self.logger.error(f"Error processing chunk {i}: {str(e)}")
                        continue
                        
        except Exception as e:
            self.logger.error(f"Preprocessing failed: {str(e)}")
            raise
            
    def _process_chunk(self, chunk: pd.DataFrame) -> pd.DataFrame:
        """Process a single data chunk."""
        chunk = chunk.copy()
        
        # Sequential processing steps
        chunk = self._handle_invalid_values(chunk)
        chunk = self._validate_meteorological_data(chunk)
        chunk = self._handle_missing_values(chunk)
        chunk = self._handle_climate_outliers(chunk)
        chunk = self._apply_quality_control(chunk)
        
        return chunk
        
    def _handle_invalid_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Replace invalid values with NaN."""
        for col in df.select_dtypes(include=[np.number]).columns:
            df[col] = df[col].replace(self.INVALID_VALUES, np.nan)
        return df
        
    def _validate_meteorological_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate values are within expected ranges."""
        for column, (min_val, max_val) in self.VALID_RANGES.items():
            if column in df.columns:
                mask = (df[column] < min_val) | (df[column] > max_val)
                df.loc[mask, column] = np.nan
        return df
        
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset"""
        df = df.copy()
        
        # Only process numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        invalid_values = [-9999.0, -999.0, -99.0, 9999.0]
        
        for col in numeric_cols:
            df[col] = df[col].replace(invalid_values, np.nan)
            df[col] = df[col].astype('float32')
        
        # Forward fill and interpolate
        df[numeric_cols] = df[numeric_cols].ffill(limit=6)
        df[numeric_cols] = df[numeric_cols].interpolate(method='linear', limit=24, limit_direction='both')
        
        return df
        
    def _get_correlated_columns(self, df: pd.DataFrame, target_col: str) -> List[str]:
        """Find columns correlated with target for interpolation."""
        correlations = df.corr()[target_col].abs()
        return correlations[correlations > 0.5].index.tolist()
        
    def _interpolate_with_correlations(self, df: pd.DataFrame, target_col: str, 
                                     corr_cols: List[str]) -> pd.Series:
        """Interpolate using correlated variables when available."""
        # Simple interpolation if no correlations
        if len(corr_cols) <= 1:
            return df[target_col].interpolate(method='linear', limit=24)
            
        # Multiple variable interpolation
        not_all_null = df[corr_cols].notna().any(axis=1)
        df.loc[not_all_null, target_col] = df.loc[not_all_null, target_col].interpolate()
        return df[target_col]
        
    def _handle_climate_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect and handle climate outliers considering physical relationships."""
        for col in self.VALID_RANGES.keys():
            if col in df.columns:
                # Calculate rolling statistics
                window = min(24, len(df))  # 24 hours or chunk length
                rolling_stats = df[col].rolling(window=window, center=True)
                
                median = rolling_stats.median()
                std = rolling_stats.std()
                
                # Define outlier bounds
                lower_bound = median - 3 * std
                upper_bound = median + 3 * std
                
                # Check if extreme values are physically consistent
                outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
                if outliers.any():
                    valid_extremes = self._verify_weather_pattern(df, col, outliers)
                    df.loc[outliers & ~valid_extremes, col] = np.nan
                    
        return df
        
    def _verify_weather_pattern(self, df: pd.DataFrame, column: str, 
                              outlier_mask: pd.Series) -> pd.Series:
        """Verify if extreme values are supported by weather patterns."""
        valid_pattern = pd.Series(False, index=df.index)
        
        if column == 'TEMPERATURA DO AR - BULBO SECO HORARIA °C':
            # Validate temperature drops with precipitation
            temp_drops = df[column].diff() < -5
            rain_events = df['PRECIPITACÃO TOTAL HORÁRIO MM'] > 0
            valid_pattern |= (temp_drops & rain_events)
            
        elif column == 'PRESSAO ATMOSFERICA AO NIVEL DA ESTACAO HORARIA MB':
            # Validate pressure changes with temperature changes
            pressure_changes = df[column].diff().abs() > 10
            temp_changes = df['TEMPERATURA DO AR - BULBO SECO HORARIA °C'].diff().abs() > 5
            valid_pattern |= (pressure_changes & temp_changes.shift(-6))
            
        return valid_pattern
        
    def _apply_quality_control(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply final quality control checks."""
        max_hourly_changes = {
            'TEMPERATURA DO AR - BULBO SECO HORARIA °C': 10,
            'PRESSAO ATMOSFERICA AO NIVEL DA ESTACAO HORARIA MB': 20,
            'UMIDADE RELATIVA DO AR HORARIA %': 50
        }
        
        for col, max_change in max_hourly_changes.items():
            if col in df.columns:
                changes = df[col].diff().abs()
                invalid_mask = changes > max_change
                df.loc[invalid_mask, col] = np.nan
                
        return df
        
    def _cleanup_memory(self):
        """Clean up memory after chunk processing."""
        gc.collect()
        memory_usage = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        if memory_usage > 1024:  # If usage exceeds 1GB
            self.logger.warning(f"High memory usage: {memory_usage:.2f} MB")

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Initialize preprocessor
    preprocessor = ClimateDataPreprocessor(
        target_variable="TEMPERATURA DO AR - BULBO SECO HORARIA °C",
        logger=logger
    )
    
    # Process train and test files
    input_files = [
        "Scripts/climate_prediction/outputs/data/train_processed.csv.gz",
        "Scripts/climate_prediction/outputs/data/test_processed.csv.gz"
    ]
    output_files = [
        "Scripts/climate_prediction/outputs/data/train_preprocessed.csv.gz",
        "Scripts/climate_prediction/outputs/data/test_preprocessed.csv.gz"
    ]
    
    for input_path, output_path in zip(input_files, output_files):
        preprocessor.preprocess(input_path, output_path)