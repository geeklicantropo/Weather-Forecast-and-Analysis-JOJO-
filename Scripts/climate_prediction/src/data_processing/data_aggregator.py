import pandas as pd
import numpy as np
from typing import Dict, Optional
import gc
import os
from datetime import datetime
from tqdm import tqdm
import psutil

class DataAggregator:
    def __init__(self, logger):
        self.logger = logger
        self.aggregation_functions = {
            'TEMPERATURA DO AR - BULBO SECO HORARIA °C': ['mean', 'min', 'max', 'std'],
            'PRECIPITACÃO TOTAL HORÁRIO MM': ['sum', 'max'],
            'PRESSAO ATMOSFERICA AO NIVEL DA ESTACAO HORARIA MB': ['mean'],
            'UMIDADE RELATIVA DO AR HORARIA %': ['mean', 'min', 'max'],
            'VENTO VELOCIDADE HORARIA M/S': ['mean', 'max']
        }
        self.validation_thresholds = {
            'min_rows': 10,
            'max_null_ratio': 0.3,
            'max_std_multiplier': 3
        }

    def aggregate_chunk(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate data with specified functions per column."""
        try:
            # Ensure datetime index
            if not isinstance(df.index, pd.DatetimeIndex):
                df = df.copy()
                df['DATETIME'] = pd.to_datetime(df['DATETIME'])
                df.set_index('DATETIME', inplace=True)

            # Group by hour and aggregate
            aggregated = df.groupby(df.index.floor('H')).agg(self.aggregation_functions)
            
            # Flatten multi-index columns
            aggregated.columns = [f"{col[0]}_{col[1]}" if isinstance(col, tuple) else col 
                                for col in aggregated.columns]

            # Calculate additional features
            self._add_derived_features(aggregated)

            return aggregated

        except Exception as e:
            self.logger.log_error(f"Aggregation failed: {str(e)}")
            return pd.DataFrame()

    def _add_derived_features(self, df: pd.DataFrame) -> None:
        """Add derived features to aggregated data."""
        try:
            # Temperature range
            if 'TEMPERATURA DO AR - BULBO SECO HORARIA °C_max' in df.columns and \
               'TEMPERATURA DO AR - BULBO SECO HORARIA °C_min' in df.columns:
                df['TEMPERATURA_RANGE'] = df['TEMPERATURA DO AR - BULBO SECO HORARIA °C_max'] - \
                                        df['TEMPERATURA DO AR - BULBO SECO HORARIA °C_min']

            # Precipitation intensity
            if 'PRECIPITACÃO TOTAL HORÁRIO MM_sum' in df.columns:
                df['PRECIPITACAO_INTENSITY'] = df['PRECIPITACÃO TOTAL HORÁRIO MM_sum'] / 1.0  # per hour

            # Add temporal features
            df['hour'] = df.index.hour
            df['day'] = df.index.day
            df['month'] = df.index.month
            df['year'] = df.index.year

        except Exception as e:
            self.logger.log_error(f"Error adding derived features: {str(e)}")

    def validate_aggregated_data(self, df: pd.DataFrame) -> bool:
        """Validate aggregated data quality."""
        try:
            # Check minimum rows
            if len(df) < self.validation_thresholds['min_rows']:
                self.logger.log_warning(f"Insufficient rows: {len(df)}")
                return False

            # Check null ratios
            null_ratios = df.isnull().mean()
            if (null_ratios > self.validation_thresholds['max_null_ratio']).any():
                self.logger.log_warning("Excessive null values detected")
                return False

            # Check for outliers
            for col in df.select_dtypes(include=[np.number]).columns:
                mean = df[col].mean()
                std = df[col].std()
                outliers = abs(df[col] - mean) > (std * self.validation_thresholds['max_std_multiplier'])
                if outliers.sum() / len(df) > 0.01:  # More than 1% outliers
                    self.logger.log_warning(f"High outlier ratio in {col}")
                    return False

            return True

        except Exception as e:
            self.logger.log_error(f"Validation error: {str(e)}")
            return False

    def get_aggregation_stats(self, df: pd.DataFrame) -> Dict:
        """Get statistics about the aggregation process."""
        return {
            'rows_before': len(df),
            'rows_after': len(df.groupby(df.index.floor('H'))),
            'null_ratios': df.isnull().mean().to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum() / 1024**2  # MB
        }

class ClimateDataPreprocessor:
    def __init__(self, target_variable: str, logger):
        self.target_variable = target_variable
        self.logger = logger
        self.aggregator = DataAggregator(logger)
        self.chunk_size = self._calculate_optimal_chunk_size()
        
    def _calculate_optimal_chunk_size(self) -> int:
        """Calculate optimal chunk size based on available system memory."""        
        try:
            # Get available memory (80% of total RAM)
            available_memory = psutil.virtual_memory().available * 0.8
            
            # Estimate row size (bytes) based on our columns
            estimated_row_bytes = 200  # Conservative estimate for our dataset
            
            # Calculate chunks to use 5% of available memory
            target_chunk_memory = available_memory * 0.05
            optimal_chunk_size = int(target_chunk_memory / estimated_row_bytes)
            
            # Set bounds: minimum 20,000, maximum 500,000
            optimal_chunk_size = max(20_000, min(500_000, optimal_chunk_size))
            
            self.logger.log_info(f"Calculated optimal chunk size: {optimal_chunk_size:,} rows")
            return optimal_chunk_size
            
        except Exception as e:
            self.logger.log_warning(f"Error calculating chunk size: {str(e)}. Using default: 100,000")
            return 100_000
        
    def preprocess(self, input_path: str, output_path: str) -> None:
        """Process data in chunks with checkpoint handling."""
        checkpoint_path = output_path + '.checkpoint'
        partial_path = output_path + '.partial'
        
        try:
            # Initialize/resume from checkpoint
            last_processed_row = 0
            if os.path.exists(checkpoint_path):
                with open(checkpoint_path, 'r') as f:
                    last_processed_row = int(f.read().strip())
            
            total_rows = sum(1 for _ in pd.read_csv(input_path, chunksize=self.chunk_size))
            first_chunk = not os.path.exists(partial_path)
            
            with tqdm(total=total_rows, initial=last_processed_row, desc="Preprocessing data") as pbar:
                # Process chunks
                chunks = pd.read_csv(input_path, chunksize=self.chunk_size, skiprows=last_processed_row)
                
                for i, chunk in enumerate(chunks, start=last_processed_row // self.chunk_size):
                    # Basic preprocessing
                    processed_chunk = self._handle_invalid_values(chunk)
                    processed_chunk = self._handle_missing_values(processed_chunk)
                    
                    # Aggregate data
                    processed_chunk = self.aggregator.aggregate_chunk(processed_chunk)
                    
                    if processed_chunk.empty:
                        self.logger.log_warning(f"Empty chunk after processing at index {i}")
                        continue
                        
                    # Validate aggregated data
                    if not self.aggregator.validate_aggregated_data(processed_chunk):
                        self.logger.log_error(f"Validation failed for chunk {i}")
                        continue
                    
                    # Save processed chunk
                    mode = 'w' if first_chunk else 'a'
                    header = first_chunk
                    processed_chunk.to_csv(
                        partial_path,
                        mode=mode,
                        header=header,
                        index=True,
                        compression={'method': 'gzip', 'compresslevel': 1}
                    )
                    
                    # Update checkpoint
                    with open(checkpoint_path, 'w') as f:
                        f.write(str((i + 1) * self.chunk_size))
                    
                    if first_chunk:
                        first_chunk = False
                    
                    pbar.update(len(chunk))
                    gc.collect()
            
            # Processing complete, rename partial to final
            if os.path.exists(partial_path):
                os.replace(partial_path, output_path)
                os.remove(checkpoint_path)
                self.logger.log_info(f"Successfully created preprocessed file: {output_path}")
            
        except Exception as e:
            self.logger.log_error(f"Preprocessing failed: {str(e)}")
            raise
            
    def _handle_invalid_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Replace invalid values with NaN."""
        df = df.copy()
        invalid_values = [-9999.0, -999.0, -99.0, 9999.0]
        
        for col in df.select_dtypes(include=[np.number]).columns:
            df[col] = df[col].replace(invalid_values, np.nan)
            
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values with forward fill and interpolation."""
        df = df.copy()
        
        # Forward fill for gaps ≤ 6 hours
        df = df.ffill(limit=6)
        
        # Interpolate remaining gaps ≤ 24 hours
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].interpolate(
            method='linear',
            limit=24,
            limit_direction='both'
        )
        
        return df