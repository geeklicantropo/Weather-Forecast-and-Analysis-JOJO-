#src/data_processing/preprocessor.py
import os
import sys
from typing import Dict, List, Optional, Tuple, Any, Union
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import gc
import psutil
from tqdm import tqdm
from pathlib import Path

class ClimateDataPreprocessor:
    """
    Preprocessor for climate data that handles large datasets through chunk processing
    and proper aggregation.
    """
    
    def __init__(self, target_variable: str, logger: Any):
        """
        Initialize the preprocessor.

        Args:
            target_variable: The main variable to focus on for preprocessing
            logger: Logger instance for tracking preprocessing operations
        """
        self.target_variable = target_variable
        self.logger = logger
        self.chunk_size = self._calculate_optimal_chunk_size()
        self.group_columns = [
            'DATA YYYY-MM-DD', 'UF', 'ESTACAO', 'CODIGO (WMO)',
            'LATITUDE', 'LONGITUDE', 'ALTITUDE', 'YEAR'
        ]
        
        # Define valid ranges for meteorological variables
        self.VALID_RANGES: Dict[str, Tuple[float, float]] = {
            'TEMPERATURA DO AR - BULBO SECO HORARIA °C': (-40, 50),
            'PRECIPITACÃO TOTAL HORÁRIO MM': (0, 500),
            'PRESSAO ATMOSFERICA AO NIVEL DA ESTACAO HORARIA MB': (800, 1100),
            'RADIACAO GLOBAL KJ/M²': (0, 5000),
            'UMIDADE RELATIVA DO AR HORARIA %': (0, 100),
            'VENTO VELOCIDADE HORARIA M/S': (0, 100)
        }
        
        # Define invalid values to be replaced
        self.INVALID_VALUES: List[float] = [-9999.0, -999.0, -99.0, 9999.0]
        
        # Initialize processing state
        self.buffer = pd.DataFrame()
        self.temp_files: List[str] = []
        
    def _calculate_optimal_chunk_size(self) -> int:
        """Calculate optimal chunk size based on available system memory."""
        try:
            available_memory = psutil.virtual_memory().available * 0.8
            estimated_row_bytes = 200
            optimal_chunk_size = int(available_memory * 0.05 / estimated_row_bytes)
            return max(20_000, min(500_000, optimal_chunk_size))
        except Exception as e:
            self.logger.error(f"Error calculating chunk size: {str(e)}")
            return 100_000

    def _create_temp_dir(self, output_path: str) -> str:
        """Create temporary directory for intermediate files."""
        temp_dir = os.path.join(os.path.dirname(output_path), 'temp')
        os.makedirs(temp_dir, exist_ok=True)
        return temp_dir

    def _get_aggregation_dict(self, df: pd.DataFrame) -> Dict[str, str]:
        """Create aggregation dictionary for numeric and categorical columns."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        # Exclude group columns from aggregation
        agg_numeric_cols = [col for col in numeric_cols if col not in self.group_columns]
        agg_categorical_cols = [col for col in categorical_cols if col not in self.group_columns]
        
        # Create aggregation dictionary
        agg_dict = {}
        agg_dict.update({col: 'mean' for col in agg_numeric_cols})
        agg_dict.update({col: lambda x: x.mode().iloc[0] if not x.mode().empty else pd.NA 
                        for col in agg_categorical_cols})
        
        return agg_dict

    def preprocess(self, input_path: str, output_path: str) -> None:
        """
        Preprocess the climate data file.
        
        Args:
            input_path: Path to input data file
            output_path: Path where processed data should be saved
        """
        checkpoint_path = f"{output_path}.checkpoint"
        temp_dir = self._create_temp_dir(output_path)
        
        try:
            # Initialize or resume from checkpoint
            last_processed_row = 0
            if os.path.exists(checkpoint_path):
                with open(checkpoint_path, 'r') as f:
                    last_processed_row = int(f.read().strip())
            
            # Calculate total rows for progress tracking
            total_rows = sum(1 for _ in pd.read_csv(input_path, chunksize=self.chunk_size))
            
            # Initialize processing state
            buffer = pd.DataFrame()
            current_group = None
            temp_file_counter = 0
            
            # Process data in chunks
            self.logger.info(f"Starting preprocessing with chunk size: {self.chunk_size}")
            with tqdm(total=total_rows, initial=last_processed_row) as pbar:
                reader = pd.read_csv(input_path, chunksize=self.chunk_size, skiprows=last_processed_row)
                
                for chunk_idx, chunk in enumerate(reader):
                    try:
                        # Basic preprocessing steps
                        chunk = self._preprocess_chunk(chunk)
                        
                        # Combine with buffer if exists
                        if not buffer.empty:
                            chunk = pd.concat([buffer, chunk])
                        
                        # Group data
                        grouped = chunk.groupby(self.group_columns)
                        
                        # Process complete groups
                        complete_groups = grouped.filter(
                            lambda x: x[self.group_columns[0]].iloc[-1] != 
                            chunk[self.group_columns[0]].iloc[-1]
                        )
                        
                        if not complete_groups.empty:
                            # Aggregate complete groups
                            agg_dict = self._get_aggregation_dict(complete_groups)
                            aggregated = complete_groups.groupby(self.group_columns).agg(agg_dict)
                            
                            # Save to temporary file
                            temp_file = os.path.join(temp_dir, f'temp_{temp_file_counter}.csv.gz')
                            aggregated.to_csv(temp_file, compression='gzip')
                            self.temp_files.append(temp_file)
                            temp_file_counter += 1
                        
                        # Update buffer with incomplete groups
                        buffer = grouped.filter(
                            lambda x: x[self.group_columns[0]].iloc[-1] == 
                            chunk[self.group_columns[0]].iloc[-1]
                        )
                        
                        # Update progress
                        pbar.update(len(chunk))
                        gc.collect()
                        
                        # Save checkpoint
                        with open(checkpoint_path, 'w') as f:
                            f.write(str((chunk_idx + 1) * self.chunk_size))
                            
                    except Exception as e:
                        self.logger.error(f"Error processing chunk {chunk_idx}: {str(e)}")
                        continue
            
            # Process final buffer
            if not buffer.empty:
                agg_dict = self._get_aggregation_dict(buffer)
                final_aggregated = buffer.groupby(self.group_columns).agg(agg_dict)
                final_temp_file = os.path.join(temp_dir, f'temp_{temp_file_counter}.csv.gz')
                final_aggregated.to_csv(final_temp_file, compression='gzip')
                self.temp_files.append(final_temp_file)
            
            # Merge all temporary files
            self._merge_temp_files(output_path)
            
            # Cleanup
            self._cleanup(temp_dir, checkpoint_path)
            
            self.logger.info(f"Successfully completed preprocessing: {output_path}")
            
        except Exception as e:
            self.logger.error(f"Preprocessing failed: {str(e)}")
            self._cleanup(temp_dir, checkpoint_path)
            raise

    def _preprocess_chunk(self, chunk: pd.DataFrame) -> pd.DataFrame:
        """Apply all preprocessing steps to a single chunk."""
        chunk = chunk.copy()
        
        # Remove HORA UTC column if exists
        if 'HORA UTC' in chunk.columns:
            chunk = chunk.drop('HORA UTC', axis=1)
        
        # Handle invalid values
        chunk = self._handle_invalid_values(chunk)
        
        # Validate ranges
        chunk = self._validate_ranges(chunk)
        
        # Handle missing values
        chunk = self._handle_missing_values(chunk)
        
        return chunk

    def _handle_invalid_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Replace invalid values with NaN."""
        for col in df.select_dtypes(include=[np.number]).columns:
            df[col] = df[col].replace(self.INVALID_VALUES, np.nan)
        return df

    def _validate_ranges(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate values are within acceptable ranges."""
        for column, (min_val, max_val) in self.VALID_RANGES.items():
            if column in df.columns:
                mask = (df[column] < min_val) | (df[column] > max_val)
                df.loc[mask, column] = np.nan
        return df

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values through forward fill and interpolation."""
        # Forward fill for small gaps
        df = df.ffill(limit=6)
        
        # Interpolate numeric columns for larger gaps
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].interpolate(
            method='linear',
            limit=24,
            limit_direction='both'
        )
        
        return df

    def _merge_temp_files(self, output_path: str) -> None:
        """Merge all temporary files into final output file."""
        first_file = True
        
        for temp_file in tqdm(self.temp_files, desc="Merging temporary files"):
            if os.path.exists(temp_file):
                chunk = pd.read_csv(temp_file, compression='gzip')
                chunk.to_csv(
                    output_path,
                    mode='w' if first_file else 'a',
                    header=first_file,
                    index=False,
                    compression={'method': 'gzip', 'compresslevel': 1}
                )
                first_file = False

    def _cleanup(self, temp_dir: str, checkpoint_path: str) -> None:
        """Clean up temporary files and checkpoint."""
        # Remove temporary files
        for temp_file in self.temp_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        
        # Remove temporary directory
        if os.path.exists(temp_dir):
            os.rmdir(temp_dir)
        
        # Remove checkpoint file
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)

if __name__ == "__main__":
    # Setup basic logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Example usage
    preprocessor = ClimateDataPreprocessor(
        target_variable="TEMPERATURA DO AR - BULBO SECO HORARIA °C",
        logger=logger
    )
    
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