# src/data_processing/data_loader.py
import dask.dataframe as dd
import pandas as pd
import psutil
import torch
from typing import Optional, Tuple, Dict
import numpy as np
from src.utils.logger import log_execution_time
from src.utils.gpu_manager import gpu_manager

class DataLoader:
    def __init__(self, data_path: str, logger):
        self.data_path = data_path
        self.logger = logger
        self.device = gpu_manager.get_device()
        self.chunk_size = self._calculate_optimal_chunk_size()
        self.required_columns = [
            'DATA YYYY-MM-DD',
            'HORA UTC',
            'PRECIPITACÃO TOTAL HORÁRIO MM',
            'PRESSAO ATMOSFERICA AO NIVEL DA ESTACAO HORARIA MB',
            'PRESSÃO ATMOSFERICA MAX.NA HORA ANT. AUT MB',
            'PRESSÃO ATMOSFERICA MIN. NA HORA ANT. AUT MB',
            'RADIACAO GLOBAL KJ/M²',
            'TEMPERATURA DO AR - BULBO SECO HORARIA °C',
            'TEMPERATURA DO PONTO DE ORVALHO °C',
            'TEMPERATURA MÁXIMA NA HORA ANT. AUT °C',
            'TEMPERATURA MÍNIMA NA HORA ANT. AUT °C',
            'TEMPERATURA ORVALHO MAX. NA HORA ANT. AUT °C',
            'TEMPERATURA ORVALHO MIN. NA HORA ANT. AUT °C',
            'UMIDADE REL. MAX. NA HORA ANT. AUT %',
            'UMIDADE REL. MIN. NA HORA ANT. AUT %',
            'UMIDADE RELATIVA DO AR HORARIA %',
            'VENTO DIRECÃO HORARIA GR ° GR',
            'VENTO RAJADA MAXIMA M/S',
            'VENTO VELOCIDADE HORARIA M/S',
            'REGIAO',
            'UF',
            'ESTACAO',
            'CODIGO (WMO)',
            'LATITUDE',
            'LONGITUDE',
            'ALTITUDE'
        ]
    
    def _calculate_optimal_chunk_size(self) -> int:
        """Calculate optimal chunk size based on available memory."""
        available_memory = psutil.virtual_memory().available
        gpu_memory = 0
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            available_memory = min(available_memory, gpu_memory * 0.8)  # Use 80% of GPU memory
        
        # Target 10% of available memory per chunk
        memory_limit = available_memory * 0.1
        estimated_row_size = 1024  # bytes per row
        chunk_size = int(memory_limit / estimated_row_size)
        return max(5000, min(chunk_size, 20000))
    
    def _check_gpu_availability(self) -> Tuple[bool, Optional[str]]:
        """Check GPU availability and memory capacity."""
        if not torch.cuda.is_available():
            return False, "GPU not available"
        
        try:
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            if gpu_memory < 4 * 1024 * 1024 * 1024:  # 4GB minimum
                return False, "Insufficient GPU memory"
            
            # Check current GPU memory usage
            memory_info = self.gpu_manager.get_memory_info()
            if memory_info['free'] < 2 * 1024:  # 2GB minimum free
                return False, "Insufficient free GPU memory"
                
            return True, None
            
        except Exception as e:
            return False, f"GPU check failed: {str(e)}"
    
    def load_data(self) -> dd.DataFrame:
        """Load data using Dask for memory efficiency."""
        try:
            self.logger.log_info(f"Loading data with chunk size: {self.chunk_size}")
            
            dtypes = {
                'PRECIPITACÃO TOTAL HORÁRIO MM': 'float32',
                'PRESSAO ATMOSFERICA AO NIVEL DA ESTACAO HORARIA MB': 'float32',
                'PRESSÃO ATMOSFERICA MAX.NA HORA ANT. AUT MB': 'float32',
                'PRESSÃO ATMOSFERICA MIN. NA HORA ANT. AUT MB': 'float32',
                'RADIACAO GLOBAL KJ/M²': 'float32',
                'TEMPERATURA DO AR - BULBO SECO HORARIA °C': 'float32',
                'TEMPERATURA DO PONTO DE ORVALHO °C': 'float32',
                'TEMPERATURA MÁXIMA NA HORA ANT. AUT °C': 'float32',
                'TEMPERATURA MÍNIMA NA HORA ANT. AUT °C': 'float32',
                'TEMPERATURA ORVALHO MAX. NA HORA ANT. AUT °C': 'float32',
                'TEMPERATURA ORVALHO MIN. NA HORA ANT. AUT °C': 'float32',
                'UMIDADE REL. MAX. NA HORA ANT. AUT %': 'float32',
                'UMIDADE REL. MIN. NA HORA ANT. AUT %': 'float32',
                'UMIDADE RELATIVA DO AR HORARIA %': 'float32',
                'VENTO DIRECÃO HORARIA GR ° GR': 'float32',
                'VENTO RAJADA MAXIMA M/S': 'float32',
                'VENTO VELOCIDADE HORARIA M/S': 'float32',
                'LATITUDE': 'float32',
                'LONGITUDE': 'float32',
                'ALTITUDE': 'float32'
            }

            df = dd.read_csv(
                self.data_path,
                compression='gzip',
                blocksize=self.chunk_size * 1024,
                dtype=dtypes,
                engine='c'
            )

            df['DATETIME'] = df['DATA YYYY-MM-DD'].astype(str) + ' ' + df['HORA UTC']
            df['DATETIME'] = dd.to_datetime(df['DATETIME'])
            df = df.set_index('DATETIME')

            # Keep only necessary columns
            columns_to_keep = list(dtypes.keys()) + ['REGIAO', 'UF', 'ESTACAO', 'CODIGO (WMO)']
            df = df[columns_to_keep]

            # Optimize partitions
            partition_size = self.chunk_size * 1024 * 1024  # Convert to bytes
            n_partitions = max(1, df.memory_usage().sum().compute() // partition_size)
            df = df.repartition(npartitions=n_partitions)

            self.logger.log_info(f"Data loaded successfully with {df.npartitions} partitions")
            return df

        except Exception as e:
            self.logger.log_error(f"Error loading data: {str(e)}")
            raise
        
    def _validate_columns(self, df: dd.DataFrame):
        """Validate required columns are present."""
        missing_columns = set(self.required_columns) - set(df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
    
    def _process_datetime_columns(self, df: dd.DataFrame, use_gpu: bool) -> dd.DataFrame:
        """Process datetime columns based on processing mode."""
        try:
            date_col = 'DATA YYYY-MM-DD'
            time_col = 'HORA UTC'
            
            if use_gpu:
                # GPU processing using cuDF directly
                df[date_col] = df[date_col].astype('datetime64[ns]')
                df['DATETIME'] = df[date_col].astype(str).str.cat(df[time_col], sep=' ').astype('datetime64[ns]')
            else:
                # CPU processing using Dask
                df[date_col] = dd.to_datetime(df[date_col])
                df['DATETIME'] = df[date_col].astype(str) + ' ' + df[time_col]
                df['DATETIME'] = dd.to_datetime(df['DATETIME'])
            
            # Set datetime as index
            df = df.set_index('DATETIME')
            
            # Sort index
            df = df.map_partitions(lambda x: x.sort_index())
            
            return df
            
        except Exception as e:
            self.logger.log_error(f"DateTime conversion failed: {str(e)}")
            raise
    
    def _optimize_partitions(self, df: dd.DataFrame) -> dd.DataFrame:
        """Optimize partition sizes for better performance."""
        try:
            # Calculate optimal number of partitions based on data size and memory
            total_size = df.memory_usage(deep=True).sum().compute()
            target_partition_size = 256 * 1024 * 1024  # 256MB
            optimal_partitions = max(1, int(total_size / target_partition_size))
            
            # Repartition if needed
            current_partitions = df.npartitions
            if abs(current_partitions - optimal_partitions) > 0.2 * current_partitions:
                df = df.repartition(npartitions=optimal_partitions)
            
            return df
            
        except Exception as e:
            self.logger.log_warning(f"Partition optimization failed: {str(e)}")
            return df
    
    def get_data_info(self, df: dd.DataFrame) -> Dict:
        """Get information about the loaded data."""
        try:
            info = {
                'num_partitions': df.npartitions,
                'columns': list(df.columns),
                'dtypes': df.dtypes.to_dict(),
                'estimated_size': df.memory_usage(deep=True).sum().compute() / (1024**2),  # MB
                'start_date': df.index.min().compute(),
                'end_date': df.index.max().compute(),
                'num_rows': len(df)
            }
            
            # Calculate basic statistics for numeric columns
            numeric_cols = df.select_dtypes(include=['float32', 'float64', 'int32', 'int64']).columns
            stats = df[numeric_cols].describe().compute()
            info['statistics'] = stats.to_dict()
            
            return info
            
        except Exception as e:
            self.logger.log_error(f"Error getting data info: {str(e)}")
            return {}
    
    def validate_loaded_data(self, df: dd.DataFrame) -> bool:
        """Validate the loaded data for basic quality checks."""
        try:
            # Check for empty dataset
            if df.npartitions == 0:
                raise ValueError("No data loaded")
            
            # Check for required columns
            missing_cols = [col for col in self.required_columns if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Check for all null columns
            null_cols = [col for col in df.columns if df[col].isnull().all().compute()]
            if null_cols:
                self.logger.log_warning(f"Columns with all null values: {null_cols}")
            
            # Check temporal continuity
            time_diff = df.index.compute().to_series().diff()
            gaps = time_diff[time_diff > pd.Timedelta(hours=1)]
            if not gaps.empty:
                self.logger.log_warning(f"Found {len(gaps)} time gaps larger than 1 hour")
            
            return True
            
        except Exception as e:
            self.logger.log_error(f"Data validation failed: {str(e)}")
            return False
    
    def _log_data_info(self, df: dd.DataFrame):
        """Log information about the loaded data."""
        info = self.get_data_info(df)
        
        self.logger.log_info("=== Data Loading Summary ===")
        self.logger.log_info(f"Number of partitions: {info['num_partitions']}")
        self.logger.log_info(f"Number of rows: {info['num_rows']}")
        self.logger.log_info(f"Estimated size: {info['estimated_size']:.2f} MB")
        self.logger.log_info(f"Date range: {info['start_date']} to {info['end_date']}")
        self.logger.log_info(f"Number of columns: {len(info['columns'])}")
        
        if 'statistics' in info:
            self.logger.log_info("Basic statistics for numeric columns:")
            for col, stats in info['statistics'].items():
                self.logger.log_info(f"{col}:")
                for stat, value in stats.items():
                    self.logger.log_info(f"  {stat}: {value}")