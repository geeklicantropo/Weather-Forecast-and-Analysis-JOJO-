# src/data_processing/data_loader.py
import dask.dataframe as dd
import pandas as pd
import cudf
import dask_cudf
from datetime import datetime
import psutil
import torch
from ..utils.logger import log_execution_time
from ..utils.gpu_manager import GPUManager

class DataLoader:
    def __init__(self, data_path, logger):
        self.data_path = data_path
        self.logger = logger
        self.gpu_manager = GPUManager()
        self.chunk_size = self._calculate_optimal_chunk_size()
    
    def _calculate_optimal_chunk_size(self):
        """Calculate optimal chunk size based on available memory"""
        available_memory = psutil.virtual_memory().available
        # Use 20% of available memory per chunk
        return int(available_memory * 0.2)
    
    def _check_gpu_availability(self):
        """Check if GPU is available and has enough memory"""
        if not torch.cuda.is_available():
            self.logger.log_info("GPU not available, falling back to CPU")
            return False
        
        gpu_memory = torch.cuda.get_device_properties(0).total_memory
        if gpu_memory < self.chunk_size:
            self.logger.log_info("Insufficient GPU memory, falling back to CPU")
            return False
        
        return True
    
    @log_execution_time
    def load_data(self):
        """Load data using either GPU or CPU based on availability"""
        self.logger.log_info(f"Starting to load data from {self.data_path}")
        
        try:
            use_gpu = self._check_gpu_availability()
            
            with self.logger.get_progress_bar(
                total=3,
                desc="Loading data",
                unit="steps"
            ) as pbar:
                # Load data
                if use_gpu:
                    df = self._load_with_gpu()
                else:
                    df = self._load_with_cpu()
                pbar.update(1)
                
                # Convert datetime columns
                df = self._convert_datetime_columns(df, use_gpu)
                pbar.update(1)
                
                # Validate loaded data
                self._validate_loaded_data(df)
                pbar.update(1)
            
            self._log_data_info(df)
            return df
            
        except Exception as e:
            self.logger.log_error(f"Error loading data: {str(e)}")
            raise
    
    def _load_with_gpu(self):
        """Load data using GPU acceleration"""
        try:
            with self.gpu_manager.memory_monitor():
                df = dask_cudf.read_csv(
                    self.data_path,
                    compression='gzip',
                    chunksize=self.chunk_size
                )
            return df
        except Exception as e:
            self.logger.log_warning(f"GPU loading failed: {str(e)}. Falling back to CPU.")
            return self._load_with_cpu()
    
    def _load_with_cpu(self):
        """Load data using CPU"""
        return dd.read_csv(
            self.data_path,
            compression='gzip',
            blocksize=self.chunk_size
        )
    
    def _convert_datetime_columns(self, df, use_gpu):
        """Convert datetime columns based on processing mode"""
        try:
            date_col = 'DATA YYYY-MM-DD'
            time_col = 'HORA UTC'
            
            if use_gpu:
                df[date_col] = dask_cudf.to_datetime(df[date_col])
                df['DATETIME'] = df[date_col].astype(str) + ' ' + df[time_col]
                df['DATETIME'] = dask_cudf.to_datetime(df['DATETIME'])
            else:
                df[date_col] = dd.to_datetime(df[date_col])
                df['DATETIME'] = df[date_col].astype(str) + ' ' + df[time_col]
                df['DATETIME'] = dd.to_datetime(df['DATETIME'])
            
            return df
        except Exception as e:
            self.logger.log_error(f"DateTime conversion failed: {str(e)}")
            raise
    
    def _validate_loaded_data(self, df):
        """Validate the loaded data"""
        if df.npartitions == 0:
            raise ValueError("No data loaded")
        
        required_cols = ['DATA YYYY-MM-DD', 'HORA UTC']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    def _log_data_info(self, df):
        """Log information about the loaded data"""
        self.logger.log_info(f"Data loading completed. Columns: {list(df.columns)}")
        memory_usage = df.memory_usage().sum().compute() / 1e9
        self.logger.log_info(f"Memory usage estimate: {memory_usage:.2f} GB")
        self.logger.log_info(f"Number of partitions: {df.npartitions}")