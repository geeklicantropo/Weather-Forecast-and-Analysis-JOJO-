# src/data_processing/data_loader.py
import dask.dataframe as dd
import pandas as pd
from datetime import datetime
from ..utils.logger import log_execution_time

class DataLoader:
    def __init__(self, data_path, logger):
        self.data_path = data_path
        self.logger = logger
    
    @log_execution_time(logger)
    def load_data(self):
        """Load data using Dask with progress tracking"""
        self.logger.log_info(f"Starting to load data from {self.data_path}")
        
        try:
            # Load data in chunks using Dask
            df = dd.read_csv(self.data_path, compression='gzip')
            self.logger.log_info(f"Initial data load complete. Partitions: {df.npartitions}")
            
            # Show progress for datetime conversions
            self.logger.log_info("Converting datetime columns...")
            with self.logger.get_progress_bar(
                total=3,
                desc="DateTime conversion",
                unit="steps"
            ) as pbar:
                # Convert date column
                df['DATA YYYY-MM-DD'] = dd.to_datetime(df['DATA YYYY-MM-DD'])
                pbar.update(1)
                
                # Create datetime string
                df['DATETIME'] = df['DATA YYYY-MM-DD'].astype(str) + ' ' + df['HORA UTC']
                pbar.update(1)
                
                # Convert to datetime
                df['DATETIME'] = dd.to_datetime(df['DATETIME'])
                pbar.update(1)
            
            # Log data info
            self.logger.log_info(f"Data loading completed. Columns: {list(df.columns)}")
            self.logger.log_info(f"Memory usage estimate: {df.memory_usage().sum().compute() / 1e9:.2f} GB")
            
            return df
            
        except Exception as e:
            self.logger.log_error(f"Error loading data: {str(e)}")
            raise