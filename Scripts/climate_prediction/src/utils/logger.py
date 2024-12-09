# src/utils/logger.py
import logging
import sys
from datetime import datetime
import os
from tqdm.auto import tqdm
from functools import wraps

class ProgressLogger:
    def __init__(self, name='climate_prediction', log_dir='Scripts/climate_prediction/outputs/logs'):
        self.log_dir = log_dir
        
        # Create log directory if it doesn't exist
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        # Remove existing handlers to avoid duplicates
        self.logger.handlers = []
        
        # Create formatters
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # File handler
        log_file = os.path.join(
            log_dir, 
            f'processing_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        )
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
    def log_info(self, message):
        """Log info message"""
        self.logger.info(message)
    
    def log_error(self, message):
        """Log error message"""
        self.logger.error(message)
    
    def log_warning(self, message):
        """Log warning message"""
        self.logger.warning(message)
        
    # Alias methods to match standard logging interface
    info = log_info
    error = log_error
    warning = log_warning
    
    def get_progress_bar(self, total, desc="Processing", unit="items"):
        """Create a tqdm progress bar that also logs progress"""
        pbar = tqdm(total=total, desc=desc, unit=unit)
        return pbar
    
    def log_with_progress(self, iterable, desc="Processing", unit="items"):
        """Create a progress bar for an iterable that logs progress"""
        return tqdm(iterable, desc=desc, unit=unit)

def log_execution_time(logger):
    """Decorator to log execution time of functions"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = datetime.now()
            logger.log_info(f"Starting {func.__name__}")
            
            try:
                result = func(*args, **kwargs)
                end_time = datetime.now()
                execution_time = end_time - start_time
                logger.log_info(
                    f"Completed {func.__name__}. "
                    f"Execution time: {execution_time}"
                )
                return result
            except Exception as e:
                logger.log_error(
                    f"Error in {func.__name__}: {str(e)}"
                )
                raise
            
        return wrapper
    return decorator