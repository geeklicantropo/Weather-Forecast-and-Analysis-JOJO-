# src/data_processing/data_validator.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging
from datetime import datetime
import gc
import psutil
from tqdm import tqdm

@dataclass
class ValidationReport:
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    missing_data: Dict
    data_quality_score: float
    memory_usage: Dict

class DataValidator:
    def __init__(self, config_manager, logger, chunk_size: int = 500000):
        self.config = config_manager.get_validation_config()
        self.preprocessing_config = config_manager.get_preprocessing_config()
        self.logger = logger
        self.chunk_size = chunk_size
        self.target_variable = self.preprocessing_config['target_variable']
        self.required_columns = self._get_required_columns()
        
    def _get_required_columns(self) -> set:
        """Define required columns based on configuration."""
        return {
            self.target_variable,
            'PRECIPITACÃO TOTAL HORÁRIO MM',
            'PRESSAO ATMOSFERICA AO NIVEL DA ESTACAO HORARIA MB',
            'UMIDADE RELATIVA DO AR HORARIA %',
            'VENTO VELOCIDADE HORARIA M/S'
        }

    def _validate_datatypes(self, df: pd.DataFrame) -> List[str]:
        """Validate essential data types."""
        errors = []
        
        # Handle datetime index/column
        try:
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index, format='mixed')
        except Exception as e:
            errors.append(f"Invalid datetime format in index: {str(e)}")
        
        # Validate numeric columns
        numeric_columns = [self.target_variable]
        for column in numeric_columns:
            if column in df.columns:
                try:
                    pd.to_numeric(df[column], errors='coerce')
                except Exception as e:
                    errors.append(f"Invalid numeric data in {column}: {str(e)}")
                    
        return errors
    
    def validate_file(self, file_path: str) -> ValidationReport:
        """Validate file with essential checks only."""
        errors = []
        warnings = []
        missing_data = {'total_missing_rate': 0.0}
        
        try:
            total_chunks = sum(1 for _ in pd.read_csv(file_path, chunksize=self.chunk_size))
            processed_chunks = 0
            
            with tqdm(total=total_chunks, desc="Validating data") as pbar:
                for chunk in pd.read_csv(file_path, chunksize=self.chunk_size):
                    # Validate data types
                    dtype_errors = self._validate_datatypes(chunk)
                    errors.extend(dtype_errors)
                    
                    # Check for missing values
                    missing_rate = chunk[self.target_variable].isnull().mean()
                    missing_data['total_missing_rate'] += missing_rate
                    
                    if missing_rate > 0.2:  # 20% threshold
                        warnings.append(f"High missing rate in chunk: {missing_rate:.2%}")
                    
                    processed_chunks += 1
                    pbar.update(1)
                    gc.collect()
            
            if processed_chunks > 0:
                missing_data['total_missing_rate'] /= processed_chunks
            
            # Calculate simple quality score
            quality_score = 1.0 - missing_data['total_missing_rate']
            
            return ValidationReport(
                is_valid=len(errors) == 0,
                errors=errors,
                warnings=warnings,
                missing_data=missing_data,
                data_quality_score=quality_score,
                memory_usage={'current': psutil.Process().memory_info().rss / 1024 / 1024}
            )
            
        except Exception as e:
            self.logger.error(f"Validation failed: {str(e)}")
            raise

if __name__ == "__main__":
    import logging
    from ..utils.config_manager import ConfigManager
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    config_manager = ConfigManager()
    
    validator = DataValidator(config_manager, logger)
    
    input_files = [
        "Scripts/climate_prediction/outputs/data/train_final.csv.gz",
        "Scripts/climate_prediction/outputs/data/test_final.csv.gz"
    ]
    
    for file_path in input_files:
        report = validator.validate_file(file_path)
        logger.info(f"\nValidation Report for {file_path}:")
        logger.info(f"Valid: {report.is_valid}")
        logger.info(f"Quality Score: {report.data_quality_score:.2f}")
        if report.errors:
            logger.error("Errors found:")
            for error in report.errors:
                logger.error(f"  - {error}")
        if report.warnings:
            logger.warning("Warnings found:")
            for warning in report.warnings:
                logger.warning(f"  - {warning}")