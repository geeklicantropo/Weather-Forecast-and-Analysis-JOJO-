# Scripts/climate_prediction/src/pipeline/data_pipeline.py
import os
from pathlib import Path
import logging
from typing import Dict, Tuple
import pandas as pd
from tqdm import tqdm

from ..data_processing.data_processing import DataProcessor
from ..data_processing.preprocessor import ClimateDataPreprocessor
from ..data_processing.feature_engineering import FeatureEngineer
from ..data_processing.data_validator import DataValidator
from ..utils.logger import ProgressLogger
from ..utils.config_manager import ConfigManager
from ..utils.gpu_manager import gpu_manager
from ..utils.file_checker import FileChecker

class DataPipeline:
    def __init__(self, config_path: str = "config/model_config.yaml"):
        self.logger = ProgressLogger(name="DataPipeline")
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.get_config()
        self.device = gpu_manager.get_device()
        self.file_checker = FileChecker()
        self._setup_directories()
        
    def _setup_directories(self):
        """Create necessary directories."""
        dirs = [
            'outputs/data',
            'outputs/models',
            'outputs/plots',
            'outputs/metrics',
            'outputs/logs'
        ]
        for dir_path in dirs:
            os.makedirs(os.path.join("Scripts/climate_prediction", dir_path), exist_ok=True)

    def process_datasets(self) -> Tuple[str, str]:
        """Process both train and test datasets with file checks."""
        # Check if final files already exist
        if self.file_checker.check_final_exists():
            self.logger.log_info("Final processed files already exist. Skipping processing.")
            return (
                str(self.file_checker.get_file_path('final', 'train')),
                str(self.file_checker.get_file_path('final', 'test'))
            )

        processed_paths = {}
        try:
            # Initialize processors
            processor = DataProcessor(chunk_size=20000, logger=self.logger)
            preprocessor = ClimateDataPreprocessor(
                target_variable=self.config['preprocessing']['target_variable'],
                logger=self.logger
            )
            feature_engineer = FeatureEngineer(
                target_variable=self.config['preprocessing']['target_variable'],
                logger=self.logger
            )
            validator = DataValidator(self.config_manager, self.logger)

            # Process each stage
            for stage in ['processed', 'preprocessed', 'final']:
                should_process, reason = self.file_checker.should_process_stage(stage)
                if not should_process:
                    self.logger.log_info(f"Skipping {stage} stage: {reason}")
                    continue

                self.logger.log_info(f"Processing {stage} stage...")
                for dataset_type in ['train', 'test']:
                    input_path = str(self.file_checker.get_file_path(
                        'split' if stage == 'processed' else stage,
                        dataset_type
                    ))
                    output_path = str(self.file_checker.get_file_path(stage, dataset_type))

                    if stage == 'processed':
                        processor.process_file(input_path, output_path)
                    elif stage == 'preprocessed':
                        preprocessor.preprocess(input_path, output_path)
                    elif stage == 'final':
                        feature_engineer.process_file(input_path, output_path)
                        processed_paths[dataset_type] = output_path

            # Validate final datasets
            for dataset_type in ['train', 'test']:
                validation_report = validator.validate_file(processed_paths[dataset_type])
                if not validation_report.is_valid:
                    raise ValueError(f"Validation failed for {dataset_type} dataset")

            return processed_paths['train'], processed_paths['test']

        except Exception as e:
            self.logger.log_error(f"Data processing failed: {str(e)}")
            raise

    def get_data_info(self, train_path: str, test_path: str) -> Dict:
        """Get information about processed datasets."""
        info = {}
        for name, path in [('train', train_path), ('test', test_path)]:
            df = pd.read_csv(path, compression='gzip', nrows=5)
            info[name] = {
                'columns': list(df.columns),
                'dtypes': df.dtypes.to_dict(),
                'file_size': os.path.getsize(path) / (1024 * 1024)  # MB
            }
        return info

    def run_pipeline(self) -> Dict:
        """Execute complete data processing pipeline with file checks."""
        try:
            self.logger.log_info("Starting data processing pipeline")
            
            # Check if final files exist
            if self.file_checker.check_final_exists():
                train_path = str(self.file_checker.get_file_path('final', 'train'))
                test_path = str(self.file_checker.get_file_path('final', 'test'))
                self.logger.log_info("Using existing processed files")
            else:
                # Process datasets
                train_path, test_path = self.process_datasets()
            
            # Get processing summary
            data_info = self.get_data_info(train_path, test_path)
            
            return {
                'status': 'success',
                'paths': {
                    'train': train_path,
                    'test': test_path
                },
                'data_info': data_info
            }
            
        except Exception as e:
            self.logger.log_error(f"Pipeline failed: {str(e)}")
            raise