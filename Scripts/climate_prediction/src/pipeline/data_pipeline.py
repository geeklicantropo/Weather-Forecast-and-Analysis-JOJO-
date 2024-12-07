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

class DataPipeline:
    def __init__(self, config_path: str = "config/model_config.yaml"):
        self.logger = ProgressLogger(name="DataPipeline")
        self.config = ConfigManager(config_path).get_config()
        self.device = gpu_manager.get_device()
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
        """Process both train and test datasets."""
        datasets = {
            'train': 'train_full_concatenated.csv.gz',
            'test': 'test_full_concatenated.csv.gz'
        }
        
        processed_paths = {}
        for dataset_type, filename in datasets.items():
            input_path = f"Scripts/climate_prediction/outputs/data/{filename}"
            final_path = f"Scripts/climate_prediction/outputs/data/{dataset_type}_final.csv.gz"
            
            self.logger.log_info(f"Processing {dataset_type} dataset")
            
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
            validator = DataValidator(self.config, self.logger)
            
            # Process pipeline stages
            processed_path = f"Scripts/climate_prediction/outputs/data/{dataset_type}_processed.csv.gz"
            preprocessed_path = f"Scripts/climate_prediction/outputs/data/{dataset_type}_preprocessed.csv.gz"
            
            processor.process_file(input_path, processed_path)
            preprocessor.preprocess(processed_path, preprocessed_path)
            feature_engineer.process_file(preprocessed_path, final_path)
            
            # Validate final dataset
            validation_report = validator.validate_file(final_path)
            if not validation_report.is_valid:
                raise ValueError(f"Validation failed for {dataset_type} dataset")
            
            processed_paths[dataset_type] = final_path
            
        return processed_paths['train'], processed_paths['test']

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
        """Execute complete data processing pipeline."""
        try:
            self.logger.log_info("Starting data processing pipeline")
            
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


if __name__ == "__main__":
    pipeline = DataPipeline()
    results = pipeline.run_pipeline()
    print(f"Pipeline completed. Processed files at: {results['paths']}")