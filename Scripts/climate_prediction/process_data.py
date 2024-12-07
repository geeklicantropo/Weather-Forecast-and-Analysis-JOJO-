import os
import sys
from pathlib import Path
import logging
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.data_processing.data_processing import DataProcessor
from src.data_processing.preprocessor import ClimateDataPreprocessor
from src.data_processing.feature_engineering import FeatureEngineer
from src.data_processing.data_validator import DataValidator
from src.utils.logger import ProgressLogger
from src.utils.config_manager import ConfigManager

def main():
    # Initialize components
    logger = ProgressLogger(name="DataProcessing")
    config_manager = ConfigManager("config/model_config.yaml")
    config = config_manager.get_config()
    
    # Setup directories
    output_dir = os.path.join(project_root, 'outputs/data')
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Initialize processors
        processor = DataProcessor(chunk_size=20000, logger=logger)
        preprocessor = ClimateDataPreprocessor(
            target_variable=config['preprocessing']['target_variable'],
            logger=logger
        )
        feature_engineer = FeatureEngineer(
            target_variable=config['preprocessing']['target_variable'],
            logger=logger
        )
        validator = DataValidator(config_manager, logger)

        # Process train data
        logger.log_info("Processing training data...")
        input_train = os.path.join(output_dir, 'train_full_concatenated.csv.gz')
        processed_train = os.path.join(output_dir, 'train_processed.csv.gz')
        preprocessed_train = os.path.join(output_dir, 'train_preprocessed.csv.gz')
        final_train = os.path.join(output_dir, 'train_final.csv.gz')

        processor.process_file(input_train, processed_train)
        preprocessor.preprocess(processed_train, preprocessed_train)
        feature_engineer.process_file(preprocessed_train, final_train)

        # Process test data
        logger.log_info("Processing test data...")
        input_test = os.path.join(output_dir, 'test_full_concatenated.csv.gz')
        processed_test = os.path.join(output_dir, 'test_processed.csv.gz')
        preprocessed_test = os.path.join(output_dir, 'test_preprocessed.csv.gz')
        final_test = os.path.join(output_dir, 'test_final.csv.gz')

        processor.process_file(input_test, processed_test)
        preprocessor.preprocess(processed_test, preprocessed_test)
        feature_engineer.process_file(preprocessed_test, final_test)

        # Validate final datasets
        logger.log_info("Validating processed datasets...")
        train_report = validator.validate_file(final_train)
        test_report = validator.validate_file(final_test)

        if not (train_report.is_valid and test_report.is_valid):
            raise ValueError("Data validation failed. Check validation reports.")

        logger.log_info("Data processing completed successfully")
        
    except Exception as e:
        logger.log_error(f"Data processing failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()