# Script for climate_prediction/main.py
import os
import sys
from pathlib import Path
import logging
import pandas as pd
import numpy as np
from datetime import datetime
import json
import yaml
from tqdm import tqdm
import torch
from typing import Tuple, Dict

import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from climate_prediction.src.data_processing.data_processing import DataProcessor
from climate_prediction.src.data_processing.preprocessor import ClimateDataPreprocessor
from climate_prediction.src.data_processing.feature_engineering import FeatureEngineer
from climate_prediction.src.data_processing.data_validator import DataValidator
from climate_prediction.src.models.train_evaluate import ModelTrainEvaluate
from climate_prediction.src.utils.logger import ProgressLogger
from climate_prediction.src.utils.config_manager import ConfigManager
from climate_prediction.src.utils.gpu_manager import gpu_manager
from climate_prediction.src.visualization.visualization_manager import VisualizationManager
from climate_prediction.src.pipeline.data_pipeline import DataPipeline
from climate_prediction.src.pipeline.model_pipeline import ModelPipeline

class ClimateModelPipeline:
    def __init__(self, config_path: str = "config/model_config.yaml"):
        self.config_path = config_path
        self.setup_directories()
        self.logger = ProgressLogger(name="ClimateModelPipeline")
        self.config_manager = ConfigManager(self.config_path)
        self.config = self.config_manager.get_config()
        self.visualizer = VisualizationManager(self.logger)
        self.device = gpu_manager.get_device()
        
        if torch.cuda.is_available():
            self.logger.log_info(f"Using GPU: {torch.cuda.get_device_name(0)}")
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            self.logger.log_info(f"GPU Memory: {gpu_memory:.2f} GB")
        else:
            self.logger.log_info("Using CPU")
    
    def setup_directories(self):
        """Create necessary directories."""
        directories = [
            'outputs/data',
            'outputs/models',
            'outputs/plots',
            'outputs/logs',
            'outputs/predictions',
            'outputs/metrics',
            'outputs/metadata',
            'config'
        ]
        for directory in directories:
            os.makedirs(os.path.join(project_root, directory), exist_ok=True)
    
    def process_data(self, data_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Process and validate climate data."""
        try:
            self.logger.log_info("Starting data processing pipeline...")
            
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
            
            # Process paths
            train_processed = os.path.join(project_root, 'outputs/data/train_processed.csv.gz')
            test_processed = os.path.join(project_root, 'outputs/data/test_processed.csv.gz')
            train_final = os.path.join(project_root, 'outputs/data/train_final.csv.gz')
            test_final = os.path.join(project_root, 'outputs/data/test_final.csv.gz')
            
            # Process data
            processor.process_file(data_path, train_processed)
            
            # Preprocess data
            preprocessor.preprocess(train_processed, train_final)
            preprocessor.preprocess(test_processed, test_final)
            
            # Feature engineering
            feature_engineer.process_file(train_final, train_final)
            feature_engineer.process_file(test_final, test_final)
            
            # Validate processed data
            train_report = validator.validate_file(train_final)
            test_report = validator.validate_file(test_final)
            
            if not (train_report.is_valid and test_report.is_valid):
                raise ValueError("Data validation failed. Check validation reports.")
            
            # Load final processed data
            train_df = pd.read_csv(train_final, compression='gzip', parse_dates=['DATETIME'], index_col='DATETIME')
            test_df = pd.read_csv(test_final, compression='gzip', parse_dates=['DATETIME'], index_col='DATETIME')
            
            return train_df, test_df
                
        except Exception as e:
            self.logger.log_error(f"Error in data processing: {str(e)}")
            raise
    
    def train_and_evaluate_models(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
        """Train and evaluate all models."""
        try:
            self.logger.log_info("Starting model training and evaluation...")
            
            train_evaluate = ModelTrainEvaluate(config_path=self.config_manager.config_path)
            results = train_evaluate.run_pipeline(train_df)
            
            self.logger.log_info("Model training and evaluation completed")
            return results
            
        except Exception as e:
            self.logger.log_error(f"Error in model training and evaluation: {str(e)}")
            raise
    
    def generate_visualizations(self, train_df: pd.DataFrame, test_df: pd.DataFrame, results: dict):
        """Generate all required visualizations."""
        try:
            self.logger.log_info("Generating visualizations...")
            
            with tqdm(desc="Generating visualizations", total=5) as pbar:
                # Time series decomposition
                self.visualizer.plot_components(train_df[self.config['preprocessing']['target_variable']])
                pbar.update(1)
                
                # Model predictions comparison
                self.visualizer.plot_predictions(
                    test_df[self.config['preprocessing']['target_variable']],
                    results['evaluation_results']
                )
                pbar.update(1)
                
                # Model metrics comparison
                metrics_df = pd.DataFrame({
                    name: res['metrics']
                    for name, res in results['evaluation_results'].items()
                })
                self.visualizer.plot_metrics_comparison(metrics_df)
                pbar.update(1)
                
                # Forecast horizon plots
                self.visualizer.plot_forecast_horizon(
                    test_df[self.config['preprocessing']['target_variable']],
                    results['future_predictions'],
                    forecast_start=test_df.index[-1]
                )
                pbar.update(1)
                
                # Feature importance plots
                if 'feature_importance' in results:
                    self.visualizer.plot_feature_importance(results['feature_importance'])
                pbar.update(1)
            
            self.logger.log_info("Visualization generation completed")
            
        except Exception as e:
            self.logger.log_error(f"Error generating visualizations: {str(e)}")
            raise
    
    def save_results(self, results: Dict):
        """Save model results and metadata."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save metrics
        metrics_path = os.path.join(project_root, 
                                  f'outputs/metrics/final_metrics_{timestamp}.json')
        
        metrics = {name: res['metrics'] 
                  for name, res in results['evaluation'].items()}
        pd.DataFrame(metrics).to_json(metrics_path)
        
        # Save metadata
        metadata = {
            'timestamp': timestamp,
            'config_used': self.config,
            'model_performance': metrics
        }
        
        metadata_path = os.path.join(project_root, 
                                   f'outputs/metadata/execution_metadata_{timestamp}.json')
        pd.DataFrame([metadata]).to_json(metadata_path)
    
    def run_pipeline(self) -> Dict:
        try:
            start_time = datetime.now()
            self.logger.log_info("Starting climate modeling pipeline...")
            
            # Initialize pipelines
            data_pipeline = DataPipeline(self.config_path)
            model_pipeline = ModelPipeline(self.config_path)
            
            # Run data processing pipeline
            data_results = data_pipeline.run_pipeline()
            if data_results['status'] != 'success':
                raise ValueError("Data processing failed")
            
            # Run model pipeline
            model_results = model_pipeline.run_pipeline()
            if model_results['status'] != 'success':
                raise ValueError("Model training failed")
            
            # Save results
            self.save_results(model_results)
            
            end_time = datetime.now()
            duration = end_time - start_time
            
            return {
                'status': 'success',
                'duration': str(duration),
                'data_info': data_results['data_info'],
                'models_trained': list(model_results['evaluation'].keys()),
                'best_model': min(model_results['evaluation'].items(),
                                key=lambda x: x[1]['metrics']['rmse'])[0]
            }
            
        except Exception as e:
            self.logger.log_error(f"Pipeline failed: {str(e)}")
            raise

def main():
    #data_path = os.path.join(project_root.parent, 'all_data/csvs_concatenated/concatenated_full/full_concatenated.csv.gz')
    config_path = os.path.join(project_root, "climate_prediction", "config", "model_config.yaml")
    
    try:
        pipeline = ClimateModelPipeline(config_path)
        summary = pipeline.run_pipeline()
        
        print("\nPipeline Execution Summary:")
        print("==========================")
        print(f"Status: {summary['status']}")
        print(f"Duration: {summary['duration']}")
        print(f"Models Trained: {', '.join(summary['models_trained'])}")
        print(f"Best Performing Model: {summary['best_model']}")
        
    except Exception as e:
        logging.error(f"Pipeline execution failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()