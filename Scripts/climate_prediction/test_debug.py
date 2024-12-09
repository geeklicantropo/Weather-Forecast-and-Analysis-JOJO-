import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import torch
from typing import Dict, Tuple
from tqdm import tqdm
import logging
import shutil

project_root = Path(__file__).parent

from src.data_processing.data_processing import DataProcessor
from src.data_processing.preprocessor import ClimateDataPreprocessor
from src.data_processing.feature_engineering import FeatureEngineer
from src.data_processing.data_validator import DataValidator
from src.models.train_evaluate import ModelTrainEvaluate
from src.utils.logger import ProgressLogger
from src.utils.config_manager import ConfigManager
from src.utils.gpu_manager import gpu_manager
from src.visualization.visualization_manager import VisualizationManager
from src.pipeline.data_pipeline import DataPipeline
from src.pipeline.model_pipeline import ModelPipeline

from src.models.lstm_model import LSTMModel
from src.models.sarima_model import SARIMAModel 
from src.models.tft_model import TFTModel

from src.models.model_evaluator import ModelEvaluator

class DebugModelPipeline:
    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = os.path.join(project_root, "config/model_config.yaml")
        self.config_path = config_path
        self.setup_logging()
        self.setup_directories()
        self.config_manager = ConfigManager(self.config_path)
        self.config = self.config_manager.get_config()
        self.visualizer = VisualizationManager(self.logger)
        self.device = gpu_manager.get_device()
        self.target_variable = self.config['preprocessing']['target_variable']
        
        self.source_paths = {
            'train': os.path.join(project_root, "outputs/data/train_final.csv.gz"),
            'test': os.path.join(project_root, "outputs/data/test_final.csv.gz")
        }
        self.debug_paths = {
            'train': os.path.join(project_root, "../outputs/data/debug_train_final.csv.gz"),
            'test': os.path.join(project_root, "../outputs/data/debug_test_final.csv.gz")
        }
        
        self._log_system_info()

    def evaluate_models(self, models: Dict[str, object], test_data: pd.DataFrame) -> Dict:
        """Evaluate all trained models."""
        evaluator = ModelEvaluator(self.logger)
        results = {}

        for name, model in models.items():
            predictions = model.predict(test_data)
            metrics = evaluator.evaluate_model(model, test_data, predictions)
            results[name] = {
                'metrics': metrics,
                'predictions': predictions
            }

        return results

    def train_models(self, train_data: pd.DataFrame, test_data: pd.DataFrame) -> Dict[str, object]:
        """Train models with sample data."""
        if not isinstance(train_data, pd.DataFrame) or not isinstance(test_data, pd.DataFrame):
            raise ValueError("Input data must be pandas DataFrames")

        models = {}
        model_classes = {
            'lstm': LSTMModel,
            'sarima': SARIMAModel,
            'tft': TFTModel
        }
        
        for name, model_class in model_classes.items():
            try:
                self.logger.log_info(f"Training {name.upper()} model")
                
                # Extract relevant configuration for each model
                if name == 'lstm':
                    model_config = {
                        'sequence_length': self.config['models']['lstm']['training']['sequence_length'],
                        'hidden_size': self.config['models']['lstm']['architecture']['hidden_size'],
                        'num_layers': self.config['models']['lstm']['architecture']['num_layers'],
                        'forecast_horizon': self.config['output']['forecast']['horizon'] // 24
                    }
                elif name == 'sarima':
                    model_config = {
                        'order_ranges': self.config['models']['sarima']['order'],
                        'seasonal_order_ranges': self.config['models']['sarima']['seasonal_order']
                    }
                elif name == 'tft':
                    model_config = {
                        'max_prediction_length': self.config['models']['tft']['training']['max_prediction_length'],
                        'max_encoder_length': self.config['models']['tft']['training']['max_encoder_length']
                    }
                
                model = model_class(self.target_variable, self.logger, **model_config)
                
                # Convert index to datetime if needed
                if not isinstance(train_data.index, pd.DatetimeIndex):
                    train_data.index = pd.to_datetime(train_data.index)
                if not isinstance(test_data.index, pd.DatetimeIndex):
                    test_data.index = pd.to_datetime(test_data.index)
                
                # Train model
                model.train(train_data, test_data)
                models[name] = model
                
            except Exception as e:
                self.logger.log_error(f"Error training {name} model: {str(e)}")
                continue
                
        return models
    
    def setup_logging(self):
        """Setup logging for debug pipeline."""
        log_dir = os.path.join(project_root, 'outputs/logs')
        os.makedirs(log_dir, exist_ok=True)
        
        self.logger = ProgressLogger(name="DebugModelPipeline")
        log_file = os.path.join(log_dir, f'debug_pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        self.logger.logger.addHandler(file_handler)
        self.logger.log_info("Debug pipeline initialized")
    
    def setup_directories(self):
        """Create necessary directories for debug outputs."""
        directories = [
            'outputs/data',
            'outputs/models',
            'outputs/plots',
            'outputs/logs',
            'outputs/predictions',
            'outputs/metrics',
            'outputs/metadata'
        ]
        
        self.logger.log_info("Creating output directories")
        with tqdm(total=len(directories), desc="Setting up directories") as pbar:
            for directory in directories:
                dir_path = os.path.join(project_root, directory)
                os.makedirs(dir_path, exist_ok=True)
                self.logger.log_info(f"Created directory: {dir_path}")
                pbar.update(1)
    
    def _log_system_info(self):
        """Log system and GPU information."""
        self.logger.log_info("\n=== System Information ===")
        if torch.cuda.is_available():
            self.logger.log_info(f"GPU Device: {torch.cuda.get_device_name(0)}")
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            self.logger.log_info(f"Total GPU Memory: {gpu_memory:.2f} GB")
            self.logger.log_info(f"CUDA Version: {torch.version.cuda}")
        else:
            self.logger.log_info("Running on CPU")
        self.logger.log_info("========================\n")
    
    def create_sample_datasets(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        try:
            self.logger.log_info("Creating sample datasets")
            
            with tqdm(total=2, desc="Creating samples") as pbar:
                self.logger.log_info(f"Loading source train data from: {self.source_paths['train']}")
                train_df = pd.read_csv(self.source_paths['train'], compression='gzip')
                train_sample = train_df.sample(n=10000, random_state=42)
                
                self.logger.log_info(f"Saving train sample ({len(train_sample)} rows)")
                train_sample.sort_index().to_csv(self.debug_paths['train'], compression='gzip', index=False)
                pbar.update(1)
                
                self.logger.log_info(f"Loading source test data from: {self.source_paths['test']}")
                test_df = pd.read_csv(self.source_paths['test'], compression='gzip')
                test_sample = test_df.sample(n=1000, random_state=42)
                
                self.logger.log_info(f"Saving test sample ({len(test_sample)} rows)")
                test_sample.sort_index().to_csv(self.debug_paths['test'], compression='gzip', index=False)
                pbar.update(1)
            
            return train_sample, test_sample
            
        except Exception as e:
            self.logger.log_error(f"Error creating sample datasets: {str(e)}")
            raise
    
    def run_debug_pipeline(self):
        try:
            start_time = datetime.now()
            
            # Load sample data
            train_data = pd.read_csv(self.debug_paths['train'], compression='gzip')
            test_data = pd.read_csv(self.debug_paths['test'], compression='gzip')
            
            # Set datetime index
            train_data['DATETIME'] = pd.to_datetime(train_data['DATA YYYY-MM-DD'])
            test_data['DATETIME'] = pd.to_datetime(test_data['DATA YYYY-MM-DD'])
            train_data.set_index('DATETIME', inplace=True)
            test_data.set_index('DATETIME', inplace=True)
            
            # Train models
            models = self.train_models(train_data, test_data)
            
            # Evaluate models
            evaluation_results = self.evaluate_models(models, test_data)
            
            end_time = datetime.now()
            duration = end_time - start_time
            
            return {
                'status': 'success',
                'duration': str(duration),
                'models': models,
                'evaluation': evaluation_results,
                'models_trained': list(models.keys())
            }
            
        except Exception as e:
            self.logger.log_error(f"Debug pipeline failed: {str(e)}")
            raise
    
    def _save_debug_results(self, results: Dict):
        """Save debug results and metrics."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.logger.log_info("Saving metrics")
        metrics = {name: res['metrics'] 
                  for name, res in results['evaluation'].items()}
        metrics_path = os.path.join(project_root, 
                                  f'outputs/metrics/debug_metrics_{timestamp}.json')
        pd.DataFrame(metrics).to_json(metrics_path)
        
        self.logger.log_info("Saving metadata")
        metadata = {
            'timestamp': timestamp,
            'config_used': self.config,
            'model_performance': metrics
        }
        metadata_path = os.path.join(project_root,
                                   f'outputs/metadata/debug_metadata_{timestamp}.json')
        pd.DataFrame([metadata]).to_json(metadata_path)
    
    def _log_summary(self, results: Dict):
        """Log execution summary."""
        self.logger.log_info("\n=== Debug Pipeline Summary ===")
        self.logger.log_info(f"Status: {results['status']}")
        self.logger.log_info(f"Total Duration: {results['duration']}")
        self.logger.log_info("\nModels Trained:")
        for model in results['models_trained']:
            self.logger.log_info(f"  - {model}")
        self.logger.log_info(f"\nBest Performing Model: {results['best_model']}")
        self.logger.log_info("============================")

def main():
    try:
        pipeline = DebugModelPipeline()
        summary = pipeline.run_debug_pipeline()
        
        print("\nDebug Pipeline Summary:")
        print("=======================")
        print(f"Status: {summary['status']}")
        print(f"Duration: {summary['duration']}")
        print(f"Models Trained: {', '.join(summary['models_trained'])}")
        print(f"Best Model: {summary['best_model']}")
        print("\nCheck the log file in outputs/logs for detailed execution information")
        
    except Exception as e:
        logging.error(f"Debug pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()