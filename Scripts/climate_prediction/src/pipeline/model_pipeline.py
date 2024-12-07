# Scripts/climate_prediction/src/pipeline/model_pipeline.py
import os
from typing import Dict, Tuple, Optional
import pandas as pd
import torch
from datetime import datetime

from ..models.lstm_model import LSTMModel
from ..models.sarima_model import SARIMAModel
from ..models.tft_model import TFTModel
from ..models.model_evaluator import ModelEvaluator
from ..visualization.visualization_manager import VisualizationManager
from .data_pipeline import DataPipeline
from ..utils.logger import ProgressLogger
from ..utils.config_manager import ConfigManager
from ..utils.gpu_manager import gpu_manager

class ModelPipeline:
    def __init__(self, config_path: str = "config/model_config.yaml"):
        self.logger = ProgressLogger(name="ModelPipeline")
        self.config = ConfigManager(config_path).get_config()
        self.device = gpu_manager.get_device()
        self.data_pipeline = DataPipeline(config_path)
        self.visualizer = VisualizationManager(self.logger)
        self.target_variable = self.config['preprocessing']['target_variable']
        
    def load_data(self, train_path: str, test_path: str, 
                  chunk_size: int = 20000) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load processed data in chunks for training."""
        train_chunks = []
        test_chunks = []
        
        # Load train data
        for chunk in pd.read_csv(train_path, compression='gzip', chunksize=chunk_size):
            train_chunks.append(chunk)
        train_data = pd.concat(train_chunks)
        
        # Load test data
        for chunk in pd.read_csv(test_path, compression='gzip', chunksize=chunk_size):
            test_chunks.append(chunk)
        test_data = pd.concat(test_chunks)
        
        return train_data, test_data
        
    def train_models(self, train_data: pd.DataFrame, 
                    test_data: pd.DataFrame) -> Dict[str, object]:
        """Train all models with provided data."""
        models = {}
        
        model_classes = {
            'lstm': LSTMModel,
            'sarima': SARIMAModel,
            'tft': TFTModel
        }
        
        for name, model_class in model_classes.items():
            self.logger.log_info(f"Training {name.upper()} model")
            model = model_class(self.target_variable, self.logger)
            
            # Process data for model
            processed_train = model.preprocess_data(train_data)
            processed_test = model.preprocess_data(test_data)
            
            # Train model
            model.train(processed_train, processed_test)
            models[name] = model
            
            # Save model
            save_path = f"Scripts/climate_prediction/outputs/models/{name}_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            model.save_model(save_path)
            
        return models
        
    def evaluate_models(self, models: Dict[str, object], 
                       test_data: pd.DataFrame) -> Dict[str, Dict]:
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
            
        # Generate comparison visualizations
        self.visualizer.plot_metrics_comparison(
            pd.DataFrame({name: res['metrics'] for name, res in results.items()})
        )
        
        return results
        
    def generate_forecasts(self, models: Dict[str, object], 
                          data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Generate future forecasts."""
        forecast_horizon = self.config['output']['forecast']['horizon']
        forecasts = {}
        
        for name, model in models.items():
            predictions = model.predict(data, forecast_horizon)
            forecasts[name] = predictions
            
            # Save predictions
            save_path = f"Scripts/climate_prediction/outputs/predictions/{name}_forecast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            predictions.to_csv(save_path)
            
        # Generate forecast visualizations
        self.visualizer.plot_forecast_horizon(
            data[self.target_variable],
            forecasts,
            forecast_start=data.index[-1]
        )
        
        return forecasts
        
    def run_pipeline(self) -> Dict:
        """Execute complete modeling pipeline."""
        try:
            # Process data
            data_results = self.data_pipeline.run_pipeline()
            train_path = data_results['paths']['train']
            test_path = data_results['paths']['test']
            
            # Load processed data
            train_data, test_data = self.load_data(train_path, test_path)
            
            # Train models
            models = self.train_models(train_data, test_data)
            
            # Evaluate models
            evaluation_results = self.evaluate_models(models, test_data)
            
            # Generate forecasts
            forecasts = self.generate_forecasts(models, train_data)
            
            return {
                'status': 'success',
                'models': models,
                'evaluation': evaluation_results,
                'forecasts': forecasts
            }
            
        except Exception as e:
            self.logger.log_error(f"Pipeline failed: {str(e)}")
            raise

if __name__ == "__main__":
    pipeline = ModelPipeline()
    results = pipeline.run_pipeline()
    print("Pipeline completed successfully")