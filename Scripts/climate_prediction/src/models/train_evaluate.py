# src/models/train_evaluate.py
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from typing import Dict, List, Tuple

from src.models.lstm_model import LSTMModel
from src.models.sarima_model import SARIMAModel
from src.models.tft_model import TFTModel
from src.models.model_evaluator import ModelEvaluator
from src.utils.logger import ProgressLogger
from src.utils.config_manager import ConfigManager
from src.visualization.visualization_manager import VisualizationManager
from src.utils.gpu_manager import gpu_manager

class ModelTrainEvaluate:
    def __init__(self, config_path: str = "config/model_config.yaml"):
        self.logger = ProgressLogger(name="ModelTrainEvaluate")
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.get_config()
        self.visualizer = VisualizationManager(self.logger)
        self.device = gpu_manager.get_device()
        self._setup_directories()
        
    def _setup_directories(self):
        """Create necessary output directories."""
        directories = [
            'outputs/models',
            'outputs/predictions',
            'outputs/plots',
            'outputs/metrics'
        ]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def prepare_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data into train, validation, and test sets."""
        data = data.sort_index()
        
        # Calculate split points based on config
        total_samples = len(data)
        train_size = int(total_samples * 0.7)
        val_size = int(total_samples * 0.15)
        
        # Split data
        train_data = data[:train_size]
        val_data = data[train_size:train_size + val_size]
        test_data = data[train_size + val_size:]
        
        self.logger.log_info(f"Data split - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
        return train_data, val_data, test_data
    
    def train_models(self, train_data: pd.DataFrame, val_data: pd.DataFrame) -> Dict:
        """Train all models with GPU support and memory optimization."""
        models = {}
        training_histories = {}
        
        try:
            # Configure models based on available GPU memory
            batch_size = gpu_manager.get_optimal_batch_size()
            memory_info = gpu_manager.get_memory_info()
            
            model_configs = {
                'lstm': (LSTMModel, {'batch_size': batch_size}),
                'sarima': (SARIMAModel, {}),
                'tft': (TFTModel, {'batch_size': batch_size})
            }
            
            for name, (model_class, params) in model_configs.items():
                if self.config['models'][name].get('enabled', True):
                    self.logger.log_info(f"Training {name.upper()} model...")
                    
                    try:
                        # Initialize model with GPU support
                        model = model_class(
                            target_variable=self.config['preprocessing']['target_variable'],
                            logger=self.logger,
                            **params
                        )
                        
                        # Clear GPU memory before training
                        if self.device.type == 'cuda':
                            gpu_manager.clear_memory()
                        
                        # Train model
                        with gpu_manager.memory_monitor():
                            history = model.train(train_data, val_data)
                            models[name] = model
                            training_histories[name] = history
                        
                        # Save model checkpoint
                        model_path = f"outputs/models/{name}_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                        model.save_model(model_path)
                        
                    except Exception as e:
                        self.logger.log_error(f"Error training {name} model: {str(e)}")
                        continue
            
            return models, training_histories
            
        except Exception as e:
            self.logger.log_error(f"Model training failed: {str(e)}")
            raise
    
    def evaluate_models(self, models: Dict, test_data: pd.DataFrame) -> Dict:
        """Evaluate all trained models with enhanced error handling."""
        evaluator = ModelEvaluator(self.logger)
        evaluation_results = {}
        
        try:
            for name, model in models.items():
                self.logger.log_info(f"Evaluating {name.upper()} model...")
                
                try:
                    # Generate predictions with GPU optimization
                    with gpu_manager.memory_monitor():
                        predictions = model.predict(test_data)
                    
                    # Calculate metrics
                    metrics = evaluator.evaluate_model(model, test_data, predictions)
                    evaluation_results[name] = {
                        'metrics': metrics,
                        'predictions': predictions
                    }
                    
                    # Generate evaluation plots
                    self.visualizer.plot_predictions(
                        test_data[self.config['preprocessing']['target_variable']],
                        {name: predictions}
                    )
                    
                    # Save predictions
                    predictions_path = f"outputs/predictions/{name}_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                    predictions.to_csv(predictions_path)
                    
                except Exception as e:
                    self.logger.log_error(f"Error evaluating {name} model: {str(e)}")
                    continue
            
            # Generate comparison visualizations
            if evaluation_results:
                self.visualizer.plot_metrics_comparison(
                    pd.DataFrame({name: results['metrics'] 
                                for name, results in evaluation_results.items()})
                )
            
            return evaluation_results
            
        except Exception as e:
            self.logger.log_error(f"Model evaluation failed: {str(e)}")
            raise
    
    def generate_future_predictions(self, models: Dict, data: pd.DataFrame, horizon: int = None) -> Dict:
        """Generate future predictions with uncertainty estimation."""
        horizon = horizon or int(self.config['output']['forecast']['horizon'])
        future_predictions = {}
        
        try:
            for name, model in models.items():
                self.logger.log_info(f"Generating predictions for {name.upper()}...")
                
                try:
                    # Generate predictions with GPU optimization
                    with gpu_manager.memory_monitor():
                        predictions = model.predict(data, horizon)
                        future_predictions[name] = predictions
                    
                    # Save predictions
                    predictions_path = f"outputs/predictions/{name}_future_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                    predictions.to_csv(predictions_path)
                    
                except Exception as e:
                    self.logger.log_error(f"Error generating predictions for {name}: {str(e)}")
                    continue
            
            # Generate forecast visualizations
            if future_predictions:
                self.visualizer.plot_forecast_horizon(
                    data[self.config['preprocessing']['target_variable']],
                    future_predictions,
                    forecast_start=data.index[-1]
                )
            
            return future_predictions
            
        except Exception as e:
            self.logger.log_error(f"Future prediction generation failed: {str(e)}")
            raise
    
    def run_pipeline(self, data: pd.DataFrame) -> Dict:
        """Execute complete training and evaluation pipeline."""
        try:
            # Prepare data
            train_data, val_data, test_data = self.prepare_data(data)
            
            # Train models
            models, training_histories = self.train_models(train_data, val_data)
            
            # Evaluate models
            evaluation_results = self.evaluate_models(models, test_data)
            
            # Generate future predictions
            future_predictions = self.generate_future_predictions(models, data)
            
            return {
                'models': models,
                'training_histories': training_histories,
                'evaluation_results': evaluation_results,
                'future_predictions': future_predictions
            }
            
        except Exception as e:
            self.logger.log_error(f"Pipeline execution failed: {str(e)}")
            raise

if __name__ == "__main__":
    # Example usage
    config_path = "config/model_config.yaml"
    pipeline = ModelTrainEvaluate(config_path)
    
    # Load your data
    data = pd.read_csv("outputs/data/processed_data.csv.gz")
    
    # Run pipeline
    results = pipeline.run_pipeline(data)