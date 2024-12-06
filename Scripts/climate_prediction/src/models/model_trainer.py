# src/models/model_trainer.py
import os
import numpy as np
import pandas as pd
import torch
from datetime import datetime, timedelta
import json
import yaml
import logging
from tqdm import tqdm
from typing import Dict, Tuple, Any, Optional
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from src.models.lstm_adapter import LSTMAdapter
from src.models.sarima_adapter import SARIMAAdapter
from src.models.tft_adapter import TFTAdapter

class ModelTrainer:
    def __init__(self, config_path: str = 'config/model_config.yaml', target_variable: str = "TEMPERATURA DO AR - BULBO SECO HORARIA °C"):
        """
        Initialize ModelTrainer with configuration and setup.
        
        Args:
            config_path: Path to model configuration file
            target_variable: Name of the target variable for prediction
        """
        self.target_variable = target_variable
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = {}
        self.results = {}
        self.load_config(config_path)
        self.setup_logging()
        
    def setup_logging(self) -> None:
        """Setup logging configuration for model training."""
        log_dir = 'outputs/logs'
        os.makedirs(log_dir, exist_ok=True)
        
        log_path = f'{log_dir}/model_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_path),
                logging.StreamHandler()
            ]
        )
    
    def load_config(self, config_path: str) -> None:
        """
        Load or create model configuration.
        
        Args:
            config_path: Path to configuration file
        """
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        if not os.path.exists(config_path):
            self._create_default_config(config_path)
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        logging.info(f"Loaded configuration from {config_path}")
    
    def _create_default_config(self, config_path: str) -> None:
        """
        Create default model configuration.
        
        Args:
            config_path: Path to save configuration
        """
        default_config = {
            'lstm': {
                'enabled': True,
                'sequence_length': 30,
                'hidden_size': 50,
                'num_layers': 2,
                'batch_size': 32,
                'epochs': 50,
                'learning_rate': 0.001,
                'dropout': 0.2
            },
            'sarima': {
                'enabled': True,
                'order': [1, 1, 1],
                'seasonal_order': [1, 1, 1, 12],
                'enforce_stationarity': True,
                'enforce_invertibility': True
            },
            'tft': {
                'enabled': True,
                'max_encoder_length': 30,
                'max_prediction_length': 10,
                'batch_size': 64,
                'epochs': 30,
                'learning_rate': 0.001,
                'hidden_size': 16,
                'attention_head_size': 4,
                'dropout': 0.1,
                'hidden_continuous_size': 8
            },
            'training': {
                'early_stopping_patience': 5,
                'min_delta': 0.001,
                'validation_frequency': 1
            }
        }
        
        with open(config_path, 'w') as f:
            yaml.dump(default_config, f, indent=4)
        
        logging.info(f"Created default configuration at {config_path}")
    
    def initialize_models(self) -> None:
        """Initialize all enabled models with configurations."""
        logging.info("Initializing models...")
        
        with tqdm(total=3, desc="Initializing Models") as pbar:
            if self.config['lstm']['enabled']:
                self.models['lstm'] = LSTMAdapter(self.config['lstm'])
                pbar.update(1)
            
            if self.config['sarima']['enabled']:
                self.models['sarima'] = SARIMAAdapter(self.config['sarima'])
                pbar.update(1)
            
            if self.config['tft']['enabled']:
                self.models['tft'] = TFTAdapter(self.config['tft'])
                pbar.update(1)
    
    def train_models(self, train_data: pd.DataFrame, val_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Train all initialized models.
        
        Args:
            train_data: Training dataset
            val_data: Validation dataset
        
        Returns:
            Dictionary containing training histories for each model
        """
        training_histories = {}
        
        for name, model in self.models.items():
            logging.info(f"Training {name.upper()} model...")
            try:
                with tqdm(desc=f"Training {name.upper()}", unit="epoch") as pbar:
                    history = model.train(
                        train_data,
                        val_data,
                        progress_callback=lambda x: pbar.update(1)
                    )
                    training_histories[name] = history
                    self._save_model(model, name)
                    
                logging.info(f"Completed training {name.upper()} model")
                
            except Exception as e:
                logging.error(f"Error training {name.upper()} model: {str(e)}")
                continue
        
        return training_histories
    
    def evaluate_models(self, test_data: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """
        Evaluate all models on test data.
        
        Args:
            test_data: Test dataset
        
        Returns:
            Dictionary containing evaluation metrics for each model
        """
        evaluation_results = {}
        
        for name, model in self.models.items():
            logging.info(f"Evaluating {name.upper()} model...")
            try:
                predictions = model.predict(test_data, len(test_data))
                metrics = self._calculate_metrics(
                    test_data[self.target_variable].values,
                    predictions.flatten()
                )
                evaluation_results[name] = metrics
                
                logging.info(f"Metrics for {name.upper()}: {metrics}")
                
            except Exception as e:
                logging.error(f"Error evaluating {name.upper()} model: {str(e)}")
                continue
        
        self._save_evaluation_results(evaluation_results)
        return evaluation_results
    
    def generate_future_predictions(self, data: pd.DataFrame, years: int = 10) -> Dict[str, np.ndarray]:
        """
        Generate predictions for specified number of years.
        
        Args:
            data: Input data for prediction
            years: Number of years to predict
        
        Returns:
            Dictionary containing predictions from each model
        """
        prediction_days = years * 365
        future_predictions = {}
        
        for name, model in self.models.items():
            logging.info(f"Generating {years}-year predictions for {name.upper()}...")
            try:
                with tqdm(desc=f"Predicting with {name.upper()}", total=1) as pbar:
                    predictions = model.predict(data, prediction_days)
                    future_predictions[name] = predictions
                    self._save_predictions(predictions, name, years)
                    pbar.update(1)
                
            except Exception as e:
                logging.error(f"Error generating predictions for {name.upper()}: {str(e)}")
                continue
        
        return future_predictions
    
    def _calculate_metrics(self, true_values: np.ndarray, predictions: np.ndarray) -> Dict[str, float]:
        """
        Calculate evaluation metrics.
        
        Args:
            true_values: Actual values
            predictions: Predicted values
        
        Returns:
            Dictionary of metric names and values
        """
        return {
            'rmse': np.sqrt(mean_squared_error(true_values, predictions)),
            'mae': mean_absolute_error(true_values, predictions),
            'r2': r2_score(true_values, predictions),
            'mape': np.mean(np.abs((true_values - predictions) / true_values)) * 100
        }
    
    def _save_model(self, model: Any, name: str) -> None:
        """
        Save trained model.
        
        Args:
            model: Model to save
            name: Model name
        """
        save_dir = f'outputs/models/{name}'
        os.makedirs(save_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f'{save_dir}/model_{timestamp}.pt'
        model.save_model(save_path)
        logging.info(f"Saved {name.upper()} model to {save_path}")
    
    def _save_evaluation_results(self, results: Dict[str, Dict[str, float]]) -> None:
        """
        Save evaluation metrics.
        
        Args:
            results: Evaluation results to save
        """
        save_dir = 'outputs/metadata'
        os.makedirs(save_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f'{save_dir}/evaluation_results_{timestamp}.json'
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=4)
        logging.info(f"Saved evaluation results to {save_path}")
    
    def _save_predictions(self, predictions: np.ndarray, model_name: str, years: int) -> None:
        """
        Save model predictions.
        
        Args:
            predictions: Predictions to save
            model_name: Name of the model
            years: Number of years predicted
        """
        save_dir = 'outputs/predictions'
        os.makedirs(save_dir, exist_ok=True)
        
        start_date = datetime.now()
        dates = pd.date_range(start=start_date, periods=len(predictions), freq='D')
        
        df_predictions = pd.DataFrame({
            'date': dates,
            'prediction': predictions.flatten()
        })
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f'{save_dir}/{model_name}_{years}yr_predictions_{timestamp}.csv'
        df_predictions.to_csv(save_path, index=False)
        logging.info(f"Saved {model_name.upper()} predictions to {save_path}")
    
    def select_best_model(self, evaluation_results: Dict[str, Dict[str, float]]) -> str:
        """
        Select the best performing model based on RMSE.
        
        Args:
            evaluation_results: Dictionary of model evaluation results
        
        Returns:
            Name of the best performing model
        """
        best_model = min(evaluation_results.items(), key=lambda x: x[1]['rmse'])
        logging.info(f"Selected best model: {best_model[0].upper()} with RMSE: {best_model[1]['rmse']}")
        return best_model[0]

if __name__ == "__main__":
    # Example usage
    trainer = ModelTrainer()
    trainer.initialize_models()