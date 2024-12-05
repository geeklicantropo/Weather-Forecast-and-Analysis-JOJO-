# src/models/base_model.py
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
from datetime import datetime, timedelta

class BaseModelAdapter(ABC):
    def __init__(self, config, target_variable="TEMPERATURA DO AR - BULBO SECO HORARIA °C"):
        self.config = config
        self.target_variable = target_variable
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.scaler = None
        self.metrics = {}
    
    @abstractmethod
    def preprocess(self, data):
        """Preprocess data for model training."""
        pass
    
    @abstractmethod
    def train(self, train_data, val_data):
        """Train the model."""
        pass
    
    @abstractmethod
    def predict(self, data, prediction_steps):
        """Make predictions."""
        pass
    
    def evaluate(self, true_values, predictions):
        """Calculate evaluation metrics."""
        metrics = {
            'rmse': np.sqrt(mean_squared_error(true_values, predictions)),
            'mae': mean_absolute_error(true_values, predictions),
            'r2': r2_score(true_values, predictions),
            'mape': np.mean(np.abs((true_values - predictions) / true_values)) * 100
        }
        self.metrics.update(metrics)
        return metrics
    
    def save_model(self, path):
        """Save model and metadata."""
        save_dict = {
            'model_state': self.model.state_dict() if hasattr(self.model, 'state_dict') else self.model,
            'scaler': self.scaler,
            'metrics': self.metrics,
            'config': self.config,
            'timestamp': datetime.now().isoformat()
        }
        torch.save(save_dict, path)
    
    def load_model(self, path):
        """Load model and metadata."""
        checkpoint = torch.load(path, map_location=self.device)
        if hasattr(self.model, 'load_state_dict'):
            self.model.load_state_dict(checkpoint['model_state'])
        else:
            self.model = checkpoint['model_state']
        self.scaler = checkpoint['scaler']
        self.metrics = checkpoint['metrics']
        self.config = checkpoint['config']