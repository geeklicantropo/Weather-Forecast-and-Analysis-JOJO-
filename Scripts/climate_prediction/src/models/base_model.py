# src/models/base_model.py
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json
from ..utils.gpu_manager import gpu_manager

class BaseModel(ABC):
    def __init__(self, target_variable, logger):
        self.target_variable = target_variable
        self.logger = logger
        self.model = None
        self.metrics = {}
        self.feature_importance = None
        self.device = gpu_manager.get_device()
    
    @abstractmethod
    def preprocess_data(self, df):
        """Preprocess data for model training"""
        pass
    
    @abstractmethod
    def train(self, train_data, validation_data=None):
        """Train the model"""
        pass
    
    @abstractmethod
    def predict(self, data, forecast_horizon=None):
        """Generate predictions"""
        pass
    
    def evaluate(self, true_values, predictions):
        """Evaluate model performance"""
        self.metrics = {
            'rmse': np.sqrt(mean_squared_error(true_values, predictions)),
            'mae': mean_absolute_error(true_values, predictions),
            'r2': r2_score(true_values, predictions)
        }
        return self.metrics
    
    def save_model(self, path):
        """Save model and metadata"""
        try:
            model_info = {
                'model_type': self.__class__.__name__,
                'target_variable': self.target_variable,
                'metrics': self.metrics,
                'feature_importance': self.feature_importance,
                'timestamp': pd.Timestamp.now().isoformat()
            }
            
            # Save metadata
            with open(f"{path}/model_info.json", 'w') as f:
                json.dump(model_info, f, indent=4)
            
            # Save model-specific data
            self._save_model_data(path)
            
        except Exception as e:
            self.logger.log_error(f"Error saving model: {str(e)}")
            raise
    
    def load_model(self, path):
        """Load model and metadata"""
        try:
            # Load metadata
            with open(f"{path}/model_info.json", 'r') as f:
                model_info = json.load(f)
            
            self.metrics = model_info['metrics']
            self.feature_importance = model_info['feature_importance']
            
            # Load model-specific data
            self._load_model_data(path)
            
        except Exception as e:
            self.logger.log_error(f"Error loading model: {str(e)}")
            raise
    
    @abstractmethod
    def _save_model_data(self, path):
        """Save model-specific data"""
        pass
    
    @abstractmethod
    def _load_model_data(self, path):
        """Load model-specific data"""
        pass
    
    def get_feature_importance(self):
        """Get feature importance if available"""
        return self.feature_importance
    
    def get_metrics(self):
        """Get model metrics"""
        return self.metrics