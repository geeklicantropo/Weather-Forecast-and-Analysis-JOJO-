# src/models/model_manager.py
import os
import json
from datetime import datetime
import torch
import joblib
import pandas as pd
from typing import Dict, Optional, Any

class ModelManager:
    def __init__(self, base_path: str = "Scripts/climate_prediction/outputs"):
        self.base_path = base_path
        self.models_path = os.path.join(base_path, "models")
        self.predictions_path = os.path.join(base_path, "predictions")
        self.metadata_path = os.path.join(base_path, "metadata")
        self._ensure_directories()

    def _ensure_directories(self):
        """Create necessary directories if they don't exist."""
        for path in [self.models_path, self.predictions_path, self.metadata_path]:
            os.makedirs(path, exist_ok=True)

    def get_latest_model(self, model_name: str) -> Optional[str]:
        """Get path to latest model of specified type."""
        model_files = [f for f in os.listdir(self.models_path) 
                      if f.startswith(f"{model_name}_model_")]
        return os.path.join(self.models_path, sorted(model_files)[-1]) if model_files else None

    def save_model(self, model: Any, model_name: str) -> str:
        """Save model with timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(self.models_path, f"{model_name}_model_{timestamp}")
        
        # Save model using its save_model method
        model.save_model(save_path)
        
        # Save metadata
        metadata = {
            'timestamp': timestamp,
            'model_type': model_name,
            'model_class': model.__class__.__name__
        }
        
        with open(os.path.join(self.metadata_path, f"{model_name}_metadata_{timestamp}.json"), 'w') as f:
            json.dump(metadata, f, indent=4)
            
        return save_path

    def load_model(self, model_class: Any, model_name: str, timestamp: Optional[str] = None) -> Any:
        """Load model of specified type."""
        if timestamp:
            model_path = os.path.join(self.models_path, f"{model_name}_model_{timestamp}")
        else:
            model_path = self.get_latest_model(model_name)
            
        if not model_path:
            raise FileNotFoundError(f"No saved model found for {model_name}")
            
        # Create new model instance and load saved state
        model = model_class()
        model.load_model(model_path)
        return model

    def save_predictions(self, predictions: pd.DataFrame, model_name: str) -> str:
        """Save model predictions with timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(self.predictions_path, f"{model_name}_predictions_{timestamp}.csv")
        predictions.to_csv(save_path)
        return save_path

    def load_predictions(self, model_name: str, timestamp: Optional[str] = None) -> pd.DataFrame:
        """Load predictions for specified model."""
        if timestamp:
            pred_path = os.path.join(self.predictions_path, f"{model_name}_predictions_{timestamp}.csv")
        else:
            pred_files = [f for f in os.listdir(self.predictions_path) 
                         if f.startswith(f"{model_name}_predictions_")]
            if not pred_files:
                raise FileNotFoundError(f"No saved predictions found for {model_name}")
            pred_path = os.path.join(self.predictions_path, sorted(pred_files)[-1])
            
        return pd.read_csv(pred_path)

    def model_exists(self, model_name: str) -> bool:
        """Check if model exists."""
        return bool(self.get_latest_model(model_name))

    def get_model_metadata(self, model_name: str, timestamp: Optional[str] = None) -> Dict:
        """Get metadata for specified model."""
        if timestamp:
            metadata_path = os.path.join(self.metadata_path, f"{model_name}_metadata_{timestamp}.json")
        else:
            metadata_files = [f for f in os.listdir(self.metadata_path) 
                            if f.startswith(f"{model_name}_metadata_")]
            if not metadata_files:
                raise FileNotFoundError(f"No metadata found for {model_name}")
            metadata_path = os.path.join(self.metadata_path, sorted(metadata_files)[-1])
            
        with open(metadata_path, 'r') as f:
            return json.load(f)