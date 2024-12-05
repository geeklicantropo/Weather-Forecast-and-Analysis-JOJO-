from .base_model import BaseModelAdapter
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pandas as pd
import numpy as np

class SARIMAAdapter(BaseModelAdapter):
    def __init__(self, config):
        super().__init__(config)
        self.order = config['sarima']['order']
        self.seasonal_order = config['sarima']['seasonal_order']
    
    def preprocess(self, data):
        """Prepare data for SARIMA."""
        return data[self.target_variable]
    
    def train(self, train_data, val_data):
        """Train SARIMA model."""
        train_series = self.preprocess(train_data)
        
        self.model = SARIMAX(
            train_series,
            order=self.order,
            seasonal_order=self.seasonal_order
        )
        
        fit_result = self.model.fit(disp=False)
        self.model = fit_result
        return fit_result
    
    def predict(self, data, prediction_steps):
        """Generate predictions."""
        return self.model.forecast(steps=prediction_steps)