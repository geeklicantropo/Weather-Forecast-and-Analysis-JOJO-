# src/models/model_evaluator.py
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

class ModelEvaluator:
    def __init__(self, models, train_data, test_data):
        self.models = models
        self.train_data = train_data
        self.test_data = test_data
        self.results = {}
        
    def evaluate_all_models(self):
        """Train and evaluate all models."""
        for model in self.models:
            print(f"\nEvaluating {model.name}...")
            
            # Train model
            model.train(self.train_data)
            
            # Make predictions
            predictions = model.predict(self.test_data)
            
            # Calculate metrics
            metrics = self._calculate_metrics(
                self.test_data[model.target_variable],
                predictions
            )
            
            self.results[model.name] = {
                'predictions': predictions,
                'metrics': metrics
            }
    
    def _calculate_metrics(self, y_true, y_pred):
        """Calculate comprehensive evaluation metrics."""
        return {
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        }
    
    def compare_models(self):
        """Create comparison dataframe of all model metrics."""
        metrics_df = pd.DataFrame()
        
        for model_name, result in self.results.items():
            metrics_df[model_name] = pd.Series(result['metrics'])
        
        return metrics_df
    
    def plot_predictions(self, save_path=None):
        """Plot actual vs predicted values for all models."""
        plt.figure(figsize=(15, 10))
        
        # Plot actual values
        plt.plot(
            self.test_data.index,
            self.test_data[self.models[0].target_variable],
            label='Actual',
            color='black'
        )
        
        # Plot predictions for each model
        colors = sns.color_palette("husl", len(self.models))
        for (model_name, result), color in zip(self.results.items(), colors):
            plt.plot(
                self.test_data.index,
                result['predictions'],
                label=f'{model_name} Predictions',
                color=color,
                alpha=0.7
            )
        
        plt.title('Model Predictions Comparison')
        plt.xlabel('Date')
        plt.ylabel('Temperature (°C)')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()