# src/models/model_evaluator.py
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import List, Dict, Any
import joblib
import json
from concurrent.futures import ProcessPoolExecutor

class ModelEvaluator:
    def __init__(self, logger):
        self.logger = logger
        self.results = {}
        self.metrics = {}
        
    def evaluate_model(self, model, test_data, forecast_horizon):
        try:
            predictions = model.predict(test_data, forecast_horizon)
            true_values = test_data[model.target_variable].values[-len(predictions):]
            
            # Calculate standard metrics
            metrics = {
                'rmse': np.sqrt(mean_squared_error(true_values, predictions['forecast'])),
                'mae': mean_absolute_error(true_values, predictions['forecast']),
                'r2': r2_score(true_values, predictions['forecast']),
                'mape': np.mean(np.abs((true_values - predictions['forecast']) / true_values)) * 100
            }
            
            # Calculate interval scores
            coverage = np.mean((true_values >= predictions['lower_bound']) & 
                             (true_values <= predictions['upper_bound'])) * 100
            
            interval_width = np.mean(predictions['upper_bound'] - predictions['lower_bound'])
            
            metrics.update({
                'prediction_interval_coverage': coverage,
                'mean_interval_width': interval_width
            })
            
            # Store results
            self.results[model.__class__.__name__] = {
                'predictions': predictions,
                'true_values': true_values
            }
            
            self.metrics[model.__class__.__name__] = metrics
            return metrics
            
        except Exception as e:
            self.logger.log_error(f"Evaluation error for {model.__class__.__name__}: {str(e)}")
            raise

    def compare_models(self, models: List[Any], test_data, forecast_horizon):
        try:
            with ProcessPoolExecutor() as executor:
                futures = [
                    executor.submit(self.evaluate_model, model, test_data, forecast_horizon)
                    for model in models
                ]
                
                for future in futures:
                    future.result()
                    
            comparison = pd.DataFrame(self.metrics).T
            comparison = comparison.round(4)
            
            best_model = comparison['rmse'].idxmin()
            
            # Save comparison results
            self._save_comparison_results(comparison)
            
            return {
                'comparison': comparison,
                'best_model': best_model,
                'detailed_results': self.results
            }
            
        except Exception as e:
            self.logger.log_error(f"Model comparison error: {str(e)}")
            raise
            
    def _save_comparison_results(self, comparison):
        try:
            output_path = 'outputs/metadata/model_comparison.json'
            results = {
                'comparison_metrics': comparison.to_dict(),
                'timestamp': pd.Timestamp.now().isoformat(),
                'best_model': comparison['rmse'].idxmin()
            }
            
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=4)
                
        except Exception as e:
            self.logger.log_error(f"Error saving comparison results: {str(e)}")
            raise

    def generate_forecast_ensemble(self, models, test_data, forecast_horizon):
        try:
            all_predictions = []
            weights = []
            
            for model in models:
                predictions = model.predict(test_data, forecast_horizon)
                all_predictions.append(predictions['forecast'])
                
                # Calculate weights based on inverse RMSE
                rmse = self.metrics[model.__class__.__name__]['rmse']
                weights.append(1 / rmse)
            
            # Normalize weights
            weights = np.array(weights) / sum(weights)
            
            # Calculate weighted ensemble prediction
            ensemble_prediction = np.zeros_like(all_predictions[0])
            for pred, weight in zip(all_predictions, weights):
                ensemble_prediction += pred * weight
                
            return ensemble_prediction
            
        except Exception as e:
            self.logger.log_error(f"Ensemble generation error: {str(e)}")
            raise