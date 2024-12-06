import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from typing import List, Dict, Any, Tuple
import joblib
import json
from datetime import datetime
from ..utils.gpu_manager import gpu_manager

class ModelEvaluator:
    def __init__(self, logger):
        self.logger = logger
        self.device = gpu_manager.get_device()
        self.results = {}
        self.metrics = {}
        
    def evaluate_model(self, model, test_data: pd.DataFrame, 
                      predictions: pd.DataFrame) -> Dict[str, float]:
        try:
            true_values = test_data[model.target_variable].values
            pred_values = predictions['forecast'].values
            
            # Calculate standard metrics
            metrics = {
                'rmse': float(np.sqrt(mean_squared_error(true_values, pred_values))),
                'mae': float(mean_absolute_error(true_values, pred_values)),
                'r2': float(r2_score(true_values, pred_values)),
                'mape': float(mean_absolute_percentage_error(true_values, pred_values) * 100)
            }
            
            # Calculate prediction interval coverage
            if 'lower_bound' in predictions and 'upper_bound' in predictions:
                coverage = np.mean(
                    (true_values >= predictions['lower_bound']) & 
                    (true_values <= predictions['upper_bound'])
                ) * 100
                interval_width = np.mean(
                    predictions['upper_bound'] - predictions['lower_bound']
                )
                
                metrics.update({
                    'prediction_interval_coverage': float(coverage),
                    'mean_interval_width': float(interval_width)
                })
            
            # Store results
            self.results[model.__class__.__name__] = {
                'predictions': predictions,
                'true_values': true_values,
                'metrics': metrics
            }
            
            return metrics
            
        except Exception as e:
            self.logger.log_error(f"Evaluation error: {str(e)}")
            raise
            
    def compare_models(self, models: Dict[str, Any], test_data: pd.DataFrame, 
                      predictions: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        try:
            comparison_metrics = {}
            
            for name, model in models.items():
                metrics = self.evaluate_model(model, test_data, predictions[name])
                comparison_metrics[name] = metrics
            
            comparison_df = pd.DataFrame(comparison_metrics).round(4)
            
            # Save comparison results
            self._save_comparison_results(comparison_df)
            
            # Determine best model
            self.best_model = comparison_df['rmse'].idxmin()
            
            return comparison_df
            
        except Exception as e:
            self.logger.log_error(f"Model comparison error: {str(e)}")
            raise
            
    def _save_comparison_results(self, comparison: pd.DataFrame):
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results = {
                'comparison_metrics': comparison.to_dict(),
                'timestamp': datetime.now().isoformat(),
                'best_model': self.best_model
            }
            
            output_path = f'outputs/metrics/model_comparison_{timestamp}.json'
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=4)
                
        except Exception as e:
            self.logger.log_error(f"Error saving comparison results: {str(e)}")
            raise
            
    def calculate_feature_importance(self, models: Dict[str, Any], 
                                   data: pd.DataFrame) -> Dict[str, pd.Series]:
        try:
            feature_importance = {}
            
            for name, model in models.items():
                if hasattr(model, 'get_feature_importance'):
                    importance = model.get_feature_importance()
                    if importance is not None:
                        feature_importance[name] = pd.Series(importance)
            
            return feature_importance
            
        except Exception as e:
            self.logger.log_error(f"Error calculating feature importance: {str(e)}")
            raise
            
    def evaluate_residuals(self, models: Dict[str, Any], test_data: pd.DataFrame,
                          predictions: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
        try:
            residual_analysis = {}
            
            for name, model in models.items():
                true_values = test_data[model.target_variable].values
                pred_values = predictions[name]['forecast'].values
                residuals = true_values - pred_values
                
                analysis = {
                    'mean': float(np.mean(residuals)),
                    'std': float(np.std(residuals)),
                    'skewness': float(pd.Series(residuals).skew()),
                    'kurtosis': float(pd.Series(residuals).kurtosis()),
                    'normality_test': float(pd.Series(residuals).describe())
                }
                
                residual_analysis[name] = analysis
            
            return residual_analysis
            
        except Exception as e:
            self.logger.log_error(f"Error analyzing residuals: {str(e)}")
            raise
            
    def generate_forecast_ensemble(self, models: Dict[str, Any], test_data: pd.DataFrame,
                                 predictions: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        try:
            weights = []
            forecasts = []
            
            # Calculate weights based on inverse RMSE
            for name, model in models.items():
                metrics = self.evaluate_model(model, test_data, predictions[name])
                weights.append(1 / metrics['rmse'])
                forecasts.append(predictions[name]['forecast'])
            
            # Normalize weights
            weights = np.array(weights) / sum(weights)
            
            # Calculate weighted ensemble prediction
            ensemble_forecast = np.zeros_like(forecasts[0])
            for forecast, weight in zip(forecasts, weights):
                ensemble_forecast += forecast * weight
            
            # Calculate ensemble prediction intervals
            lower_bounds = np.array([pred['lower_bound'] for pred in predictions.values()])
            upper_bounds = np.array([pred['upper_bound'] for pred in predictions.values()])
            
            ensemble_df = pd.DataFrame({
                'forecast': ensemble_forecast,
                'lower_bound': np.percentile(lower_bounds, 2.5, axis=0),
                'upper_bound': np.percentile(upper_bounds, 97.5, axis=0)
            }, index=predictions[list(predictions.keys())[0]].index)
            
            return ensemble_df
            
        except Exception as e:
            self.logger.log_error(f"Error generating ensemble forecast: {str(e)}")
            raise
            
    def save_evaluation_results(self, output_dir: str = 'outputs/metrics'):
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save detailed results
            results_path = f'{output_dir}/evaluation_results_{timestamp}.json'
            with open(results_path, 'w') as f:
                json.dump(self.results, f, indent=4, default=str)
            
        except Exception as e:
            self.logger.log_error(f"Error saving evaluation results: {str(e)}")
            raise