# src/models/pipeline.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import os
from datetime import datetime
import torch
from sklearn.model_selection import TimeSeriesSplit

from .lstm_model import LSTMModel
from .sarima_model import SARIMAModel
from .tft_model import TFTModel
from .model_evaluator import ModelEvaluator
from ..visualization.visualization_manager import VisualizationManager
from ..utils.gpu_manager import gpu_manager

class ModelPipeline:
    def __init__(self, target_variable: str, logger, output_dir: str = "outputs",
                forecast_horizon: int = 3650):  # 10 years
        self.target_variable = target_variable
        self.logger = logger
        self.output_dir = output_dir
        self.forecast_horizon = forecast_horizon
        self.models = {}
        self.evaluator = ModelEvaluator(logger)
        self.visualizer = VisualizationManager(logger)
        self._setup_directories()
        
    def _setup_directories(self):
        """Create necessary directories for outputs"""
        dirs = ['models', 'plots', 'metrics', 'forecasts']
        for dir_name in dirs:
            os.makedirs(f"{self.output_dir}/{dir_name}", exist_ok=True)
            
    def prepare_data(self, df: pd.DataFrame, test_size: float = 0.2, 
                    validation_size: float = 0.1) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Prepare train, validation, and test sets with proper time ordering"""
        try:
            df = df.sort_index()
            val_cutoff = int(len(df) * (1 - test_size - validation_size))
            test_cutoff = int(len(df) * (1 - test_size))
            
            train_data = df.iloc[:val_cutoff]
            validation_data = df.iloc[val_cutoff:test_cutoff]
            test_data = df.iloc[test_cutoff:]
            
            self._validate_data_split(train_data, validation_data, test_data)
            return train_data, validation_data, test_data
            
        except Exception as e:
            self.logger.log_error(f"Error in data preparation: {str(e)}")
            raise
            
    def _validate_data_split(self, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame):
        """Validate data splits for time continuity and feature consistency"""
        # Check time continuity
        assert train.index.max() < val.index.min(), "Train-validation overlap detected"
        assert val.index.max() < test.index.min(), "Validation-test overlap detected"
        
        # Check feature consistency
        for df in [train, val, test]:
            assert self.target_variable in df.columns, f"Target variable missing in split"
            assert not df[self.target_variable].isna().any(), f"NaN values in target variable"
        
    def train_models(self, train_data: pd.DataFrame, validation_data: pd.DataFrame):
        """Train all models with cross-validation and GPU support"""
        model_configs = {
            'lstm': {'class': LSTMModel, 'params': {'sequence_length': 30, 'hidden_size': 50}},
            'sarima': {'class': SARIMAModel, 'params': {}},
            'tft': {'class': TFTModel, 'params': {'max_prediction_length': self.forecast_horizon}}
        }
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        for name, config in model_configs.items():
            try:
                self.logger.log_info(f"Training {name} model...")
                model = config['class'](self.target_variable, self.logger, **config['params'])
                
                # Cross-validation
                cv_scores = []
                for train_idx, val_idx in tscv.split(train_data):
                    cv_train = train_data.iloc[train_idx]
                    cv_val = train_data.iloc[val_idx]
                    
                    processed_train = model.preprocess_data(cv_train)
                    processed_val = model.preprocess_data(cv_val)
                    
                    model.train(processed_train, processed_val)
                    cv_scores.append(model.evaluate(cv_val[self.target_variable], 
                                                  model.predict(cv_val)))
                
                # Final training on full dataset
                processed_train = model.preprocess_data(train_data)
                processed_val = model.preprocess_data(validation_data)
                model.train(processed_train, processed_val)
                
                self.models[name] = {
                    'model': model,
                    'cv_scores': cv_scores
                }
                
            except Exception as e:
                self.logger.log_error(f"Error training {name} model: {str(e)}")
                
    def evaluate_all_models(self, test_data: pd.DataFrame) -> Dict:
        """Evaluate models with comprehensive metrics"""
        results = {}
        for name, model_info in self.models.items():
            model = model_info['model']
            predictions = model.predict(test_data)
            metrics = self.evaluator.evaluate_model(model, test_data, predictions)
            
            results[name] = {
                'metrics': metrics,
                'predictions': predictions,
                'cv_scores': model_info['cv_scores']
            }
            
        self._save_evaluation_results(results)
        return results
    
    def generate_forecasts(self) -> Dict:
        """Generate forecasts with uncertainty estimation"""
        forecasts = {}
        
        for name, model_info in self.models.items():
            try:
                model = model_info['model']
                forecast = model.predict(data=None, forecast_horizon=self.forecast_horizon)
                forecasts[name] = forecast
            except Exception as e:
                self.logger.log_error(f"Error generating forecast for {name}: {str(e)}")
        
        ensemble_forecast = self._create_ensemble_forecast(forecasts)
        forecasts['ensemble'] = ensemble_forecast
        
        self._save_forecasts(forecasts)
        return forecasts
    
    def _create_ensemble_forecast(self, forecasts: Dict) -> pd.DataFrame:
        """Create weighted ensemble forecast based on model performance"""
        weights = self._calculate_weights()
        weighted_forecasts = []
        
        for name, forecast in forecasts.items():
            if name in weights:
                weighted_forecasts.append(forecast * weights[name])
        
        ensemble = pd.DataFrame({
            'forecast': sum(weighted_forecasts),
            'lower_bound': np.percentile([f['lower_bound'] for f in forecasts.values()], 2.5, axis=0),
            'upper_bound': np.percentile([f['upper_bound'] for f in forecasts.values()], 97.5, axis=0)
        })
        
        return ensemble
    
    def _calculate_weights(self) -> Dict[str, float]:
        """Calculate model weights based on performance metrics"""
        comparison = self.evaluator.compare_models()
        rmse_scores = comparison['rmse']
        
        # Calculate weights using softmax of inverse RMSE
        inverse_rmse = 1 / rmse_scores
        exp_scores = np.exp(inverse_rmse)
        weights = exp_scores / exp_scores.sum()
        
        return weights.to_dict()
    
    def generate_visualizations(self, data: pd.DataFrame, forecasts: Dict):
        """Generate comprehensive visualizations"""
        # Model predictions and comparisons
        self.visualizer.plot_predictions(data[self.target_variable], 
                                       {name: info['model'].predict(data) 
                                        for name, info in self.models.items()})
        
        # Metrics comparison
        self.visualizer.plot_metrics_comparison(self.evaluator.compare_models())
        
        # Residuals analysis
        residuals = {name: data[self.target_variable].values - info['model'].predict(data)
                    for name, info in self.models.items()}
        self.visualizer.plot_residuals_analysis(residuals)
        
        # Forecast visualization
        self.visualizer.plot_forecast_horizon(data[self.target_variable], forecasts,
                                            forecast_start=data.index[-1])
        
        # Time series components
        self.visualizer.plot_components(data[self.target_variable])
    
    def _save_evaluation_results(self, results: Dict):
        """Save evaluation results with timestamp"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"{self.output_dir}/metrics/evaluation_{timestamp}.json"
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = pd.json_normalize(results).to_dict(orient='records')[0]
        pd.DataFrame(serializable_results).to_json(output_path)
    
    def _save_forecasts(self, forecasts: Dict):
        """Save forecasts with metadata"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for name, forecast in forecasts.items():
            forecast.to_csv(f"{self.output_dir}/forecasts/{name}_forecast_{timestamp}.csv")
    
    def run_pipeline(self, df: pd.DataFrame) -> Tuple[Dict, Dict]:
        """Execute complete modeling pipeline"""
        try:
            # Prepare data
            train_data, validation_data, test_data = self.prepare_data(df)
            
            # Train and evaluate models
            self.train_models(train_data, validation_data)
            evaluation_results = self.evaluate_all_models(test_data)
            
            # Generate and visualize forecasts
            forecasts = self.generate_forecasts()
            self.generate_visualizations(df, forecasts)
            
            return evaluation_results, forecasts
            
        except Exception as e:
            self.logger.log_error(f"Pipeline execution failed: {str(e)}")
            raise