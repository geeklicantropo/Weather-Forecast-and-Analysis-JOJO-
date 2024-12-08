# Scripts/climate_prediction/src/pipeline/model_pipeline.py
import os
from typing import Dict, Tuple, Optional
import pandas as pd
import torch
from datetime import datetime
import psutil
import json
import numpy as np
import dask.dataframe as dd
import gc

from ..models.lstm_model import LSTMModel
from ..models.sarima_model import SARIMAModel
from ..models.tft_model import TFTModel
from ..models.model_evaluator import ModelEvaluator
from ..visualization.visualization_manager import VisualizationManager
from .data_pipeline import DataPipeline
from ..utils.logger import ProgressLogger
from ..utils.config_manager import ConfigManager
from ..utils.gpu_manager import gpu_manager
from ..utils.file_checker import FileChecker


class ModelPipeline:
    def __init__(self, config_path: str = "config/model_config.yaml"):
        self.logger = ProgressLogger(name="ModelPipeline")
        self.config = ConfigManager(config_path).get_config()
        self.device = gpu_manager.get_device()
        self.data_pipeline = DataPipeline(config_path)
        self.visualizer = VisualizationManager(self.logger)
        self.target_variable = self.config['preprocessing']['target_variable']
        self.file_checker = FileChecker()
        
    def load_data(self, train_path: str, test_path: str, 
              chunk_size: int = 500000) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load processed data with memory optimization."""
        if not os.path.exists(train_path) or not os.path.exists(test_path):
            self.logger.log_error("Required data files not found")
            raise FileNotFoundError("Missing required data files")

        try:
            # Calculate total size and available memory
            total_size = os.path.getsize(train_path) + os.path.getsize(test_path)
            available_memory = psutil.virtual_memory().available
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3 if torch.cuda.is_available() else 0
            
            # Use the smaller of available system memory or GPU memory
            target_memory = min(available_memory, gpu_memory * 1024**3 if gpu_memory else float('inf'))
            
            # If total size is too large, raise error
            if total_size > target_memory * 0.8:  # Use 80% of available memory
                raise MemoryError(
                    f"Data size ({total_size/1024**3:.2f} GB) exceeds available memory "
                    f"({target_memory/1024**3:.2f} GB). Consider processing in smaller chunks."
                )
            
            # Load data with optimized chunk size
            optimal_chunk_size = int((target_memory * 0.1) / (2 * 1024))  # 10% of memory per chunk
            chunk_size = max(1000, min(optimal_chunk_size, chunk_size))
            
            self.logger.log_info(f"Loading data with chunk size: {chunk_size}")
            
            # Load with dtype optimization
            dtypes = {
                'PRECIPITACÃO TOTAL HORÁRIO MM': 'float32',
                'PRESSAO ATMOSFERICA AO NIVEL DA ESTACAO HORARIA MB': 'float32',
                'TEMPERATURA DO AR - BULBO SECO HORARIA °C': 'float32',
                'UMIDADE RELATIVA DO AR HORARIA %': 'float32',
                'VENTO VELOCIDADE HORARIA M/S': 'float32',
                'LATITUDE': 'float32',
                'LONGITUDE': 'float32',
                'ALTITUDE': 'float32'
            }
            
            # Load train data
            train_data = pd.read_csv(train_path, compression='gzip', dtype=dtypes)
            if 'DATETIME' in train_data.columns:
                train_data['DATETIME'] = pd.to_datetime(train_data['DATETIME'])
                train_data.set_index('DATETIME', inplace=True)
            else:
                train_data['DATETIME'] = pd.to_datetime(train_data['DATA YYYY-MM-DD'])
                train_data.set_index('DATETIME', inplace=True)
            
            # Load test data
            test_data = pd.read_csv(test_path, compression='gzip', dtype=dtypes)
            if 'DATETIME' in test_data.columns:
                test_data['DATETIME'] = pd.to_datetime(test_data['DATETIME'])
                test_data.set_index('DATETIME', inplace=True)
            else:
                test_data['DATETIME'] = pd.to_datetime(test_data['DATA YYYY-MM-DD'])
                test_data.set_index('DATETIME', inplace=True)
            
            return train_data, test_data
            
        except Exception as e:
            self.logger.log_error(f"Error loading data: {str(e)}")
            raise
        
    def train_models(self, train_data: pd.DataFrame, 
                    test_data: pd.DataFrame) -> Dict[str, object]:
        """Train all models with provided data."""
        if not self.file_checker.check_final_exists():
            raise ValueError("Final processed data files not found")

        models = {}
        model_classes = {
            'lstm': LSTMModel,
            'sarima': SARIMAModel,
            'tft': TFTModel
        }
        
        for name, model_class in model_classes.items():
            self.logger.log_info(f"Training {name.upper()} model")
            
            # Check if model already exists
            model_path = f"Scripts/climate_prediction/outputs/models/{name}_model_latest"
            if os.path.exists(model_path):
                self.logger.log_info(f"Loading existing {name} model")
                model = model_class(self.target_variable, self.logger)
                model.load_model(model_path)
                models[name] = model
                continue
            
            model = model_class(self.target_variable, self.logger)
            
            # Process data for model
            processed_train = model.preprocess_data(train_data)
            processed_test = model.preprocess_data(test_data)
            
            # Train model
            model.train(processed_train, processed_test)
            models[name] = model
            
            # Save model
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            save_path = f"Scripts/climate_prediction/outputs/models/{name}_model_{timestamp}"
            model.save_model(save_path)
            
            # Create latest symlink
            if os.path.exists(model_path):
                os.remove(model_path)
            os.symlink(save_path, model_path)
            
        return models
        
    def evaluate_models(self, models: Dict[str, object], 
                       test_data: pd.DataFrame) -> Dict[str, Dict]:
        """Evaluate all trained models."""
        evaluator = ModelEvaluator(self.logger)
        results = {}
        
        for name, model in models.items():
            predictions = model.predict(test_data)
            metrics = evaluator.evaluate_model(model, test_data, predictions)
            results[name] = {
                'metrics': metrics,
                'predictions': predictions
            }
            
        # Generate comparison visualizations
        self.visualizer.plot_metrics_comparison(
            pd.DataFrame({name: res['metrics'] for name, res in results.items()})
        )
        
        return results
    
    def generate_forecasts(self, models: Dict[str, object], 
                      train_data: dd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Generate forecasts with uncertainty estimation."""
        forecasts = {}
        forecast_horizon = self.config['output']['forecast']['horizon']
        
        try:
            for name, model in models.items():
                self.logger.log_info(f"Generating forecast for {name.upper()} model")
                
                try:
                    # Generate forecast with uncertainty bounds
                    with gpu_manager.memory_monitor():
                        forecast = model.predict(train_data, forecast_horizon)
                        forecasts[name] = forecast
                    
                    # Save individual model forecasts
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    save_path = f"{self.output_dir}/forecasts/{name}_forecast_{timestamp}.csv"
                    forecast.to_csv(save_path)
                    
                    # Clear memory after each forecast
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        
                except Exception as e:
                    self.logger.log_error(f"Error generating forecast for {name}: {str(e)}")
                    continue
            
            # Generate ensemble forecast if we have multiple models
            if len(forecasts) > 1:
                try:
                    self.logger.log_info("Generating ensemble forecast")
                    ensemble_forecast = self._create_ensemble_forecast(forecasts)
                    forecasts['ensemble'] = ensemble_forecast
                    
                    # Save ensemble forecast
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    ensemble_forecast.to_csv(
                        f"{self.output_dir}/forecasts/ensemble_forecast_{timestamp}.csv"
                    )
                    
                except Exception as e:
                    self.logger.log_error(f"Error generating ensemble forecast: {str(e)}")
            
            return forecasts
            
        except Exception as e:
            self.logger.log_error(f"Error in forecast generation: {str(e)}")
            raise

    def _create_ensemble_forecast(self, forecasts: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Create weighted ensemble forecast based on model performance."""
        # Get weights from model evaluations
        weights = self._calculate_model_weights(forecasts)
        
        # Initialize arrays for weighted forecasts
        weighted_forecasts = []
        lower_bounds = []
        upper_bounds = []
        
        # Calculate weighted forecasts and confidence intervals
        for name, forecast in forecasts.items():
            if name == 'ensemble':
                continue
                
            weight = weights.get(name, 0)
            weighted_forecasts.append(forecast['forecast'] * weight)
            lower_bounds.append(forecast['lower_bound'])
            upper_bounds.append(forecast['upper_bound'])
        
        # Combine forecasts
        ensemble_forecast = pd.DataFrame({
            'forecast': sum(weighted_forecasts),
            'lower_bound': np.percentile(lower_bounds, 2.5, axis=0),
            'upper_bound': np.percentile(upper_bounds, 97.5, axis=0)
        }, index=forecasts[list(forecasts.keys())[0]].index)
        
        return ensemble_forecast

    def _calculate_model_weights(self, forecasts: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """Calculate weights for ensemble based on model performance."""
        # Get the latest evaluation metrics for each model
        metrics = {}
        for name in forecasts.keys():
            if name == 'ensemble':
                continue
            try:
                metadata_path = os.path.join(
                    self.output_dir, 
                    'metrics', 
                    f'{name}_metrics_latest.json'
                )
                with open(metadata_path, 'r') as f:
                    model_metrics = json.load(f)
                metrics[name] = model_metrics.get('rmse', float('inf'))
            except Exception:
                metrics[name] = float('inf')
        
        # Calculate weights using softmax of inverse RMSE
        inverse_rmse = {name: 1/rmse for name, rmse in metrics.items()}
        total = sum(inverse_rmse.values())
        weights = {name: value/total for name, value in inverse_rmse.items()}
        
        return weights
        
    def run_pipeline(self) -> Dict:
        """Execute complete modeling pipeline with file checks."""
        try:
            # Check if final processed files exist
            if not self.file_checker.check_final_exists():
                self.logger.log_info("Processing data using data pipeline...")
                data_results = self.data_pipeline.run_pipeline()
                train_path = data_results['paths']['train']
                test_path = data_results['paths']['test']
            else:
                train_path = str(self.file_checker.get_file_path('final', 'train'))
                test_path = str(self.file_checker.get_file_path('final', 'test'))
                self.logger.log_info("Using existing processed data files")
            
            # Load processed data
            train_data, test_data = self.load_data(train_path, test_path)
            
            # Train models
            models = self.train_models(train_data, test_data)
            
            # Evaluate models
            evaluation_results = self.evaluate_models(models, test_data)
            
            # Generate forecasts
            forecasts = self.generate_forecasts(models, train_data)
            
            return {
                'status': 'success',
                'models': models,
                'evaluation': evaluation_results,
                'forecasts': forecasts
            }
            
        except Exception as e:
            self.logger.log_error(f"Pipeline failed: {str(e)}")
            raise