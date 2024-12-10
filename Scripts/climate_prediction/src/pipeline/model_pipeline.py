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
from dask.distributed import Client, LocalCluster
from tqdm import tqdm
from pathlib import Path

from ..models.lstm_model import LSTMModel
from ..models.sarima_model import SARIMAModel
from ..models.tft_model import TFTModel
from ..models.model_evaluator import ModelEvaluator
from src.models.prediction_manager import PredictionManager
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
        self.prediction_manager = PredictionManager(self.logger)

    def _get_chunk_path(self, model_name: str) -> Path:
        """Get appropriate chunk path based on model type."""
        base_path = Path("Scripts/climate_prediction/outputs/data/temp")
        chunk_paths = {
            'lstm': base_path / 'sequence_chunks',
            'sarima': base_path / 'sarima_chunks',
            'tft': base_path / 'tft_chunks'
        }
        return chunk_paths[model_name]

    def load_data(self, train_path: str, test_path: str, chunk_size: int = 100000) -> Tuple[dd.DataFrame, dd.DataFrame]:
        """Load processed data with proper chunk management."""
        if not os.path.exists(train_path) or not os.path.exists(test_path):
            raise FileNotFoundError("Missing required data files")

        try:
            self.logger.log_info("Starting data loading process")
            
            # Calculate optimal chunk size
            available_memory = psutil.virtual_memory().available
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3 if torch.cuda.is_available() else 0
            target_memory = min(available_memory, gpu_memory * 1024**3 if gpu_memory else float('inf'))
            chunk_size = max(1000, min(int((target_memory * 0.1) / (2 * 1024)), chunk_size))
            
            # Initialize chunk tracking
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

            # Create chunk directories for each model type
            for model_type in ['lstm', 'sarima', 'tft']:
                chunk_dir = self._get_chunk_path(model_type)
                chunk_dir.mkdir(parents=True, exist_ok=True)

            # Process train data
            self.logger.log_info("Loading train data...")
            train_size = sum(1 for _ in pd.read_csv(train_path, chunksize=chunk_size))
            train_chunks = {model_type: [] for model_type in ['lstm', 'sarima', 'tft']}
            
            with tqdm(total=train_size, desc="Processing train chunks") as pbar:
                for chunk_idx, chunk in enumerate(pd.read_csv(train_path, chunksize=chunk_size, dtype=dtypes)):
                    # Handle datetime index
                    if 'DATETIME' in chunk.columns:
                        chunk['DATETIME'] = pd.to_datetime(chunk['DATETIME'])
                        chunk.set_index('DATETIME', inplace=True)
                    else:
                        chunk['DATETIME'] = pd.to_datetime(chunk['DATA YYYY-MM-DD'])
                        chunk.set_index('DATETIME', inplace=True)
                    
                    # Save chunks for each model type
                    for model_type in ['lstm', 'sarima', 'tft']:
                        chunk_dir = self._get_chunk_path(model_type)
                        chunk_path = chunk_dir / f'chunk_{chunk_idx:04d}.parquet'
                        
                        if not chunk_path.exists():
                            chunk.to_parquet(chunk_path)
                        train_chunks[model_type].append(chunk_path)
                    
                    pbar.update(1)
                    
                    # Clear memory periodically
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()

            # Process test data
            self.logger.log_info("Loading test data...")
            test_size = sum(1 for _ in pd.read_csv(test_path, chunksize=chunk_size))
            test_chunks = {model_type: [] for model_type in ['lstm', 'sarima', 'tft']}
            
            with tqdm(total=test_size, desc="Processing test chunks") as pbar:
                for chunk_idx, chunk in enumerate(pd.read_csv(test_path, chunksize=chunk_size, dtype=dtypes)):
                    if 'DATETIME' in chunk.columns:
                        chunk['DATETIME'] = pd.to_datetime(chunk['DATETIME'])
                        chunk.set_index('DATETIME', inplace=True)
                    else:
                        chunk['DATETIME'] = pd.to_datetime(chunk['DATA YYYY-MM-DD'])
                        chunk.set_index('DATETIME', inplace=True)
                    
                    # Save chunks for each model type
                    for model_type in ['lstm', 'sarima', 'tft']:
                        chunk_dir = self._get_chunk_path(model_type)
                        chunk_path = chunk_dir / f'test_chunk_{chunk_idx:04d}.parquet'
                        
                        if not chunk_path.exists():
                            chunk.to_parquet(chunk_path)
                        test_chunks[model_type].append(chunk_path)
                    
                    pbar.update(1)
                    
                    # Clear memory periodically
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()

            # Load data based on current model type
            current_model = self.current_model_type if hasattr(self, 'current_model_type') else 'lstm'
            
            # Load train data for current model
            train_data = dd.read_parquet(
                train_chunks[current_model],
                engine='pyarrow'
            ).repartition(npartitions=max(1, len(train_chunks[current_model])))

            # Load test data for current model
            test_data = dd.read_parquet(
                test_chunks[current_model],
                engine='pyarrow'
            ).repartition(npartitions=max(1, len(test_chunks[current_model])))

            self.logger.log_info(f"Loaded {current_model.upper()} train data: {len(train_data)} rows")
            self.logger.log_info(f"Loaded {current_model.upper()} test data: {len(test_data)} rows")

            return train_data, test_data

        except Exception as e:
            self.logger.log_error(f"Error loading data: {str(e)}")
            raise
        
    def _setup_directories(self):
        """Create necessary directories."""
        dirs = [
            'outputs/models',
            'outputs/predictions',
            'outputs/plots',
            'outputs/metrics',
            'outputs/logs'
        ]
        for dir_path in dirs:
            os.makedirs(os.path.join("Scripts/climate_prediction", dir_path), exist_ok=True)

    def _handle_existing_model(self, model_path: str) -> bool:
        """Check if model exists and handle symlink."""
        if os.path.exists(model_path):
            if os.path.islink(model_path):
                os.remove(model_path)
                return True
            return True
        return False

    def train_models(self, train_data: pd.DataFrame, test_data: pd.DataFrame) -> Dict[str, object]:
        """Train models strictly sequentially with comprehensive tracking."""
        if not self.file_checker.check_final_exists():
            raise ValueError("Final processed data files not found")

        models = {}
        
        # Sequential model order
        model_sequence = [
            ('sarima', SARIMAModel),
            ('lstm', LSTMModel),
            ('tft', TFTModel)
        ]
        
        with tqdm(total=len(model_sequence), desc="Training models") as pbar:
            for name, model_class in model_sequence:
                self.logger.log_info(f"\n{'='*20} Processing {name.upper()} Model {'='*20}")
                
                try:
                    # 1. Data Processing Stage
                    with tqdm(total=3, desc=f"{name} pipeline stages") as stage_pbar:
                        self.logger.log_info(f"Starting {name} data processing")
                        
                        model = model_class(self.target_variable, self.logger)
                        processed_train = model.preprocess_data(train_data)
                        processed_test = model.preprocess_data(test_data)
                        stage_pbar.update(1)
                        
                        # 2. Training Stage
                        self.logger.log_info(f"Training {name} model")
                        history = model.train(processed_train, processed_test)
                        stage_pbar.update(1)
                        
                        # 3. Evaluation and Results Stage
                        self.logger.log_info(f"Generating {name} results and visualizations")
                        
                        # Generate predictions
                        predictions = model.predict(test_data)
                        evaluator = ModelEvaluator(self.logger)
                        metrics = evaluator.evaluate_model(model, test_data, predictions)
                        
                        # Save all results
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        
                        # Save predictions
                        predictions.to_csv(f"Scripts/climate_prediction/outputs/predictions/{name}_predictions_{timestamp}.csv")
                        
                        # Save metrics
                        pd.DataFrame([metrics]).to_json(
                            f"Scripts/climate_prediction/outputs/metrics/{name}_metrics_{timestamp}.json"
                        )
                        
                        # Generate visualizations
                        self.visualizer.plot_predictions(
                            test_data[self.target_variable], 
                            {name: predictions}
                        )
                        
                        # Save model
                        save_path = f"Scripts/climate_prediction/outputs/models/{name}_model_{timestamp}"
                        model.save_model(save_path)
                        
                        # Update model symlink
                        model_path = f"Scripts/climate_prediction/outputs/models/{name}_model_latest"
                        if os.path.exists(model_path) and os.path.islink(model_path):
                            os.remove(model_path)
                        os.symlink(save_path, model_path)
                        
                        stage_pbar.update(1)
                        
                        # Store model
                        models[name] = model
                        
                        # Clear memory before next model
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        gc.collect()
                        
                        self.logger.log_info(f"\n{'='*20} Completed {name.upper()} Model {'='*20}\n")
                        
                except Exception as e:
                    self.logger.log_error(f"Error processing {name} model: {str(e)}")
                    continue
                    
                # Ensure complete synchronization before next model
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                pbar.update(1)
        
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
                       data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Generate forecasts with uncertainty estimation."""
        try:
            # Convert data to Dask DataFrame using PredictionManager
            dask_data = self.prediction_manager.prepare_data_for_prediction(data)
            
            # Use PredictionManager to handle forecast generation
            forecasts = self.prediction_manager.run_distributed_predictions(models, dask_data)
            
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
            
            # Compute model agreement
            agreement_df = self.prediction_manager.compute_model_agreement(forecasts)
            agreement_df.to_csv(f"{self.output_dir}/model_agreement.csv", index=True)
            
            return {
                'status': 'success',
                'models': models,
                'evaluation': evaluation_results,
                'forecasts': forecasts,
                'model_agreement': agreement_df
            }
            
        except Exception as e:
            self.logger.log_error(f"Pipeline failed: {str(e)}")
            raise

    def setup_dask_client(self):
        cluster = LocalCluster(
            n_workers=8,
            threads_per_worker=2,
            memory_limit='4GB'
        )
        return Client(cluster)