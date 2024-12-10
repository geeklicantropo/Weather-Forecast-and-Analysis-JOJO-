# src/models/sarima_model.py
import numpy as np
import pandas as pd
import dask.dataframe as dd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.model_selection import TimeSeriesSplit
import itertools
import joblib
from concurrent.futures import ProcessPoolExecutor
import psutil
import gc
import torch
from tqdm import tqdm
from datetime import datetime
from typing import Dict, Optional, Tuple, Any, List, Union
import os
import json
from pathlib import Path
from scipy import stats
import sys

from .base_model import BaseModel
from ..utils.gpu_manager import gpu_manager

class SARIMAModel(BaseModel):
    def __init__(self, target_variable: str, logger: Any, order_ranges=None, seasonal_order_ranges=None):
        super().__init__(target_variable, logger)
        self.order_ranges = order_ranges or {
            'p': range(0, 3),
            'd': range(0, 2),
            'q': range(0, 3)
        }
        self.seasonal_order_ranges = seasonal_order_ranges or {
            'P': range(0, 2),
            'D': range(0, 2),
            'Q': range(0, 2),
            'S': [12, 24]
        }
        self.best_params = None
        self.decomposition = None
        self.chunk_size = self._calculate_optimal_chunk_size()
        
        # Setup directories
        self.base_dir = Path("Scripts/climate_prediction/outputs")
        self.data_dir = self.base_dir / "data"
        self.model_dir = self.base_dir / "models"
        self.temp_dir = self.data_dir / "temp" / "sarima"
        
        # Create directories
        for dir_path in [self.data_dir, self.model_dir, self.temp_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize file tracking
        self.chunk_files = []
        self.checkpoint_file = self.temp_dir / "checkpoint.json"
        self.load_checkpoint()

    def _calculate_optimal_chunk_size(self) -> int:
        """Calculate optimal chunk size based on available system memory."""
        try:
            # Get 80% of available memory
            available_memory = psutil.virtual_memory().available * 0.8
            
            # Estimate row size in bytes (timestamps + numeric columns)
            estimated_row_bytes = 200
            
            # Target using 5% of available memory for processing
            target_memory = available_memory * 0.05
            optimal_chunk_size = int(target_memory / estimated_row_bytes)
            
            # Set bounds
            optimal_chunk_size = max(20_000, min(500_000, optimal_chunk_size))
            
            self.logger.log_info(f"Calculated optimal chunk size: {optimal_chunk_size:,} rows")
            return optimal_chunk_size
            
        except Exception as e:
            self.logger.log_warning(f"Error calculating chunk size: {str(e)}. Using default: 100,000")
            return 100_000

    def _process_chunk(self, chunk: pd.DataFrame) -> pd.DataFrame:
        """Process a single chunk of data."""
        try:
            chunk_dir = Path("Scripts/climate_prediction/outputs/data/temp/sarima_chunks")
            chunk_dir.mkdir(parents=True, exist_ok=True)

            # Ensure datetime index
            if not isinstance(chunk.index, pd.DatetimeIndex):
                chunk = chunk.copy()
                if 'DATETIME' in chunk.columns:
                    chunk.set_index('DATETIME', inplace=True)
                else:
                    chunk['DATETIME'] = pd.to_datetime(chunk['DATA YYYY-MM-DD'])
                    chunk.set_index('DATETIME', inplace=True)

            # Handle missing values
            chunk = self._handle_missing_values(chunk)

            # Basic processing
            processed = pd.DataFrame()
            processed[self.target_variable] = chunk[self.target_variable].astype('float32')
            processed = processed.resample('H').mean()

            # Save processed chunk
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            chunk_path = chunk_dir / f"chunk_{timestamp}.pkl"
            processed.to_pickle(chunk_path)
            self.chunk_files.append(chunk_path)

            return processed

        except Exception as e:
            self.logger.log_error(f"Error processing chunk: {str(e)}")
            raise

    def load_checkpoint(self):
        """Load checkpoint if exists."""
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, 'r') as f:
                    self.checkpoint = json.load(f)
                self.logger.log_info(f"Loaded checkpoint: {self.checkpoint['timestamp']}")
            except Exception as e:
                self.logger.log_warning(f"Failed to load checkpoint: {e}")
                self.checkpoint = self._create_new_checkpoint()
        else:
            self.checkpoint = self._create_new_checkpoint()

    def _create_new_checkpoint(self) -> Dict:
        """Create new checkpoint structure."""
        return {
            'timestamp': datetime.now().isoformat(),
            'processed_chunks': [],
            'completed_stages': [],
            'best_params': None,
            'last_processed_row': 0
        }

    def save_checkpoint(self):
        """Save current checkpoint."""
        try:
            with open(self.checkpoint_file, 'w') as f:
                json.dump(self.checkpoint, f)
        except Exception as e:
            self.logger.log_error(f"Failed to save checkpoint: {e}")

    def _get_chunk_path(self, chunk_index: int) -> Path:
        """Get path for chunk file."""
        return self.temp_dir / f"chunk_{chunk_index:04d}.pkl"

    def _chunk_exists(self, chunk_index: int) -> bool:
        """Check if chunk file exists and is valid."""
        chunk_path = self._get_chunk_path(chunk_index)
        return chunk_path.exists() and chunk_path.stat().st_size > 0

    def preprocess_data(self, df: Union[pd.DataFrame, dd.DataFrame]) -> pd.Series:
        """Preprocess data with checkpointing and resumption."""
        if df is None:
            return None

        try:
            self.logger.log_info(f"Starting SARIMA preprocessing for dataset of size {len(df):,} rows")

            # Convert Dask DataFrame to pandas if needed
            if isinstance(df, dd.DataFrame):
                df = df.compute()

            # Calculate total chunks
            total_chunks = (len(df) + self.chunk_size - 1) // self.chunk_size
            processed_chunks = set(self.checkpoint['processed_chunks'])

            temp_dir = Path("Scripts/climate_prediction/outputs/data/temp/sarima_chunks")
            temp_dir.mkdir(parents=True, exist_ok=True)

            with tqdm(total=total_chunks, desc="Processing SARIMA chunks") as pbar:
                pbar.update(len(processed_chunks))

                for chunk_idx in range(total_chunks):
                    if chunk_idx in processed_chunks:
                        continue

                    chunk_path = self._get_chunk_path(chunk_idx)
                    if chunk_path.exists():
                        try:
                            pd.read_pickle(chunk_path)
                            processed_chunks.add(chunk_idx)
                            self.chunk_files.append(chunk_path)
                            continue
                        except Exception:
                            chunk_path.unlink()

                    # Process chunk using loc instead of iloc
                    start_idx = df.index[chunk_idx * self.chunk_size]
                    end_idx = df.index[min((chunk_idx + 1) * self.chunk_size, len(df)) - 1]
                    chunk = df.loc[start_idx:end_idx]
                    
                    processed_chunk = self._process_chunk(chunk)
                    processed_chunk.to_pickle(chunk_path)
                    
                    self.chunk_files.append(chunk_path)
                    processed_chunks.add(chunk_idx)
                    
                    self.checkpoint['processed_chunks'] = list(processed_chunks)
                    self.checkpoint['last_processed_row'] = chunk_idx * self.chunk_size + len(chunk)
                    self.save_checkpoint()
                    
                    pbar.update(1)
                    gc.collect()

            final_series = self._combine_chunks()
            return final_series

        except Exception as e:
            self.logger.log_error(f"Error in SARIMA preprocessing: {str(e)}")
            raise

    def train(self, train_data: pd.DataFrame, validation_data: Optional[pd.DataFrame] = None, 
          progress_callback: callable = None) -> Dict:
        try:
            self.logger.log_info("\n=== Starting SARIMA Model Training ===")
            
            with tqdm(total=100, desc="Overall Progress", position=0) as pbar:
                # Check existing model
                model_path = os.path.join("Scripts/climate_prediction/outputs/models", "sarima_model_latest")
                if os.path.exists(model_path):
                    self.logger.log_info("Found existing model, loading...")
                    self._load_model_data(model_path)
                    pbar.update(100)
                    return {}

                # Preprocessing
                pbar.set_description("Preprocessing Data")
                self.logger.log_info("Preprocessing training data...")
                max_train_size = 5000
                train_sample = train_data.sample(n=min(len(train_data), max_train_size))
                train_series = self.preprocess_data(train_sample)
                pbar.update(20)

                # Grid Search
                pbar.set_description("Parameter Search")
                self.logger.log_info("Starting grid search...")
                
                from concurrent.futures import ThreadPoolExecutor, TimeoutError
                with ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        self._grid_search, 
                        train_series,
                        validation_data.sample(n=min(len(validation_data), max_train_size//2)) if validation_data is not None else None
                    )
                    try:
                        with tqdm(total=1, desc="Grid Search", position=1, leave=False) as search_pbar:
                            best_params = future.result(timeout=3600)
                            search_pbar.update(1)
                            self.logger.log_info(f"Found optimal parameters: {best_params}")
                    except TimeoutError:
                        self.logger.log_warning("Grid search timed out, using default parameters")
                        best_params = ((1,1,1), (1,1,1,24))
                pbar.update(30)

                # Model Fitting
                pbar.set_description("Training Model")
                self.logger.log_info("Training final model with best parameters...")
                
                with tqdm(total=1, desc="Model Fitting", position=1, leave=False) as fit_pbar:
                    self.model = SARIMAX(
                        train_series,
                        order=best_params[0],
                        seasonal_order=best_params[1],
                        enforce_stationarity=False,
                        enforce_invertibility=False
                    )
                    self.fitted_model = self.model.fit(disp=False)
                    fit_pbar.update(1)
                pbar.update(30)

                # Saving Model
                pbar.set_description("Saving Model")
                self.logger.log_info("Saving trained model...")
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = os.path.join("Scripts/climate_prediction/outputs/models", 
                                    f"sarima_model_{timestamp}")
                self._save_model_data(save_path)
                
                latest_path = os.path.join("Scripts/climate_prediction/outputs/models", 
                                        "sarima_model_latest")
                if os.path.exists(latest_path):
                    os.remove(latest_path)
                os.symlink(save_path, latest_path)
                pbar.update(20)

                self.logger.log_info("\n=== SARIMA Training Complete ===")
                self.logger.log_info(f"Model saved to: {save_path}")
                self.logger.log_info(f"Final parameters: Order={best_params[0]}, Seasonal={best_params[1]}")
                
                return {
                    'order': best_params[0],
                    'seasonal_order': best_params[1],
                    'aic': self.fitted_model.aic,
                    'bic': self.fitted_model.bic,
                    'training_time': timestamp
                }

        except Exception as e:
            self.logger.log_error(f"\n=== Training Failed ===\nError: {str(e)}")
            raise

    def predict(self, data: pd.DataFrame, forecast_horizon: Optional[int] = None) -> pd.DataFrame:
        """Generate predictions with checkpoint handling."""
        try:
            # Check for existing predictions
            pred_file = self.temp_dir / f"predictions_{datetime.now().strftime('%Y%m%d')}.pkl"
            if pred_file.exists():
                try:
                    predictions = pd.read_pickle(pred_file)
                    self.logger.log_info("Loaded existing predictions")
                    return predictions
                except Exception:
                    pred_file.unlink()

            # Generate new predictions
            input_series = self.preprocess_data(data) if isinstance(data, pd.DataFrame) else data
            
            results = pd.DataFrame()
            chunk_size = self.chunk_size

            for i in range(0, len(input_series), chunk_size):
                chunk = input_series.iloc[i:i + chunk_size]
                chunk_pred = self.fitted_model.get_prediction(start=chunk.index[0], 
                                                           end=chunk.index[-1])
                
                chunk_results = pd.DataFrame({
                    'forecast': chunk_pred.predicted_mean,
                    'lower_bound': chunk_pred.conf_int().iloc[:, 0],
                    'upper_bound': chunk_pred.conf_int().iloc[:, 1]
                })
                
                results = pd.concat([results, chunk_results])
                gc.collect()

            # Save predictions
            results.to_pickle(pred_file)
            return results

        except Exception as e:
            self.logger.log_error(f"Prediction error: {str(e)}")
            raise

    def cleanup(self):
        """Clean up temporary files while preserving checkpoint."""
        try:
            # Keep checkpoint file but clean up old chunks
            if 'training_complete' in self.checkpoint.get('completed_stages', []):
                for chunk_file in self.chunk_files:
                    if os.path.exists(chunk_file):
                        os.remove(chunk_file)
                self.chunk_files = []
            
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            self.logger.log_error(f"Error during cleanup: {str(e)}")

    def _evaluate_sarima_params(self, params: tuple, train_data: pd.Series, 
                              val_data: pd.Series) -> Tuple[tuple, tuple]:
        """Evaluate SARIMA parameters using validation set."""
        order, seasonal_order = params
        try:
            # Check if evaluation exists in checkpoint
            param_key = f"{order}_{seasonal_order}"
            if 'param_evaluations' in self.checkpoint and param_key in self.checkpoint['param_evaluations']:
                return (order, seasonal_order), self.checkpoint['param_evaluations'][param_key]

            # Use smaller subset for parameter evaluation
            eval_size = min(len(train_data), 5000)
            train_subset = train_data[-eval_size:]
            
            model = SARIMAX(
                train_subset,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            
            # Use chunked fitting for memory efficiency
            fitted_model = model.fit(disp=False)
            
            # Generate forecasts in chunks
            chunk_size = self.chunk_size
            forecasts = []
            
            for i in range(0, len(val_data), chunk_size):
                chunk_end = min(i + chunk_size, len(val_data))
                chunk_steps = chunk_end - i
                chunk_forecast = fitted_model.forecast(steps=chunk_steps)
                forecasts.append(chunk_forecast)
                
                gc.collect()
            
            forecast = pd.concat(forecasts)
            rmse = np.sqrt(np.mean((val_data - forecast) ** 2))
            aic = fitted_model.aic
            
            # Save evaluation result in checkpoint
            if 'param_evaluations' not in self.checkpoint:
                self.checkpoint['param_evaluations'] = {}
            self.checkpoint['param_evaluations'][param_key] = (rmse, aic)
            self.save_checkpoint()
            
            # Clean up
            del model, fitted_model, forecasts
            gc.collect()
            
            return (order, seasonal_order), (rmse, aic)
            
        except Exception as e:
            self.logger.log_warning(f"Parameter evaluation failed: {str(e)}")
            return (order, seasonal_order), (np.inf, np.inf)

    def _grid_search(self, data: pd.Series) -> tuple:
        """Perform grid search for optimal parameters with memory management."""
        self.logger.log_info("Starting grid search for SARIMA parameters")
        
        # Check if grid search was completed
        if self.checkpoint.get('grid_search_complete'):
            self.logger.log_info("Loading completed grid search results")
            return self.checkpoint['best_params']
        
        # Generate parameter combinations
        p, d, q = self.order_ranges['p'], self.order_ranges['d'], self.order_ranges['q']
        P, D, Q, S = (self.seasonal_order_ranges['P'], self.seasonal_order_ranges['D'],
                      self.seasonal_order_ranges['Q'], self.seasonal_order_ranges['S'])
        
        orders = list(itertools.product(p, d, q))
        seasonal_orders = list(itertools.product(P, D, Q, S))
        
        # Set up cross-validation with memory-efficient splits
        tscv = TimeSeriesSplit(n_splits=3)
        best_score = np.inf
        best_params = None
        
        # Load progress from checkpoint
        evaluated_params = set(self.checkpoint.get('evaluated_params', []))
        
        # Calculate total iterations for progress bar
        total_iterations = len(orders) * len(seasonal_orders) * 3
        completed_iterations = len(evaluated_params) * 3
        
        with ProcessPoolExecutor() as executor:
            with tqdm(total=total_iterations, 
                     initial=completed_iterations,
                     desc="Grid search progress") as pbar:
                
                for train_idx, val_idx in tscv.split(data):
                    train = data.iloc[train_idx]
                    val = data.iloc[val_idx]
                    
                    param_combinations = [
                        (order, seasonal_order) 
                        for order in orders 
                        for seasonal_order in seasonal_orders
                        if f"{order}_{seasonal_order}" not in evaluated_params
                    ]
                    
                    if not param_combinations:
                        continue
                    
                    futures = [
                        executor.submit(self._evaluate_sarima_params, params, train, val)
                        for params in param_combinations
                    ]
                    
                    for future in futures:
                        try:
                            params, (rmse, aic) = future.result()
                            param_key = f"{params[0]}_{params[1]}"
                            
                            if rmse < best_score:
                                best_score = rmse
                                best_params = params
                                self.checkpoint['best_params'] = params
                                self.checkpoint['best_score'] = float(best_score)
                                self.save_checkpoint()
                                
                            evaluated_params.add(param_key)
                            self.checkpoint['evaluated_params'] = list(evaluated_params)
                            self.save_checkpoint()
                            
                        except Exception as e:
                            self.logger.log_error(f"Error in parameter evaluation: {str(e)}")
                            continue
                        
                        finally:
                            pbar.update(1)
                            gc.collect()
                
        self.checkpoint['grid_search_complete'] = True
        self.save_checkpoint()
        
        return best_params

    def _combine_chunks(self) -> pd.Series:
        """Combine processed chunks efficiently."""
        combined_series = pd.Series(dtype=float)
        chunk_size = self.chunk_size
        
        for chunk_file in sorted(self.chunk_files):
            try:
                chunk = pd.read_pickle(chunk_file)
                combined_series = pd.concat([combined_series, chunk])
                
                if len(combined_series) > chunk_size:
                    # Save intermediate results if getting too large
                    temp_file = self.temp_dir / "intermediate_combined.pkl"
                    combined_series.to_pickle(temp_file)
                    
                    # Clear memory
                    del combined_series
                    gc.collect()
                    
                    # Reload from disk
                    combined_series = pd.read_pickle(temp_file)
                
            except Exception as e:
                self.logger.log_warning(f"Error loading chunk {chunk_file}: {str(e)}")
                continue
            
            finally:
                gc.collect()
        
        return combined_series.sort_index()

    def _save_model_data(self, path: str):
        """Save model and metadata."""
        try:
            model_path = Path(path)
            model_path.mkdir(parents=True, exist_ok=True)
            
            # Save model in chunks if too large
            model_size = sys.getsizeof(self.fitted_model.params) / (1024 * 1024)  # Size in MB
            
            if model_size > 100:  # If model is larger than 100MB
                # Save in chunks
                chunk_dir = model_path / "model_chunks"
                chunk_dir.mkdir(exist_ok=True)
                
                for i, param_chunk in enumerate(np.array_split(self.fitted_model.params, 10)):
                    chunk_file = chunk_dir / f"params_chunk_{i}.pkl"
                    joblib.dump(param_chunk, chunk_file)
            else:
                # Save normally
                self.fitted_model.save(str(model_path / "sarima.pkl"))
            
            # Save metadata
            metadata = {
                'best_params': self.best_params,
                'order_ranges': self.order_ranges,
                'seasonal_order_ranges': self.seasonal_order_ranges,
                'decomposition': self.decomposition,
                'chunk_info': {'size': model_size, 'chunked': model_size > 100}
            }
            
            with open(model_path / "metadata.json", 'w') as f:
                json.dump(metadata, f, indent=4)
            
            self.logger.log_info(f"Saved model to {path}")
            
        except Exception as e:
            self.logger.log_error(f"Error saving model: {str(e)}")
            raise

    def _load_model_data(self, path: str):
        """Load model and metadata with chunk handling."""
        try:
            model_path = Path(path)
            
            # Load metadata
            with open(model_path / "metadata.json", 'r') as f:
                metadata = json.load(f)
            
            self.best_params = metadata['best_params']
            self.order_ranges = metadata['order_ranges']
            self.seasonal_order_ranges = metadata['seasonal_order_ranges']
            self.decomposition = metadata['decomposition']
            
            # Load model based on chunk info
            if metadata.get('chunk_info', {}).get('chunked', False):
                chunk_dir = model_path / "model_chunks"
                model_params = []
                
                for chunk_file in sorted(chunk_dir.glob("params_chunk_*.pkl")):
                    param_chunk = joblib.load(chunk_file)
                    model_params.append(param_chunk)
                
                model_params = np.concatenate(model_params)
                self.fitted_model = SARIMAX.from_params(model_params)
            else:
                self.fitted_model = SARIMAX.load(str(model_path / "sarima.pkl"))
            
            self.logger.log_info(f"Loaded model from {path}")
            
        except Exception as e:
            self.logger.log_error(f"Error loading model: {str(e)}")
            raise

    def get_diagnostics(self) -> Dict:
        """Return model diagnostics using chunked processing."""
        if not hasattr(self, 'fitted_model'):
            raise ValueError("Model not trained yet")
            
        diagnostics = {}
        chunk_size = self.chunk_size
        
        try:
            # Process diagnostics in chunks
            residuals = []
            for i in range(0, len(self.fitted_model.resid), chunk_size):
                chunk = self.fitted_model.resid[i:i + chunk_size]
                residuals.extend(chunk)
                gc.collect()
            
            diagnostics['residual_mean'] = float(np.mean(residuals))
            diagnostics['residual_std'] = float(np.std(residuals))
            
            # Calculate other statistics
            diagnostics['normality_test'] = stats.jarque_bera(residuals)[1]
            diagnostics['ljung_box_test'] = self.fitted_model.test_serial_correlation(method='ljungbox')[0][0]
            
            return diagnostics
            
        except Exception as e:
            self.logger.log_error(f"Error calculating diagnostics: {str(e)}")
            return {}