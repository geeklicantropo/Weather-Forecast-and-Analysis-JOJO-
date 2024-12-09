# src/models/sarima_model.py
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.model_selection import TimeSeriesSplit
import itertools
import joblib
from concurrent.futures import ProcessPoolExecutor
from src.models.base_model import BaseModel

import dask.dataframe as dd
import gc
import torch
from ..utils.gpu_manager import gpu_manager

from tqdm import tqdm
import psutil

from datetime import datetime
from typing import Dict, Optional, Tuple, Any
import os

class SARIMAModel(BaseModel):
    def __init__(self, target_variable, logger, order_ranges=None, seasonal_order_ranges=None):
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
        
    def _convert_to_dask(self, df: pd.DataFrame) -> dd.DataFrame:
        """Convert pandas DataFrame to Dask DataFrame if not already."""
        if not isinstance(df, (pd.DataFrame, dd.DataFrame)):
            return df
        if isinstance(df, dd.DataFrame):
            return df
        
        # Calculate optimal partitions
        chunk_bytes = 128 * 1024 * 1024
        npartitions = max(1, len(df) // (chunk_bytes // df.memory_usage(deep=True).mean()))
        
        return dd.from_pandas(df, npartitions=int(npartitions))
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.Series:
        try:
            self.logger.log_info(f"Starting SARIMA preprocessing for dataset of size {len(df):,} rows")
            
            ddf = self._convert_to_dask(df)
            batch_size = 500000
            n_partitions = max(1, len(df) // batch_size)
            ddf = ddf.repartition(npartitions=n_partitions)
            
            processed_chunks = []
            
            with tqdm(total=ddf.npartitions, desc="Processing SARIMA batches") as pbar:
                for chunk_df in ddf.partitions:
                    chunk = chunk_df.compute()
                    chunk_series = chunk[self.target_variable]
                    
                    if not isinstance(chunk_series.index, pd.DatetimeIndex):
                        chunk_series.index = pd.to_datetime(chunk_series.index)
                    
                    chunk_series = chunk_series.resample('H').mean()
                    chunk_series = chunk_series.fillna(method='ffill', limit=6)
                    chunk_series = chunk_series.fillna(method='bfill', limit=6)
                    
                    processed_chunks.append(chunk_series)
                    pbar.update(1)
                    gc.collect()
            
            final_series = pd.concat(processed_chunks)
            return final_series.sort_index()
            
        except Exception as e:
            self.logger.error(f"Error in SARIMA preprocessing: {str(e)}")
            raise
            
    def _evaluate_sarima_params(self, params, train_data, val_data):
        """Evaluate SARIMA parameters using validation set"""
        order, seasonal_order = params
        try:
            model = SARIMAX(
                train_data,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            fitted_model = model.fit(disp=False)
            forecast = fitted_model.forecast(steps=len(val_data))
            rmse = np.sqrt(np.mean((val_data - forecast) ** 2))
            aic = fitted_model.aic
            return (order, seasonal_order), (rmse, aic)
        except:
            return (order, seasonal_order), (np.inf, np.inf)
            
    def _grid_search(self, data):
        """Perform grid search for optimal parameters"""
        self.logger.log_info("Starting grid search for SARIMA parameters")
        
        # Generate parameter combinations
        p, d, q = self.order_ranges['p'], self.order_ranges['d'], self.order_ranges['q']
        P, D, Q, S = (self.seasonal_order_ranges['P'], self.seasonal_order_ranges['D'],
                      self.seasonal_order_ranges['Q'], self.seasonal_order_ranges['S'])
                      
        orders = list(itertools.product(p, d, q))
        seasonal_orders = list(itertools.product(P, D, Q, S))
        
        # Set up time series cross-validation
        tscv = TimeSeriesSplit(n_splits=3)
        best_score = np.inf
        best_params = None
        
        # Parallelize parameter evaluation
        with ProcessPoolExecutor() as executor:
            for train_idx, val_idx in tscv.split(data):
                train = data.iloc[train_idx]
                val = data.iloc[val_idx]
                
                param_combinations = list(itertools.product(orders, seasonal_orders))
                futures = [
                    executor.submit(self._evaluate_sarima_params, params, train, val)
                    for params in param_combinations
                ]
                
                # Collect results
                for future in futures:
                    params, (rmse, aic) = future.result()
                    if rmse < best_score:
                        best_score = rmse
                        best_params = params
                        self.logger.log_info(
                            f"New best parameters found: {params}, RMSE: {rmse:.4f}, AIC: {aic:.4f}"
                        )
        
        self.best_params = best_params
        return best_params
        
    def train(self, train_data: pd.DataFrame, validation_data: Optional[pd.DataFrame] = None) -> Dict:
        """Train SARIMA model with optimized parameters."""
        try:
            # Check for existing model
            model_path = os.path.join("Scripts/climate_prediction/outputs/models", "sarima_model_latest")
            if os.path.exists(model_path):
                self.logger.log_info("Loading existing SARIMA model")
                self._load_model_data(model_path)
                return {}
                
            train_series = self.preprocess_data(train_data)
            
            # Find optimal parameters if not set
            if self.best_params is None:
                self.best_params = self._grid_search(train_series)
                
            order, seasonal_order = self.best_params
            
            self.model = SARIMAX(
                train_series,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            
            self.fitted_model = self.model.fit(disp=False)
            
            # Store residuals and feature importance
            self.residuals = self.fitted_model.resid
            self.feature_importance = pd.Series(
                self.fitted_model.params,
                index=self.fitted_model.param_names
            ).abs().sort_values(ascending=False)
            
            # Perform diagnostics
            self._perform_diagnostics()
            
            # Save model
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join("Scripts/climate_prediction/outputs/models", f"sarima_model_{timestamp}")
            self._save_model_data(save_path)
            
            # Create/update latest symlink
            latest_path = os.path.join("Scripts/climate_prediction/outputs/models", "sarima_model_latest")
            if os.path.exists(latest_path):
                os.remove(latest_path)
            os.symlink(save_path, latest_path)
            
            # Return training metadata
            return {
                'order': order,
                'seasonal_order': seasonal_order,
                'aic': self.fitted_model.aic,
                'bic': self.fitted_model.bic,
                'residual_std': self.residuals.std()
            }
            
        except Exception as e:
            self.logger.log_error(f"Training error: {str(e)}")
            raise
            
    def _perform_diagnostics(self):
        """Perform model diagnostics and store results"""
        try:
            diagnostics = {
                'aic': self.fitted_model.aic,
                'bic': self.fitted_model.bic,
                'residual_mean': self.residuals.mean(),
                'residual_std': self.residuals.std(),
                'ljung_box_test': self.fitted_model.test_serial_correlation(method='ljungbox'),
                'normality_test': self.fitted_model.test_normality(method='jarquebera'),
                'heteroskedasticity_test': self.fitted_model.test_heteroskedasticity(method='breakvar')
            }
            self.diagnostics = diagnostics
            
            # Log important diagnostic information
            self.logger.log_info(f"Model AIC: {diagnostics['aic']:.2f}")
            self.logger.log_info(f"Model BIC: {diagnostics['bic']:.2f}")
            
        except Exception as e:
            self.logger.log_error(f"Diagnostics error: {str(e)}")
            
    def predict(self, data, forecast_horizon=None):
        try:
            if forecast_horizon is None:
                # In-sample predictions
                return self.fitted_model.get_prediction(start=data.index[0]).predicted_mean
            
            # Out-of-sample forecast with confidence intervals
            forecast = self.fitted_model.forecast(steps=forecast_horizon)
            forecast_ci = self.fitted_model.get_forecast(steps=forecast_horizon).conf_int()
            
            return pd.DataFrame({
                'forecast': forecast,
                'lower_ci': forecast_ci.iloc[:, 0],
                'upper_ci': forecast_ci.iloc[:, 1]
            })
            
        except Exception as e:
            self.logger.log_error(f"Prediction error: {str(e)}")
            raise
            
    def get_diagnostics(self):
        """Return model diagnostics"""
        return self.diagnostics
        
    def get_decomposition(self):
        """Return seasonal decomposition results"""
        return self.decomposition
            
    def _save_model_data(self, path):
        model_data = {
            'best_params': self.best_params,
            'order_ranges': self.order_ranges,
            'seasonal_order_ranges': self.seasonal_order_ranges,
            'decomposition': self.decomposition,
            'diagnostics': self.diagnostics,
            'residuals': self.residuals,
            'fitted_model': self.fitted_model.save(f"{path}/sarima.pkl")
        }
        joblib.dump(model_data, f"{path}/sarima_data.pkl")
        
    def _load_model_data(self, path):
        model_data = joblib.load(f"{path}/sarima_data.pkl")
        self.best_params = model_data['best_params']
        self.order_ranges = model_data['order_ranges']
        self.seasonal_order_ranges = model_data['seasonal_order_ranges']
        self.decomposition = model_data['decomposition']
        self.diagnostics = model_data['diagnostics']
        self.residuals = model_data['residuals']
        self.fitted_model = SARIMAX.load(f"{path}/sarima.pkl")