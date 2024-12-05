# src/models/sarima_model.py
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.model_selection import TimeSeriesSplit
import itertools
import joblib
from concurrent.futures import ProcessPoolExecutor
from .base_model import BaseModel

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
            'S': [12, 24]  # 12 for yearly, 24 for daily seasonality
        }
        self.best_params = None
        self.decomposition = None
        
    def preprocess_data(self, df):
        try:
            df = df.copy()
            df.index = pd.to_datetime(df.index)
            
            # Ensure hourly frequency and handle missing values
            df = df.resample('H').mean()
            df = df.fillna(method='ffill', limit=6)  # Forward fill up to 6 hours
            df = df.fillna(method='bfill', limit=6)  # Backward fill remaining gaps
            
            # Remove remaining NaN if any
            df = df.dropna()
            
            # Perform seasonal decomposition
            self.decomposition = seasonal_decompose(
                df[self.target_variable], 
                period=24,  # 24 hours seasonality
                extrapolate_trend='freq'
            )
            
            return df[self.target_variable]
            
        except Exception as e:
            self.logger.log_error(f"Preprocessing error: {str(e)}")
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
        
    def train(self, train_data, validation_data=None):
        try:
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
            
            # Calculate and store residuals
            self.residuals = self.fitted_model.resid
            
            # Store feature importance based on model coefficients
            self.feature_importance = pd.Series(
                self.fitted_model.params,
                index=self.fitted_model.param_names
            ).abs().sort_values(ascending=False)
            
            self.logger.log_info("SARIMA model training completed")
            self.logger.log_info(f"Final model parameters: {self.best_params}")
            
            # Model diagnostics
            self._perform_diagnostics()
            
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