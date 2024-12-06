# src/data_processing/feature_engineering.py
import numpy as np
import pandas as pd
import cudf
import dask_cudf
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
from dask.diagnostics import ProgressBar
import dask.dataframe as dd

class FeatureEngineer:
    def __init__(self, target_variable, logger):
        self.target_variable = target_variable
        self.logger = logger
        self.temp_col = target_variable
        self.precip_col = 'PRECIPITACÃO TOTAL HORÁRIO MM'
        self.pressure_col = 'PRESSAO ATMOSFERICA AO NIVEL DA ESTACAO HORARIA MB'
        self.humidity_col = 'UMIDADE RELATIVA DO AR HORARIA %'

    def create_features(self, df, use_gpu=False):
        try:
            with ProgressBar():
                steps = [
                    self.add_temporal_features,
                    self.add_rolling_features,
                    self.add_climate_indices,
                    self.add_extreme_indicators,
                    self.add_seasonal_features
                ]
                
                for step in steps:
                    self.logger.log_info(f"Executing {step.__name__}")
                    df = step(df, use_gpu)
                
            return df
            
        except Exception as e:
            self.logger.log_error(f"Feature engineering failed: {str(e)}")
            raise

    def add_temporal_features(self, df, use_gpu=False):
        def process(df):
            # Basic temporal features
            df['year'] = df.index.year
            df['month'] = df.index.month
            df['day'] = df.index.day
            df['dayofyear'] = df.index.dayofyear
            
            # Seasonal features
            df['season'] = ((df.index.month % 12 + 3) // 3).map({
                1: 'Summer', 2: 'Fall', 3: 'Winter', 4: 'Spring'
            })
            
            # Cyclic encoding
            if use_gpu:
                df['month_sin'] = cudf.sin(2 * np.pi * df.index.month / 12)
                df['month_cos'] = cudf.cos(2 * np.pi * df.index.month / 12)
                df['day_sin'] = cudf.sin(2 * np.pi * df.index.dayofyear / 365)
                df['day_cos'] = cudf.cos(2 * np.pi * df.index.dayofyear / 365)
            else:
                df['month_sin'] = np.sin(2 * np.pi * df.index.month / 12)
                df['month_cos'] = np.cos(2 * np.pi * df.index.month / 12)
                df['day_sin'] = np.sin(2 * np.pi * df.index.dayofyear / 365)
                df['day_cos'] = np.cos(2 * np.pi * df.index.dayofyear / 365)
            
            return df
        
        return df.map_partitions(process)

    def add_rolling_features(self, df, use_gpu=False):
        def process(df):
            # Temperature variability for multiple windows
            for window in [7, 30, 90, 365]:
                suffix = f'{window}d'
                df[f'temp_mean_{suffix}'] = df[self.temp_col].rolling(window=window).mean()
                df[f'temp_std_{suffix}'] = df[self.temp_col].rolling(window=window).std()
                df[f'temp_max_{suffix}'] = df[self.temp_col].rolling(window=window).max()
                df[f'temp_min_{suffix}'] = df[self.temp_col].rolling(window=window).min()
                
                # Anomalies
                df[f'temp_anomaly_{suffix}'] = (
                    df[self.temp_col] - df[f'temp_mean_{suffix}']
                ) / df[f'temp_std_{suffix}']

            # Precipitation patterns
            df['dry_days'] = (df[self.precip_col] == 0).rolling(window=30).sum()
            df['wet_days'] = (df[self.precip_col] > 0).rolling(window=30).sum()
            
            # Rate of change features
            df['temp_change_24h'] = df[self.temp_col].diff(periods=24)
            df['temp_change_7d'] = df[self.temp_col].diff(periods=24*7)
            
            return df
        
        return df.map_partitions(process)

    def add_climate_indices(self, df, use_gpu=False):
        def process(df):
            # Temperature indices
            temp_q90 = df[self.temp_col].quantile(0.9)
            temp_q10 = df[self.temp_col].quantile(0.1)
            df['hot_days'] = (df[self.temp_col] > temp_q90).astype(int)
            df['cold_days'] = (df[self.temp_col] < temp_q10).astype(int)
            
            # Growing degree days
            df['growing_degree_days'] = df[self.temp_col].clip(lower=10) - 10
            
            # Heat and humidity indices
            df['heat_index'] = self._calculate_heat_index(
                df[self.temp_col], 
                df[self.humidity_col]
            )
            
            df['vapor_pressure_deficit'] = self._calculate_vpd(
                df[self.temp_col], 
                df[self.humidity_col]
            )
            
            return df
        
        return df.map_partitions(process)

    def add_extreme_indicators(self, df, use_gpu=False):
        def process(df):
            # Temperature extremes
            temp_std = df[self.temp_col].std()
            temp_mean = df[self.temp_col].mean()
            
            df['extreme_heat'] = (df[self.temp_col] > (temp_mean + 2 * temp_std)).astype(int)
            df['extreme_cold'] = (df[self.temp_col] < (temp_mean - 2 * temp_std)).astype(int)
            
            # Precipitation extremes
            precip_95th = df[self.precip_col].quantile(0.95)
            df['extreme_precip'] = (df[self.precip_col] > precip_95th).astype(int)
            
            return df
        
        return df.map_partitions(process)

    def add_seasonal_features(self, df, use_gpu=False):
        def process(df):
            # Seasonal patterns
            df['seasonal_mean'] = df.groupby([df.index.month, df.index.day])[self.temp_col].transform('mean')
            df['seasonal_std'] = df.groupby([df.index.month, df.index.day])[self.temp_col].transform('std')
            df['seasonal_anomaly'] = (df[self.temp_col] - df['seasonal_mean']) / df['seasonal_std']
            
            # Seasonal decomposition if enough data
            if len(df) >= 24*7:
                decomposition = seasonal_decompose(
                    df[self.temp_col],
                    period=24,
                    extrapolate_trend='freq'
                )
                df['seasonal'] = decomposition.seasonal
                df['trend'] = decomposition.trend
                df['residual'] = decomposition.resid
            
            return df
        
        return df.map_partitions(process)

    @staticmethod
    def _calculate_heat_index(temp, humidity):
        return temp + 0.555 * (humidity/100) * (temp - 14.5)

    @staticmethod
    def _calculate_vpd(temp, humidity):
        es = 0.6108 * np.exp(17.27 * temp / (temp + 237.3))
        ea = es * (humidity / 100)
        return es - ea