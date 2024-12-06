import numpy as np
import pandas as pd
import dask.dataframe as dd
import cudf
import dask_cudf
from scipy import stats
from typing import Dict, Tuple, Optional
import logging
from datetime import datetime, timedelta

class ClimateDataPreprocessor:
    # Physical constraints for meteorological variables
    VALID_RANGES = {
        'TEMPERATURA DO AR - BULBO SECO HORARIA °C': (-40, 50),
        'PRECIPITACÃO TOTAL HORÁRIO MM': (0, 500),
        'PRESSAO ATMOSFERICA AO NIVEL DA ESTACAO HORARIA MB': (800, 1100),
        'RADIACAO GLOBAL KJ/M²': (0, 5000),
        'UMIDADE RELATIVA DO AR HORARIA %': (0, 100),
        'VENTO VELOCIDADE HORARIA M/S': (0, 100)
    }
    
    INVALID_VALUES = [-9999.0, -999.0, -99.0, 9999.0]
    
    def __init__(self, target_variable: str, logger):
        self.target_variable = target_variable
        self.logger = logger
        self.essential_columns = [
            'DATETIME',
            self.target_variable,
            'PRECIPITACÃO TOTAL HORÁRIO MM',
            'PRESSAO ATMOSFERICA AO NIVEL DA ESTACAO HORARIA MB',
            'RADIACAO GLOBAL KJ/M²',
            'UMIDADE RELATIVA DO AR HORARIA %',
            'VENTO VELOCIDADE HORARIA M/S',
            'LATITUDE',
            'LONGITUDE',
            'ALTITUDE'
        ]

    def preprocess(self, df: pd.DataFrame, use_gpu: bool = False) -> pd.DataFrame:
        """Main preprocessing pipeline for climate data."""
        try:
            self.logger.info("Starting climate data preprocessing")
            
            # Replace invalid values
            df = self._handle_invalid_values(df)
            
            # Validate meteorological constraints
            df = self._validate_meteorological_data(df)
            
            # Handle missing values with climate-specific logic
            df = self._handle_missing_values(df)
            
            # Remove statistical outliers while preserving extreme weather events
            df = self._handle_climate_outliers(df)
            
            # Quality control checks
            df = self._apply_quality_control(df)
            
            # Select and order essential columns
            df = df[self.essential_columns].copy()
            
            # Set datetime index
            if 'DATETIME' in df.columns:
                df = df.set_index('DATETIME').sort_index()
            
            self.logger.info("Preprocessing completed successfully")
            return df
            
        except Exception as e:
            self.logger.error(f"Preprocessing failed: {str(e)}")
            raise

    def _handle_invalid_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Replace known invalid values with NaN."""
        for val in self.INVALID_VALUES:
            for col in df.select_dtypes(include=[np.number]).columns:
                df[col] = df[col].replace(val, np.nan)
        return df

    def _validate_meteorological_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate data based on physical constraints."""
        for column, (min_val, max_val) in self.VALID_RANGES.items():
            if column in df.columns:
                invalid_mask = (df[column] < min_val) | (df[column] > max_val)
                if invalid_mask.any():
                    self.logger.warning(
                        f"Found {invalid_mask.sum()} invalid values in {column}"
                    )
                    df.loc[invalid_mask, column] = np.nan
        return df

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values using climate-specific strategies."""
        # Short gaps (≤ 6 hours): Forward fill
        df = df.fillna(method='ffill', limit=6)
        
        # Medium gaps (≤ 24 hours): Interpolate with time and seasonal patterns
        for col in df.columns:
            if col in self.VALID_RANGES:
                # Use time-based interpolation for medium gaps
                df[col] = df[col].interpolate(
                    method='time',
                    limit=24,
                    limit_direction='both'
                )
        
        # Long gaps (> 24 hours): Use seasonal patterns
        df = self._fill_long_gaps(df)
        
        return df

    def _fill_long_gaps(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill long gaps using seasonal patterns."""
        for col in df.columns:
            if col in self.VALID_RANGES:
                # Calculate seasonal patterns
                hourly_mean = df.groupby(df.index.hour)[col].mean()
                daily_mean = df.groupby(df.index.dayofyear)[col].mean()
                
                # Fill remaining gaps using seasonal patterns
                missing_mask = df[col].isna()
                if missing_mask.any():
                    df.loc[missing_mask, col] = df.index[missing_mask].map(
                        lambda x: hourly_mean[x.hour] * 
                        (daily_mean[x.dayofyear] / daily_mean.mean())
                    )
        
        return df

    def _handle_climate_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle outliers while preserving extreme weather events."""
        for col in df.columns:
            if col in self.VALID_RANGES:
                # Calculate rolling statistics
                rolling_mean = df[col].rolling(window=24*7).mean()
                rolling_std = df[col].rolling(window=24*7).std()
                
                # Define dynamic thresholds
                threshold = 4  # More permissive than standard 3-sigma
                lower_bound = rolling_mean - threshold * rolling_std
                upper_bound = rolling_mean + threshold * rolling_std
                
                # Identify statistical outliers
                outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
                
                # Verify if outliers are part of a weather pattern
                if outliers.any():
                    weather_pattern = self._verify_weather_pattern(df, col, outliers)
                    # Only remove outliers not part of weather patterns
                    df.loc[outliers & ~weather_pattern, col] = np.nan
        
        return df

    def _verify_weather_pattern(self, df: pd.DataFrame, column: str, 
                              outlier_mask: pd.Series) -> pd.Series:
        """Verify if outliers are part of legitimate weather patterns."""
        weather_pattern = pd.Series(False, index=df.index)
        
        # Check for sustained patterns (3 or more consecutive values)
        rolling_outliers = outlier_mask.rolling(window=3).sum()
        weather_pattern |= (rolling_outliers >= 3)
        
        # Check for correlated changes in related variables
        if column == 'TEMPERATURA DO AR - BULBO SECO HORARIA °C':
            # Temperature drops often correlate with precipitation
            temp_drops = df[column].diff() < -5
            rain_events = df['PRECIPITACÃO TOTAL HORÁRIO MM'] > 0
            weather_pattern |= (temp_drops & rain_events)
            
        elif column == 'PRESSAO ATMOSFERICA AO NIVEL DA ESTACAO HORARIA MB':
            # Pressure changes often precede temperature changes
            pressure_changes = abs(df[column].diff()) > 10
            temp_changes = abs(df['TEMPERATURA DO AR - BULBO SECO HORARIA °C'].diff()) > 5
            weather_pattern |= (pressure_changes & temp_changes.shift(-6))
        
        return weather_pattern

    def _apply_quality_control(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply final quality control checks."""
        # Check for physically impossible rates of change
        max_hourly_changes = {
            'TEMPERATURA DO AR - BULBO SECO HORARIA °C': 10,  # °C per hour
            'PRESSAO ATMOSFERICA AO NIVEL DA ESTACAO HORARIA MB': 20,  # mb per hour
            'UMIDADE RELATIVA DO AR HORARIA %': 50  # % per hour
        }
        
        for col, max_change in max_hourly_changes.items():
            if col in df.columns:
                changes = df[col].diff().abs()
                invalid_changes = changes > max_change
                if invalid_changes.any():
                    self.logger.warning(
                        f"Found {invalid_changes.sum()} suspicious rate changes in {col}"
                    )
                    df.loc[invalid_changes, col] = np.nan
        
        # Ensure temporal consistency
        if isinstance(df.index, pd.DatetimeIndex):
            time_gaps = df.index.to_series().diff() > pd.Timedelta(hours=1)
            if time_gaps.any():
                self.logger.warning(
                    f"Found {time_gaps.sum()} time gaps larger than 1 hour"
                )
        
        return df

    def get_statistics(self, df: pd.DataFrame) -> Dict:
        """Calculate preprocessing statistics."""
        return {
            'missing_values': df.isnull().sum().to_dict(),
            'data_ranges': {
                col: {'min': df[col].min(), 'max': df[col].max()}
                for col in df.columns if col in self.VALID_RANGES
            },
            'completeness': (1 - df.isnull().mean()).to_dict()
        }