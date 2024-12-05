# src/data_processing/preprocessor.py
import numpy as np
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
from datetime import datetime, timedelta

class DataPreprocessor:
    def __init__(self, target_variable="TEMPERATURA DO AR - BULBO SECO HORARIA °C"):
        self.target_variable = target_variable
        self.essential_columns = [
            'DATETIME',
            'TEMPERATURA DO AR - BULBO SECO HORARIA °C',
            'PRECIPITACÃO TOTAL HORÁRIO MM',
            'PRESSAO ATMOSFERICA AO NIVEL DA ESTACAO HORARIA MB',
            'RADIACAO GLOBAL KJ/M²',
            'UMIDADE RELATIVA DO AR HORARIA %',
            'VENTO VELOCIDADE HORARIA M/S',
            'LATITUDE',
            'LONGITUDE',
            'ALTITUDE'
        ]
    
    def preprocess(self, df):
        """Enhanced preprocessing with climate-specific handling."""
        # Replace -9999.0 with NaN and handle other invalid values
        df = df.map_partitions(self._clean_invalid_values)
        
        # Select and rename columns for clarity
        df = df[self.essential_columns]
        
        # Convert to daily frequency for climate analysis
        df = self._resample_to_daily(df)
        
        # Add quality flags
        df = self._add_quality_flags(df)
        
        return df
    
    def _clean_invalid_values(self, df):
        """Clean invalid values with climate-specific thresholds."""
        df = df.replace(-9999.0, np.nan)
        
        # Temperature constraints (in Celsius)
        df.loc[df[self.target_variable] < -40, self.target_variable] = np.nan
        df.loc[df[self.target_variable] > 50, self.target_variable] = np.nan
        
        # Pressure constraints (in MB)
        pressure_col = 'PRESSAO ATMOSFERICA AO NIVEL DA ESTACAO HORARIA MB'
        df.loc[df[pressure_col] < 800, pressure_col] = np.nan
        df.loc[df[pressure_col] > 1100, pressure_col] = np.nan
        
        # Humidity constraints (in %)
        humidity_col = 'UMIDADE RELATIVA DO AR HORARIA %'
        df.loc[df[humidity_col] < 0, humidity_col] = np.nan
        df.loc[df[humidity_col] > 100, humidity_col] = np.nan
        
        return df
    
    def _resample_to_daily(self, df):
        """Resample hourly data to daily frequency with appropriate aggregations."""
        agg_dict = {
            self.target_variable: {
                'mean_temp': 'mean',
                'max_temp': 'max',
                'min_temp': 'min',
                'temp_range': lambda x: x.max() - x.min()
            },
            'PRECIPITACÃO TOTAL HORÁRIO MM': 'sum',
            'PRESSAO ATMOSFERICA AO NIVEL DA ESTACAO HORARIA MB': 'mean',
            'RADIACAO GLOBAL KJ/M²': 'sum',
            'UMIDADE RELATIVA DO AR HORARIA %': 'mean',
            'VENTO VELOCIDADE HORARIA M/S': 'mean'
        }
        
        return df.resample('D').agg(agg_dict)
    
    def _add_quality_flags(self, df):
        """Add quality control flags for climate data."""
        def add_flags(df):
            # Flag for rapid temperature changes
            df['temp_change'] = df[self.target_variable].diff()
            df['rapid_temp_change'] = abs(df['temp_change']) > 10
            
            # Flag for long periods of constant values
            df['constant_temp'] = (df[self.target_variable] == 
                                 df[self.target_variable].shift()).rolling(24).sum() >= 23
            
            # Flag for suspicious daily temperature ranges
            df['suspicious_range'] = df['temp_range'] > 30
            
            return df
        
        return df.map_partitions(add_flags)
    
    def handle_missing_values(self, df):
        """Enhanced missing value handling for climate data."""
        def climate_interpolation(df):
            # Short gaps: Linear interpolation (up to 6 hours)
            df = df.interpolate(method='linear', limit=6)
            
            # Medium gaps: Use daily cycle (6-24 hours)
            daily_pattern = df[self.target_variable].groupby(df.index.hour).mean()
            
            for col in df.columns:
                if df[col].isna().any():
                    # Use seasonal patterns for longer gaps
                    seasonal_pattern = (
                        df[col]
                        .groupby([df.index.month, df.index.hour])
                        .mean()
                    )
                    
                    # Fill remaining gaps with seasonal patterns
                    df[col] = df[col].fillna(
                        df.groupby([df.index.month, df.index.hour])
                        .transform(lambda x: x.fillna(x.mean()))
                    )
            
            return df
        
        return df.map_partitions(climate_interpolation)