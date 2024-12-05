# src/data_processing/preprocessor.py
import numpy as np
import dask.dataframe as dd
import cudf
import dask_cudf
from scipy import stats

class DataPreprocessor:
    def __init__(self, target_variable, logger):
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
    
    def preprocess(self, df, use_gpu=False):
        """Main preprocessing pipeline"""
        self.logger.log_info("Starting preprocessing")
        
        try:
            # Replace invalid values
            df = self._replace_invalid_values(df, use_gpu)
            
            # Remove outliers
            df = self._handle_outliers(df, use_gpu)
            
            # Select essential columns
            df = df[self.essential_columns]
            
            # Sort and set index
            df = df.set_index('DATETIME').sort_index()
            
            return df
            
        except Exception as e:
            self.logger.log_error(f"Preprocessing failed: {str(e)}")
            raise
    
    def _replace_invalid_values(self, df, use_gpu):
        """Replace invalid values with NaN"""
        invalid_values = [-9999.0, -999.0, -99.0, 9999.0]
        
        if use_gpu:
            for val in invalid_values:
                df = df.map_partitions(lambda x: x.replace(val, cudf.NA))
        else:
            for val in invalid_values:
                df = df.map_partitions(lambda x: x.replace(val, np.nan))
        
        return df
    
    def _handle_outliers(self, df, use_gpu):
        """Remove statistical outliers"""
        def remove_outliers(x):
            if use_gpu:
                z_scores = (x - x.mean()) / x.std()
                return x.mask(abs(z_scores) > 3)
            else:
                z_scores = stats.zscore(x)
                return x.mask(abs(z_scores) > 3)
        
        numeric_columns = [
            self.target_variable,
            'PRECIPITACÃO TOTAL HORÁRIO MM',
            'PRESSAO ATMOSFERICA AO NIVEL DA ESTACAO HORARIA MB',
            'RADIACAO GLOBAL KJ/M²'
        ]
        
        for col in numeric_columns:
            df[col] = df[col].map_partitions(remove_outliers)
        
        return df
    
    def handle_missing_values(self, df, use_gpu=False):
        """Enhanced missing value handling for climate data"""
        def process_partition(df):
            # Handle different time gaps
            df = self._handle_small_gaps(df)
            df = self._handle_medium_gaps(df)
            df = self._handle_large_gaps(df)
            return df
        
        return df.map_partitions(process_partition)
    
    def _handle_small_gaps(self, df, max_hours=6):
        """Handle gaps ≤ 6 hours using forward fill"""
        return df.fillna(method='ffill', limit=max_hours)
    
    def _handle_medium_gaps(self, df, max_hours=24):
        """Handle gaps ≤ 24 hours using interpolation"""
        # Linear interpolation for temperature and pressure
        temp_pressure_cols = [
            self.target_variable,
            'PRESSAO ATMOSFERICA AO NIVEL DA ESTACAO HORARIA MB'
        ]
        
        for col in temp_pressure_cols:
            df[col] = df[col].interpolate(method='linear', limit=max_hours)
        
        # Seasonal interpolation for other variables
        other_cols = [col for col in df.columns if col not in temp_pressure_cols]
        for col in other_cols:
            df[col] = df[col].interpolate(method='time', limit=max_hours)
        
        return df
    
    def _handle_large_gaps(self, df):
        """Handle gaps > 24 hours using seasonal patterns"""
        def seasonal_fill(series):
            # Get seasonal pattern
            if len(series) > 24*7:  # Need at least a week of data
                seasonal_pattern = series.groupby(series.index.hour).mean()
                filled = series.fillna(seasonal_pattern[series.index.hour])
                return filled
            return series
        
        return df.apply(seasonal_fill)