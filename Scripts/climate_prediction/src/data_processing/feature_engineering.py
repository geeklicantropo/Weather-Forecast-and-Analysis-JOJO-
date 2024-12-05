import numpy as np
import dask.dataframe as dd
from scipy import stats
from dask.diagnostics import ProgressBar

class FeatureEngineer:
    def __init__(self, target_variable="TEMPERATURA DO AR - BULBO SECO HORARIA °C"):
        self.target_variable = target_variable
        self.temp_col = 'mean_temp'  # After resampling to daily
        self.precip_col = 'PRECIPITACÃO TOTAL HORÁRIO MM'
        self.pressure_col = 'PRESSAO ATMOSFERICA AO NIVEL DA ESTACAO HORARIA MB'
        self.humidity_col = 'UMIDADE RELATIVA DO AR HORARIA %'
    
    def create_features(self, df):
        """Create comprehensive climate-specific features."""
        # Add basic temporal features
        df = self.add_temporal_features(df)
        
        # Add rolling statistics
        df = self.add_rolling_features(df)
        
        # Add climate indices
        df = self.add_climate_indices(df)
        
        # Add extreme event indicators
        df = self.add_extreme_indicators(df)
        
        # Add seasonal decomposition features
        df = self.add_seasonal_features(df)
        
        return df
    
    def add_temporal_features(self, df):
        """Add enhanced temporal features for climate analysis."""
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
            
            # Cyclic encoding of temporal features
            df['month_sin'] = np.sin(2 * np.pi * df.index.month / 12)
            df['month_cos'] = np.cos(2 * np.pi * df.index.month / 12)
            df['day_sin'] = np.sin(2 * np.pi * df.index.dayofyear / 365)
            df['day_cos'] = np.cos(2 * np.pi * df.index.dayofyear / 365)
            
            return df
        
        return df.map_partitions(process)
    
    def add_rolling_features(self, df):
        """Add rolling statistics at climate-relevant windows."""
        def process(df):
            # Temperature variability
            for window in [7, 30, 90, 365]:
                suffix = f'{window}d'
                df[f'temp_mean_{suffix}'] = df[self.temp_col].rolling(window=window).mean()
                df[f'temp_std_{suffix}'] = df[self.temp_col].rolling(window=window).std()
                df[f'temp_max_{suffix}'] = df[self.temp_col].rolling(window=window).max()
                df[f'temp_min_{suffix}'] = df[self.temp_col].rolling(window=window).min()
                
                # Calculate anomalies
                df[f'temp_anomaly_{suffix}'] = (
                    df[self.temp_col] - df[f'temp_mean_{suffix}']
                ) / df[f'temp_std_{suffix}']
            
            # Precipitation features
            df['dry_days'] = (df[self.precip_col] == 0).rolling(window=30).sum()
            df['wet_days'] = (df[self.precip_col] > 0).rolling(window=30).sum()
            
            return df
        
        return df.map_partitions(process)
    
    def add_climate_indices(self, df):
        """Add climate indices for trend analysis."""
        def process(df):
            # Temperature indices
            df['hot_days'] = (df[self.temp_col] > df[self.temp_col].quantile(0.9)).astype(int)
            df['cold_days'] = (df[self.temp_col] < df[self.temp_col].quantile(0.1)).astype(int)
            
            # Calculate growing degree days (base 10°C)
            df['growing_degree_days'] = df[self.temp_col].clip(lower=10) - 10
            
            # Heat stress index
            df['heat_index'] = self._calculate_heat_index(
                df[self.temp_col], 
                df[self.humidity_col]
            )
            
            return df
        
        return df.map_partitions(process)
    
    def add_extreme_indicators(self, df):
        """Add indicators for extreme weather events."""
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
    
    def add_seasonal_features(self, df):
        """Add seasonal decomposition features."""
        def process(df):
            # Calculate seasonal means
            df['seasonal_mean'] = df.groupby([df.index.month, df.index.day])[self.temp_col].transform('mean')
            df['seasonal_std'] = df.groupby([df.index.month, df.index.day])[self.temp_col].transform('std')
            
            # Calculate anomalies from seasonal pattern
            df['seasonal_anomaly'] = (df[self.temp_col] - df['seasonal_mean']) / df['seasonal_std']
            
            return df
        
        return df.map_partitions(process)
    
    @staticmethod
    def _calculate_heat_index(temp, humidity):
        """Calculate heat index using temperature and humidity."""
        # Simplified heat index calculation
        return temp + 0.555 * (humidity/100) * (temp - 14.5)