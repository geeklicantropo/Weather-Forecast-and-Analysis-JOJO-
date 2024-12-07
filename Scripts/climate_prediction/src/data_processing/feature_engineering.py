import pandas as pd
import numpy as np
from typing import Dict, List
import gc
from tqdm import tqdm
import logging
from datetime import datetime, timedelta
import psutil

class FeatureEngineer:
    def __init__(self, target_variable: str, logger, chunk_size: int = 20000):
        self.target_variable = target_variable
        self.logger = logger
        self.chunk_size = chunk_size
        self.temp_col = target_variable
        self.precip_col = 'PRECIPITACÃO TOTAL HORÁRIO MM'
        self.pressure_col = 'PRESSAO ATMOSFERICA AO NIVEL DA ESTACAO HORARIA MB'
        self.humidity_col = 'UMIDADE RELATIVA DO AR HORARIA %'
        
    def process_file(self, input_path: str, output_path: str) -> None:
        try:
            total_rows = sum(1 for _ in pd.read_csv(input_path, chunksize=self.chunk_size))
            buffer = pd.DataFrame()
            overlap = 24 * 7  # One week overlap for rolling features
            
            with tqdm(total=total_rows, desc="Engineering features") as pbar:
                for i, chunk in enumerate(pd.read_csv(input_path, chunksize=self.chunk_size)):
                    try:
                        # Combine with buffer
                        current_data = pd.concat([buffer, chunk]) if not buffer.empty else chunk
                        
                        # Process features
                        processed_chunk = self._process_chunk(current_data)
                        
                        # Save non-overlapping part
                        if i > 0:  # Not first chunk
                            processed_chunk = processed_chunk.iloc[overlap:]
                        
                        # Save chunk
                        mode = 'w' if i == 0 else 'a'
                        header = i == 0
                        processed_chunk.to_csv(output_path, compression='gzip',
                                            mode=mode, header=header, index=True)
                        
                        # Update buffer
                        buffer = chunk.iloc[-overlap:].copy() if i < total_rows - 1 else pd.DataFrame()
                        
                        # Update progress and cleanup
                        pbar.update(len(chunk))
                        self._cleanup_memory()
                        
                    except Exception as e:
                        self.logger.error(f"Error processing chunk {i}: {str(e)}")
                        continue
                        
        except Exception as e:
            self.logger.error(f"Feature engineering failed: {str(e)}")
            raise
            
    def _process_chunk(self, chunk: pd.DataFrame) -> pd.DataFrame:
        """Process a single chunk of data."""
        df = chunk.copy()
        df.index = pd.to_datetime(df.index)
        
        # Sequential feature creation
        df = self._add_temporal_features(df)
        df = self._add_rolling_features(df)
        df = self._add_climate_indices(df)
        df = self._add_extreme_indicators(df)
        df = self._add_seasonal_features(df)
        
        return df
        
    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features."""
        df['hour'] = df.index.hour
        df['day'] = df.index.day
        df['month'] = df.index.month
        df['dayofyear'] = df.index.dayofyear
        df['week'] = df.index.isocalendar().week
        
        # Cyclic encoding
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_sin'] = np.sin(2 * np.pi * df['dayofyear'] / 365.25)
        df['day_cos'] = np.cos(2 * np.pi * df['dayofyear'] / 365.25)
        
        return df
        
    def _add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create rolling statistical features."""
        windows = [24, 24*7, 24*30]  # 1 day, 1 week, 1 month
        
        for window in windows:
            suffix = f'{window}h'
            
            # Temperature features
            rolling_temp = df[self.temp_col].rolling(window=window, min_periods=1)
            df[f'temp_mean_{suffix}'] = rolling_temp.mean()
            df[f'temp_std_{suffix}'] = rolling_temp.std()
            df[f'temp_range_{suffix}'] = rolling_temp.max() - rolling_temp.min()
            
            # Precipitation features
            rolling_precip = df[self.precip_col].rolling(window=window, min_periods=1)
            df[f'precip_sum_{suffix}'] = rolling_precip.sum()
            df[f'dry_spell_{suffix}'] = (df[self.precip_col] == 0).rolling(window).sum()
            
            # Pressure trends
            df[f'pressure_trend_{suffix}'] = df[self.pressure_col].diff(periods=window)
            
        return df
        
    def _add_climate_indices(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate climate indices."""
        # Heat index
        df['heat_index'] = self._calculate_heat_index(
            df[self.temp_col],
            df[self.humidity_col]
        )
        
        # Vapor pressure deficit
        df['vapor_pressure_deficit'] = self._calculate_vpd(
            df[self.temp_col],
            df[self.humidity_col]
        )
        
        # Wind chill for cold temperatures
        wind_speed = df['VENTO VELOCIDADE HORARIA M/S']
        df['wind_chill'] = np.where(
            df[self.temp_col] < 10,
            self._calculate_wind_chill(df[self.temp_col], wind_speed),
            df[self.temp_col]
        )
        
        return df
        
    def _add_extreme_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create extreme weather indicators."""
        # Calculate thresholds from chunk
        temp_stats = df[self.temp_col].quantile([0.05, 0.95])
        precip_95th = df[self.precip_col].quantile(0.95)
        
        # Create indicators
        df['extreme_heat'] = (df[self.temp_col] > temp_stats[0.95]).astype(int)
        df['extreme_cold'] = (df[self.temp_col] < temp_stats[0.05]).astype(int)
        df['heavy_precip'] = (df[self.precip_col] > precip_95th).astype(int)
        
        return df
        
    def _add_seasonal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add seasonal and interaction features."""
        # Seasonal indicators
        df['season'] = pd.cut(df['month'],
                            bins=[0, 3, 6, 9, 12],
                            labels=['Winter', 'Spring', 'Summer', 'Fall'])
        
        # Day/night indicator
        df['is_daytime'] = ((df['hour'] >= 6) & (df['hour'] <= 18)).astype(int)
        
        # Temperature-humidity interactions
        df['temp_humidity_interaction'] = df[self.temp_col] * df[self.humidity_col] / 100
        
        return df
        
    @staticmethod
    def _calculate_heat_index(temp: pd.Series, humidity: pd.Series) -> pd.Series:
        """Calculate heat index using temperature and humidity."""
        return temp + 0.555 * (humidity/100) * (temp - 14.5)
        
    @staticmethod
    def _calculate_vpd(temp: pd.Series, humidity: pd.Series) -> pd.Series:
        """Calculate vapor pressure deficit."""
        es = 0.6108 * np.exp(17.27 * temp / (temp + 237.3))
        ea = es * (humidity / 100)
        return es - ea
        
    @staticmethod
    def _calculate_wind_chill(temp: pd.Series, wind_speed: pd.Series) -> pd.Series:
        """Calculate wind chill temperature."""
        return 13.12 + 0.6215 * temp - 11.37 * (wind_speed**0.16) + \
               0.3965 * temp * (wind_speed**0.16)
               
    def _cleanup_memory(self):
        """Clean up memory after chunk processing."""
        gc.collect()
        memory_usage = psutil.Process().memory_info().rss / 1024 / 1024
        if memory_usage > 1024:
            self.logger.warning(f"High memory usage: {memory_usage:.2f} MB")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    engineer = FeatureEngineer(
        target_variable="TEMPERATURA DO AR - BULBO SECO HORARIA °C",
        logger=logger
    )
    
    input_files = [
        "Scripts/climate_prediction/outputs/data/train_preprocessed.csv.gz",
        "Scripts/climate_prediction/outputs/data/test_preprocessed.csv.gz"
    ]
    output_files = [
        "Scripts/climate_prediction/outputs/data/train_final.csv.gz",
        "Scripts/climate_prediction/outputs/data/test_final.csv.gz"
    ]
    
    for input_path, output_path in zip(input_files, output_files):
        engineer.process_file(input_path, output_path)