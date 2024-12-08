from typing import Dict
import pandas as pd
import numpy as np
import dask.dataframe as dd

class DataStandardizer:
    def __init__(self, logger):
        self.logger = logger
        self.numeric_dtypes = {
            'PRECIPITACÃO TOTAL HORÁRIO MM': 'float32',
            'PRESSAO ATMOSFERICA AO NIVEL DA ESTACAO HORARIA MB': 'float32',
            'PRESSÃO ATMOSFERICA MAX.NA HORA ANT. AUT MB': 'float32',
            'PRESSÃO ATMOSFERICA MIN. NA HORA ANT. AUT MB': 'float32',
            'RADIACAO GLOBAL KJ/M²': 'float32',
            'TEMPERATURA DO AR - BULBO SECO HORARIA °C': 'float32',
            'TEMPERATURA DO PONTO DE ORVALHO °C': 'float32',
            'TEMPERATURA MÁXIMA NA HORA ANT. AUT °C': 'float32',
            'TEMPERATURA MÍNIMA NA HORA ANT. AUT °C': 'float32',
            'TEMPERATURA ORVALHO MAX. NA HORA ANT. AUT °C': 'float32',
            'TEMPERATURA ORVALHO MIN. NA HORA ANT. AUT °C': 'float32',
            'UMIDADE REL. MAX. NA HORA ANT. AUT %': 'float32',
            'UMIDADE REL. MIN. NA HORA ANT. AUT %': 'float32',
            'UMIDADE RELATIVA DO AR HORARIA %': 'float32',
            'VENTO DIRECÃO HORARIA GR ° GR': 'float32',
            'VENTO RAJADA MAXIMA M/S': 'float32',
            'VENTO VELOCIDADE HORARIA M/S': 'float32',
            'LATITUDE': 'float32',
            'LONGITUDE': 'float32',
            'ALTITUDE': 'float32'
        }
        
        self.categorical_dtypes = {
            'REGIAO': 'category',
            'UF': 'category',
            'ESTACAO': 'category',
            'CODIGO (WMO)': 'category'
        }
        
        self.datetime_columns = ['DATA YYYY-MM-DD', 'DATETIME']

    def standardize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize data types for all columns."""
        try:
            # Handle numeric columns
            for col, dtype in self.numeric_dtypes.items():
                if col in df.columns:
                    df[col] = df[col].astype(dtype)
            
            # Handle categorical columns
            for col, dtype in self.categorical_dtypes.items():
                if col in df.columns:
                    df[col] = df[col].astype(dtype)
            
            # Handle datetime columns
            for col in self.datetime_columns:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col])
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error standardizing data types: {str(e)}")
            return df

    def standardize_dask(self, ddf: dd.DataFrame) -> dd.DataFrame:
        """Standardize data types for Dask DataFrame."""
        try:
            # Combine all dtype specifications
            all_dtypes = {**self.numeric_dtypes, **self.categorical_dtypes}
            
            # Apply dtypes
            ddf = ddf.astype({col: dtype 
                             for col, dtype in all_dtypes.items() 
                             if col in ddf.columns})
            
            # Handle datetime columns separately
            for col in self.datetime_columns:
                if col in ddf.columns:
                    ddf[col] = dd.to_datetime(ddf[col])
            
            return ddf
            
        except Exception as e:
            self.logger.error(f"Error standardizing Dask data types: {str(e)}")
            return ddf

    def get_dtype_info(self, df: pd.DataFrame) -> Dict:
        """Get information about current data types."""
        return {
            'numeric_columns': {col: str(df[col].dtype) 
                              for col in self.numeric_dtypes.keys() 
                              if col in df.columns},
            'categorical_columns': {col: str(df[col].dtype) 
                                  for col in self.categorical_dtypes.keys() 
                                  if col in df.columns},
            'datetime_columns': {col: str(df[col].dtype) 
                               for col in self.datetime_columns 
                               if col in df.columns}
        }