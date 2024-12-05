class DataValidator:
    @staticmethod
    def validate_schema(df):
        """Validate the data schema and types."""
        required_columns = [
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
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        return True

    @staticmethod
    def validate_data_quality(df):
        """Check data quality metrics."""
        # Basic quality checks
        null_percentages = df.isnull().mean() * 100
        high_null_cols = null_percentages[null_percentages > 20].index.tolist()
        
        if high_null_cols:
            print(f"Warning: Following columns have >20% null values: {high_null_cols}")
        
        return null_percentages