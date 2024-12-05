# src/data_processing/data_processing.py
import os
import dask.dataframe as dd
from datetime import datetime
import json
from tqdm import tqdm
from .preprocessor import DataPreprocessor
from .feature_engineering import FeatureEngineer
from .data_validator import DataValidator
from .data_versioning import DataVersioning
import logging

class DataProcessingPipeline:
    def __init__(self, data_path, target_variable="TEMPERATURA DO AR - BULBO SECO HORARIA °C"):
        self.data_path = data_path
        self.target_variable = target_variable
        self._initialize_components()
        
    def _initialize_components(self):
        """Initialize pipeline components."""
        try:
            self.preprocessor = DataPreprocessor(self.target_variable)
            self.feature_engineer = FeatureEngineer(self.target_variable)
            self.validator = DataValidator()
            self.versioning = DataVersioning()
            logging.info("Pipeline components initialized successfully")
        except Exception as e:
            logging.error(f"Error initializing components: {str(e)}")
            raise
    
    def load_data(self):
        """Load data using Dask for memory efficiency."""
        logging.info(f"Loading data from {self.data_path}")
        try:
            with tqdm(total=1, desc="Loading Data") as pbar:
                df = dd.read_csv(self.data_path, compression='gzip')
                # Convert date and time columns
                df['DATA YYYY-MM-DD'] = dd.to_datetime(df['DATA YYYY-MM-DD'])
                df['DATETIME'] = df['DATA YYYY-MM-DD'].astype(str) + ' ' + df['HORA UTC']
                df['DATETIME'] = dd.to_datetime(df['DATETIME'])
                df = df.set_index('DATETIME')
                df = df.compute()  # Convert to pandas for time series processing
                pbar.update(1)
            
            logging.info(f"Data loaded successfully. Shape: {df.shape}")
            return df
            
        except Exception as e:
            logging.error(f"Error loading data: {str(e)}")
            raise
    
    def process_data(self, df):
        """Process the loaded data."""
        try:
            # Validate initial data
            logging.info("Validating data schema...")
            with tqdm(total=1, desc="Validating Schema") as pbar:
                self.validator.validate_schema(df)
                pbar.update(1)
            
            # Preprocess data
            logging.info("Preprocessing data...")
            with tqdm(total=2, desc="Preprocessing") as pbar:
                df = self.preprocessor.preprocess(df)
                pbar.update(1)
                df = self.preprocessor.handle_missing_values(df)
                pbar.update(1)
            
            # Create features
            logging.info("Performing feature engineering...")
            with tqdm(total=1, desc="Feature Engineering") as pbar:
                df = self.feature_engineer.create_features(df)
                pbar.update(1)
            
            # Final validation
            logging.info("Performing final validation...")
            with tqdm(total=1, desc="Final Validation") as pbar:
                quality_metrics = self.validator.validate_data_quality(df)
                pbar.update(1)
            
            return df, quality_metrics
            
        except Exception as e:
            logging.error(f"Error processing data: {str(e)}")
            raise
    
    def save_results(self, df, quality_metrics):
        """Save processed data and version information."""
        try:
            # Save processed data
            output_path = 'outputs/data/processed_data.parquet'
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            logging.info(f"Saving processed data to {output_path}")
            with tqdm(total=1, desc="Saving Data") as pbar:
                df.to_parquet(output_path)
                pbar.update(1)
            
            # Save version info
            version_info = {
                'timestamp': datetime.now().isoformat(),
                'data_stats': {
                    'total_samples': len(df),
                    'date_range': [df.index.min().isoformat(), df.index.max().isoformat()],
                    'features': list(df.columns),
                    'quality_metrics': quality_metrics
                },
                'preprocessing_steps': [
                    'Cleaned invalid values',
                    'Handled missing values',
                    'Created temporal features',
                    'Added climate indices',
                    'Performed quality validation'
                ]
            }
            
            return version_info
            
        except Exception as e:
            logging.error(f"Error saving results: {str(e)}")
            raise
    
    def run_pipeline(self):
        """Execute the complete data processing pipeline."""
        try:
            logging.info("Starting data processing pipeline...")
            
            # Load data
            df = self.load_data()
            
            # Process data
            df, quality_metrics = self.process_data(df)
            
            # Save results and get version info
            version_info = self.save_results(df, quality_metrics)
            
            logging.info("Data processing pipeline completed successfully!")
            
            return df, version_info
            
        except Exception as e:
            logging.error(f"Pipeline failed: {str(e)}")
            raise

if __name__ == "__main__":
    # Example usage
    data_path = './Scripts/all_data/csvs_concatenated/concatenated_full/full_concatenated.csv.gz'
    pipeline = DataProcessingPipeline(data_path)
    df, version_info = pipeline.run_pipeline()