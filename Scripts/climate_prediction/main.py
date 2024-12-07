import os
import sys
from pathlib import Path
import logging
import pandas as pd
import numpy as np
from datetime import datetime
import json
import yaml
from tqdm import tqdm
import torch
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.data_processing.data_processing import DataProcessingPipeline
from src.data_processing.data_loader import DataLoader
from src.models.train_evaluate import ModelTrainEvaluate
from src.utils.logger import ProgressLogger
from src.utils.config_manager import ConfigManager
from src.utils.gpu_manager import gpu_manager
from src.visualization.visualization_manager import VisualizationManager

class ClimateModelPipeline:
    def __init__(self, config_path: str = "config/model_config.yaml"):
        # Convert config_path to absolute path
        self.config_path = os.path.join(project_root, config_path)
        self.setup_directories()
        self.logger = ProgressLogger(name="ClimateModelPipeline")
        self.config_manager = ConfigManager(self.config_path)
        self.config = self.config_manager.get_config()
        self.visualizer = VisualizationManager(self.logger)
        
        # Initialize GPU if available
        self.device = gpu_manager.get_device()
        if torch.cuda.is_available():
            self.logger.log_info(f"Using GPU: {torch.cuda.get_device_name(0)}")
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            self.logger.log_info(f"GPU Memory: {gpu_memory:.2f} GB")
        else:
            self.logger.log_info("Using CPU")
    
    def setup_directories(self):
        """Create all necessary directories."""
        directories = [
            'Scripts/climate_prediction/outputs/data',
            'Scripts/climate_prediction/outputs/models', 
            'Scripts/climate_prediction/outputs/plots',
            'Scripts/climate_prediction/outputs/logs',
            'Scripts/climate_prediction/outputs/predictions',
            'Scripts/climate_prediction/outputs/metrics',
            'Scripts/climate_prediction/outputs/metadata',
            'Scripts/climate_prediction/config'
        ]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def load_and_process_data(self, data_path: str) -> pd.DataFrame:
        """Load and process the climate data."""
        try:
            self.logger.log_info("Starting data loading and processing...")
            
            # Initialize data loader
            data_loader = DataLoader(data_path, self.logger)
            
            # Load data with progress tracking
            with tqdm(desc="Loading data", unit="steps") as pbar:
                df = data_loader.load_data()
                pbar.update(1)
            
            # Initialize data processing pipeline
            pipeline = DataProcessingPipeline(
                self.config['preprocessing']['target_variable'],
                self.logger
            )
            
            # Process data with progress tracking
            with tqdm(desc="Processing data", unit="steps") as pbar:
                df, version_info = pipeline.run_pipeline(df)
                pbar.update(1)
            
            # Save processed data
            output_path = 'outputs/data/processed_data.parquet'
            df.to_parquet(output_path)
            
            # Save version info
            version_path = f'outputs/metadata/data_version_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
            with open(version_path, 'w') as f:
                json.dump(version_info, f, indent=4)
            
            self.logger.log_info(f"Data processing completed. Shape: {df.shape}")
            return df
            
        except Exception as e:
            self.logger.log_error(f"Error in data loading and processing: {str(e)}")
            raise
    
    def train_and_evaluate_models(self, df: pd.DataFrame) -> dict:
        """Train and evaluate all models."""
        try:
            self.logger.log_info("Starting model training and evaluation...")
            
            # Initialize training pipeline
            train_evaluate = ModelTrainEvaluate(
                config_path=self.config_manager.config_path
            )
            
            # Run training and evaluation pipeline
            with tqdm(desc="Training and evaluating models", unit="steps") as pbar:
                results = train_evaluate.run_pipeline(df)
                pbar.update(1)
            
            self.logger.log_info("Model training and evaluation completed")
            return results
            
        except Exception as e:
            self.logger.log_error(f"Error in model training and evaluation: {str(e)}")
            raise
    
    def generate_visualizations(self, df: pd.DataFrame, results: dict):
        """Generate all required visualizations."""
        try:
            self.logger.log_info("Generating visualizations...")
            
            # Time series decomposition
            with tqdm(desc="Generating visualizations", total=5) as pbar:
                self.visualizer.plot_components(
                    df[self.config['preprocessing']['target_variable']]
                )
                pbar.update(1)
                
                # Model predictions comparison
                self.visualizer.plot_predictions(
                    df[self.config['preprocessing']['target_variable']],
                    results['evaluation_results']
                )
                pbar.update(1)
                
                # Model metrics comparison
                metrics_df = pd.DataFrame({
                    name: res['metrics']
                    for name, res in results['evaluation_results'].items()
                })
                self.visualizer.plot_metrics_comparison(metrics_df)
                pbar.update(1)
                
                # Forecast horizon plots
                self.visualizer.plot_forecast_horizon(
                    df[self.config['preprocessing']['target_variable']],
                    results['future_predictions'],
                    forecast_start=df.index[-1]
                )
                pbar.update(1)
                
                # Feature importance plots (if available)
                if 'feature_importance' in results:
                    self.visualizer.plot_feature_importance(
                        results['feature_importance']
                    )
                pbar.update(1)
            
            self.logger.log_info("Visualization generation completed")
            
        except Exception as e:
            self.logger.log_error(f"Error generating visualizations: {str(e)}")
            raise
    
    def save_results(self, results: dict):
        """Save all results and metadata."""
        try:
            self.logger.log_info("Saving results...")
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save evaluation metrics
            metrics_path = f'outputs/metrics/final_metrics_{timestamp}.json'
            with open(metrics_path, 'w') as f:
                json.dump(
                    {name: res['metrics'] 
                     for name, res in results['evaluation_results'].items()},
                    f, indent=4
                )
            
            # Save predictions
            for model_name, predictions in results['future_predictions'].items():
                pred_path = f'outputs/predictions/{model_name}_predictions_{timestamp}.csv'
                predictions.to_csv(pred_path)
            
            # Save execution metadata
            metadata = {
                'timestamp': timestamp,
                'config_used': self.config,
                'model_performance': {
                    name: res['metrics']
                    for name, res in results['evaluation_results'].items()
                },
                'training_duration': {
                    name: hist.get('training_time', None)
                    for name, hist in results['training_histories'].items()
                }
            }
            
            metadata_path = f'outputs/metadata/execution_metadata_{timestamp}.json'
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)
            
            self.logger.log_info("Results saved successfully")
            
        except Exception as e:
            self.logger.log_error(f"Error saving results: {str(e)}")
            raise
    
    def run_pipeline(self, data_path: str):
        """Execute the complete modeling pipeline."""
        try:
            self.logger.log_info("Starting climate modeling pipeline...")
            start_time = datetime.now()
            
            # Load and process data
            df = self.load_and_process_data(data_path)
            
            # Train and evaluate models
            results = self.train_and_evaluate_models(df)
            
            # Generate visualizations
            self.generate_visualizations(df, results)
            
            # Save results
            self.save_results(results)
            
            end_time = datetime.now()
            duration = end_time - start_time
            
            self.logger.log_info(f"Pipeline completed successfully in {duration}")
            
            # Return summary of results
            return {
                'status': 'success',
                'duration': str(duration),
                'data_shape': df.shape,
                'models_trained': list(results['evaluation_results'].keys()),
                'best_model': min(
                    results['evaluation_results'].items(),
                    key=lambda x: x[1]['metrics']['rmse']
                )[0]
            }
            
        except Exception as e:
            self.logger.log_error(f"Pipeline failed: {str(e)}")
            raise

def main():
    # Update config path to match project structure
    config_path = "./config/model_config.yaml"
    data_path = './Scripts/all_data/csvs_concatenated/concatenated_full/full_concatenated.csv.gz'
    
    try:
        # Initialize and run pipeline
        pipeline = ClimateModelPipeline(config_path)
        summary = pipeline.run_pipeline(data_path)
        
        # Print final summary
        print("\nPipeline Execution Summary:")
        print("==========================")
        print(f"Status: {summary['status']}")
        print(f"Duration: {summary['duration']}")
        print(f"Data Shape: {summary['data_shape']}")
        print(f"Models Trained: {', '.join(summary['models_trained'])}")
        print(f"Best Performing Model: {summary['best_model']}")
        
    except Exception as e:
        logging.error(f"Pipeline execution failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()