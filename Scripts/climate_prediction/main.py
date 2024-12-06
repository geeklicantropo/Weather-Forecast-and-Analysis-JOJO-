# climate_prediction/main.py
import os
import logging
import pandas as pd
from datetime import datetime
import json
from src.data_processing.data_processing import DataProcessingPipeline
from src.models.model_trainer import ModelTrainer
import warnings
warnings.filterwarnings('ignore')

def setup_directories():
    """Create necessary project directories."""
    directories = [
        'outputs/data',
        'outputs/models',
        'outputs/plots',
        'outputs/logs',
        'outputs/predictions',
        'outputs/metrics',
        'config'
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def setup_logging():
    """Setup logging configuration."""
    log_filename = f'outputs/logs/pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )

def main():
    # Setup
    setup_directories()
    setup_logging()
    logging.info("Starting climate prediction pipeline...")
    
    try:
        # Initialize data processing pipeline
        data_path = './Scripts/all_data/csvs_concatenated/concatenated_full/full_concatenated.csv.gz'
        pipeline = DataProcessingPipeline(data_path)
        
        # Process data
        logging.info("Processing data...")
        df, version_info = pipeline.run_pipeline()
        
        # Split data chronologically into train, validation, and test sets
        logging.info("Splitting data...")
        total_days = (df.index.max() - df.index.min()).days
        train_end = df.index.min() + pd.Timedelta(days=int(total_days * 0.7))
        val_end = train_end + pd.Timedelta(days=int(total_days * 0.15))
        
        train_data = df[df.index <= train_end]
        val_data = df[(df.index > train_end) & (df.index <= val_end)]
        test_data = df[df.index > val_end]
        
        logging.info(f"Train data shape: {train_data.shape}, from {train_data.index.min()} to {train_data.index.max()}")
        logging.info(f"Validation data shape: {val_data.shape}, from {val_data.index.min()} to {val_data.index.max()}")
        logging.info(f"Test data shape: {test_data.shape}, from {test_data.index.min()} to {test_data.index.max()}")
        
        # Initialize model trainer
        trainer = ModelTrainer()
        trainer.initialize_models()
        
        # Train models
        logging.info("Training models...")
        training_histories = trainer.train_models(train_data, val_data)
        
        # Evaluate models
        logging.info("Evaluating models...")
        evaluation_results = trainer.evaluate_models(test_data)
        
        # Select best model
        best_model = trainer.select_best_model(evaluation_results)
        logging.info(f"Best performing model: {best_model}")
        
        # Generate future predictions
        logging.info("Generating future predictions...")
        future_predictions = trainer.generate_future_predictions(test_data, years=10)
        
        # Save final results
        results = {
            'evaluation_results': evaluation_results,
            'best_model': best_model,
            'data_version': version_info,
            'training_timestamp': datetime.now().isoformat()
        }
        
        results_path = f'outputs/metrics/final_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)
        
        logging.info(f"Results saved to {results_path}")
        
        # Print final results
        print("\nFinal Results:")
        for model_name, metrics in evaluation_results.items():
            print(f"\n{model_name.upper()} Metrics:")
            for metric_name, value in metrics.items():
                print(f"{metric_name}: {value:.4f}")
        
        print(f"\nBest Model: {best_model.upper()}")
        
    except Exception as e:
        logging.error(f"Pipeline failed: {str(e)}")
        raise
    
    logging.info("Pipeline completed successfully!")

if __name__ == "__main__":
    main()