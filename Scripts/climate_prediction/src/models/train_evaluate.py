# climate_prediction/src/models/train_evaluate.py
import os
import sys
from pathlib import Path

# Add the project root directory to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.models.lstm_model import LSTMModel
from src.models.sarima_model import SARIMAModel
from src.models.tft_model import TFTModel
from src.models.model_evaluator import ModelEvaluator
import pandas as pd

def main():
    # Create necessary directories if they don't exist
    output_dirs = ['outputs/metrics', 'outputs/plots', 'outputs/models']
    for dir_path in output_dirs:
        os.makedirs(dir_path, exist_ok=True)
    
    # Load processed data
    processed_data_path = 'outputs/data/processed_data.parquet'
    if not os.path.exists(processed_data_path):
        print("Please run the data processing pipeline first (main.py)")
        sys.exit(1)
    
    df = pd.read_parquet(processed_data_path)
    
    # Split data into train and test sets
    train_size = int(len(df) * 0.8)
    train_data = df[:train_size]
    test_data = df[train_size:]
    
    # Initialize models
    models = [
        LSTMModel(),
        SARIMAModel(),
        TFTModel()
    ]
    
    # Evaluate models
    evaluator = ModelEvaluator(models, train_data, test_data)
    evaluator.evaluate_all_models()
    
    # Compare model performance
    comparison = evaluator.compare_models()
    print("\nModel Comparison:")
    print(comparison)
    
    # Plot predictions
    evaluator.plot_predictions(save_path='outputs/plots/model_comparison.png')
    
    # Save results
    comparison.to_csv('outputs/metrics/model_comparison.csv')

if __name__ == "__main__":
    main()