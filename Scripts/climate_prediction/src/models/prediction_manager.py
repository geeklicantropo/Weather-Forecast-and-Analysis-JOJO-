# Scripts/climate_prediction/src/models/prediction_manager.py
import dask.dataframe as dd
import pandas as pd
import numpy as np
import torch
from typing import Dict, List
import os
from datetime import datetime
import gc
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from ..utils.gpu_manager import gpu_manager
from ..visualization.visualization_manager import VisualizationManager

class PredictionManager:
    def __init__(self, logger, output_dir="outputs/predictions"):
        self.logger = logger
        self.output_dir = output_dir
        self.device = gpu_manager.get_device()
        self.visualizer = VisualizationManager(logger)
        os.makedirs(output_dir, exist_ok=True)

    def prepare_data_for_prediction(self, df: pd.DataFrame) -> dd.DataFrame:
        """Convert pandas DataFrame to Dask DataFrame with optimal chunking."""
        try:
            # Calculate optimal chunk size based on available memory
            available_memory = gpu_manager.get_memory_info()['free'] if torch.cuda.is_available() else None
            chunk_size = gpu_manager.get_optimal_batch_size()

            # Convert to Dask DataFrame
            ddf = dd.from_pandas(df, npartitions=max(1, len(df) // chunk_size))
            
            self.logger.log_info(f"Converted to Dask DataFrame with {ddf.npartitions} partitions")
            return ddf

        except Exception as e:
            self.logger.log_error(f"Error preparing data for prediction: {str(e)}")
            raise

    def run_distributed_predictions(self, models: Dict, data: dd.DataFrame) -> Dict:
        """Run predictions for all models using Dask and GPU."""
        predictions = {}
        
        try:
            for name, model in models.items():
                self.logger.log_info(f"Running predictions for {name} model")
                
                try:
                    # Clear GPU memory before each model
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    # Run predictions
                    with gpu_manager.memory_monitor():
                        model_predictions = model.predict(data)
                        
                    # Save predictions
                    self._save_predictions(model_predictions, name)
                    predictions[name] = model_predictions
                    
                except Exception as e:
                    self.logger.log_error(f"Error in {name} predictions: {str(e)}")
                    continue
                
                # Force garbage collection
                gc.collect()
            
            # Generate comparison visualizations
            if predictions:
                self._generate_comparison_plots(predictions)
            
            return predictions
            
        except Exception as e:
            self.logger.log_error(f"Error in distributed predictions: {str(e)}")
            raise

    def _save_predictions(self, predictions: pd.DataFrame, model_name: str):
        """Save predictions to disk."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{model_name}_predictions_{timestamp}.csv.gz"
        filepath = os.path.join(self.output_dir, filename)
        
        predictions.to_csv(filepath, compression='gzip')
        self.logger.log_info(f"Saved predictions for {model_name} to {filepath}")

    def _generate_comparison_plots(self, predictions: Dict):
        """Generate comparison plots for all models."""
        try:
            # Compare predictions
            self.visualizer.plot_predictions(
                actual_values=None,  # For future predictions
                predictions=predictions,
                title="Model Predictions Comparison"
            )
            
            # Model differences
            self._plot_model_differences(predictions)
            
            # Uncertainty comparison
            self._plot_uncertainty_comparison(predictions)
            
        except Exception as e:
            self.logger.log_error(f"Error generating comparison plots: {str(e)}")

    def _plot_model_differences(self, predictions: Dict):
        """Plot differences between model predictions."""
        try:
            models = list(predictions.keys())
            diff_matrix = np.zeros((len(models), len(models)))
            
            for i, model1 in enumerate(models):
                for j, model2 in enumerate(models):
                    if i < j:
                        diff = np.mean(np.abs(
                            predictions[model1]['forecast'].values - 
                            predictions[model2]['forecast'].values
                        ))
                        diff_matrix[i, j] = diff
                        diff_matrix[j, i] = diff
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(
                diff_matrix,
                xticklabels=models,
                yticklabels=models,
                annot=True,
                fmt='.2f',
                cmap='coolwarm'
            )
            plt.title('Average Absolute Differences Between Models')
            
            # Save plot
            plt.savefig(os.path.join(self.output_dir, 'model_differences.png'))
            plt.close()
            
        except Exception as e:
            self.logger.log_error(f"Error plotting model differences: {str(e)}")

    def _plot_uncertainty_comparison(self, predictions: Dict):
        """Plot uncertainty comparison between models."""
        try:
            plt.figure(figsize=(15, 8))
            
            for name, pred in predictions.items():
                if 'lower_bound' in pred and 'upper_bound' in pred:
                    uncertainty = pred['upper_bound'] - pred['lower_bound']
                    plt.plot(pred.index, uncertainty, label=f'{name} Uncertainty')
            
            plt.title('Prediction Uncertainty Comparison')
            plt.xlabel('Date')
            plt.ylabel('Uncertainty Range')
            plt.legend()
            
            # Save plot
            plt.savefig(os.path.join(self.output_dir, 'uncertainty_comparison.png'))
            plt.close()
            
        except Exception as e:
            self.logger.log_error(f"Error plotting uncertainty comparison: {str(e)}")

    def compute_model_agreement(self, predictions: Dict) -> pd.DataFrame:
        """Compute agreement between model predictions."""
        try:
            all_forecasts = pd.DataFrame()
            
            for name, pred in predictions.items():
                all_forecasts[name] = pred['forecast']
            
            # Calculate standard deviation between models
            all_forecasts['model_std'] = all_forecasts.std(axis=1)
            
            # Calculate mean prediction
            all_forecasts['ensemble_mean'] = all_forecasts[predictions.keys()].mean(axis=1)
            
            return all_forecasts
            
        except Exception as e:
            self.logger.log_error(f"Error computing model agreement: {str(e)}")
            return pd.DataFrame()