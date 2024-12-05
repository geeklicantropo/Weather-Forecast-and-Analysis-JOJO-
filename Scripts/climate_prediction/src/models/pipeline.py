# src/models/pipeline.py
import pandas as pd
from typing import Dict, List
import os
from datetime import datetime

from .lstm_model import LSTMModel
from .sarima_model import SARIMAModel
from .tft_model import TFTModel
from .model_evaluator import ModelEvaluator
from ..visualization.visualization_manager import VisualizationManager

class ModelPipeline:
    def __init__(self, target_variable, logger, output_dir="outputs"):
        self.target_variable = target_variable
        self.logger = logger
        self.output_dir = output_dir
        self.models = {}
        self.evaluator = ModelEvaluator(logger)
        self.visualizer = VisualizationManager(logger)
        
    def prepare_data(self, df: pd.DataFrame, test_size: float = 0.2):
        cutoff = int(len(df) * (1 - test_size))
        train_data = df.iloc[:cutoff]
        test_data = df.iloc[cutoff:]
        return train_data, test_data
        
    def train_models(self, train_data: pd.DataFrame, validation_data: pd.DataFrame):
        models = {
            'lstm': LSTMModel(self.target_variable, self.logger),
            'sarima': SARIMAModel(self.target_variable, self.logger),
            'tft': TFTModel(self.target_variable, self.logger)
        }
        
        for name, model in models.items():
            try:
                self.logger.log_info(f"Training {name} model...")
                processed_train = model.preprocess_data(train_data)
                processed_val = model.preprocess_data(validation_data) if validation_data is not None else None
                model.train(processed_train, processed_val)
                self.models[name] = model
            except Exception as e:
                self.logger.log_error(f"Error training {name} model: {str(e)}")
                
    def evaluate_all_models(self, test_data: pd.DataFrame):
        results = self.evaluator.evaluate_models(self.models, test_data)
        self._save_evaluation_results(results)
        return results
    
    def generate_forecasts(self, horizon: int = 3650):  # 10 years
        forecasts = {}
        for name, model in self.models.items():
            try:
                forecast = model.predict(data=None, forecast_horizon=horizon)
                forecasts[name] = forecast
            except Exception as e:
                self.logger.log_error(f"Error generating forecast for {name}: {str(e)}")
        
        return self._create_ensemble_forecast(forecasts)
    
    def _create_ensemble_forecast(self, forecasts: Dict):
        weights = self._calculate_weights()
        return self.evaluator.forecast_ensemble(weights)
    
    def _calculate_weights(self) -> Dict:
        comparison = self.evaluator.compare_models()
        rmse_scores = comparison['rmse']
        inverse_rmse = 1 / rmse_scores
        weights = inverse_rmse / inverse_rmse.sum()
        return weights.to_dict()
    
    def generate_visualizations(self, data: pd.DataFrame, forecasts: Dict):
        viz_dir = f"{self.output_dir}/plots"
        os.makedirs(viz_dir, exist_ok=True)
        
        # Historical performance
        self.visualizer.plot_predictions(
            data[self.target_variable],
            {name: model.predict(data) for name, model in self.models.items()},
            save_path=f"{viz_dir}/model_predictions.png"
        )
        
        # Metrics comparison
        self.visualizer.plot_metrics_comparison(
            self.evaluator.compare_models(),
            save_path=f"{viz_dir}/metrics_comparison.png"
        )
        
        # Residuals analysis
        residuals = {
            name: data[self.target_variable].values - model.predict(data)
            for name, model in self.models.items()
        }
        self.visualizer.plot_residuals_analysis(
            residuals,
            save_path=f"{viz_dir}/residuals_analysis.png"
        )
        
        # Future forecasts
        historical = data[self.target_variable]
        self.visualizer.plot_forecast(
            historical,
            forecasts['ensemble'],
            save_path=f"{viz_dir}/forecast.png"
        )
        
        # Seasonal decomposition
        self.visualizer.plot_seasonal_decomposition(
            data[self.target_variable],
            save_path=f"{viz_dir}/seasonal_decomposition.png"
        )
    
    def _save_evaluation_results(self, results: Dict):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_df = pd.DataFrame.from_dict(results, orient='index')
        results_df.to_csv(f"{self.output_dir}/evaluation_results_{timestamp}.csv")
        
    def run_pipeline(self, df: pd.DataFrame):
        train_data, test_data = self.prepare_data(df)
        self.train_models(train_data, test_data)
        evaluation_results = self.evaluate_all_models(test_data)
        forecasts = self.generate_forecasts()
        self.generate_visualizations(df, forecasts)
        return evaluation_results, forecasts