# src/models/model_comparison.py
import pandas as pd
from .lstm_model import LSTMModel
from .sarima_model import SARIMAModel
from .tft_model import TFTModel
from .model_evaluator import ModelEvaluator
from ..utils.logger import ProgressLogger

def run_model_comparison(train_data, test_data, target_variable, forecast_horizon=24*7):
    logger = ProgressLogger(name="ModelComparison")
    
    try:
        # Initialize models
        models = [
            LSTMModel(target_variable, logger),
            SARIMAModel(target_variable, logger),
            TFTModel(target_variable, logger)
        ]
        
        # Train models
        for model in models:
            logger.log_info(f"Training {model.__class__.__name__}")
            model.train(train_data, test_data)
        
        # Evaluate and compare
        evaluator = ModelEvaluator(logger)
        comparison_results = evaluator.compare_models(models, test_data, forecast_horizon)
        
        # Generate ensemble forecast
        ensemble_predictions = evaluator.generate_forecast_ensemble(
            models, test_data, forecast_horizon
        )
        
        return {
            'comparison_results': comparison_results,
            'ensemble_predictions': ensemble_predictions,
            'models': models
        }
        
    except Exception as e:
        logger.log_error(f"Model comparison failed: {str(e)}")
        raise