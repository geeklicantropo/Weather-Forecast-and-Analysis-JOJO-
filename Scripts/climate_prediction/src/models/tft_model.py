# src/models/tft_model.py
import numpy as np
import pandas as pd
import torch
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer, NaNLabelEncoder
from pytorch_forecasting.metrics import QuantileLoss, RMSE, MAE, MAPE
from pytorch_lightning import Trainer, callbacks
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
import joblib
from src.models.base_model import BaseModel
from src.utils.gpu_manager import gpu_manager

class CustomMetrics:
    def __init__(self):
        self.metrics = {
            'rmse': RMSE(),
            'mae': MAE(),
            'mape': MAPE(),
            'quantile': QuantileLoss()
        }

class TFTModel(BaseModel):
    def __init__(self, target_variable, logger, 
                max_prediction_length=24*7,  # 7 days ahead
                max_encoder_length=24*30,    # 30 days history
                learning_rate=0.001,
                hidden_size=32,
                attention_head_size=4,
                dropout=0.1,
                hidden_continuous_size=16,
                loss_fn=None):
        super().__init__(target_variable, logger)
        self.max_prediction_length = max_prediction_length
        self.max_encoder_length = max_encoder_length
        self.batch_size = gpu_manager.get_optimal_batch_size()
        self.learning_rate = learning_rate
        self.hidden_size = hidden_size
        self.attention_head_size = attention_head_size
        self.dropout = dropout
        self.hidden_continuous_size = hidden_continuous_size
        self.loss_fn = loss_fn or CustomMetrics().metrics['quantile']
        self.metrics = CustomMetrics().metrics
        
    def _create_features(self, df):
        """Create time-based features"""
        df = df.copy()
        df['time_idx'] = (df.index - df.index.min()).total_seconds() // 3600
        
        # Time features
        df['hour'] = df.index.hour
        df['day'] = df.index.day
        df['month'] = df.index.month
        df['day_of_week'] = df.index.dayofweek
        df['week_of_year'] = df.index.isocalendar().week
        
        # Special features
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['hour_of_week'] = df['hour'] + df['day_of_week'] * 24
        
        return df
        
    def preprocess_data(self, df):
        try:
            df = self._create_features(df)
            
            # Select features
            static_categoricals = ['ESTACAO', 'UF']
            static_reals = ['LATITUDE', 'LONGITUDE', 'ALTITUDE']
            time_varying_known_categoricals = [
                'hour', 'day', 'month', 'day_of_week', 'is_weekend'
            ]
            time_varying_known_reals = [
                'time_idx', 'hour_of_week', 'week_of_year'
            ]
            time_varying_unknown_reals = [
                self.target_variable,
                'PRECIPITACÃO TOTAL HORÁRIO MM',
                'PRESSAO ATMOSFERICA AO NIVEL DA ESTACAO HORARIA MB',
                'UMIDADE RELATIVA DO AR HORARIA %',
                'VENTO VELOCIDADE HORARIA M/S'
            ]
            
            self.training = TimeSeriesDataSet(
                df,
                time_idx="time_idx",
                target=self.target_variable,
                group_ids=static_categoricals,
                min_encoder_length=self.max_encoder_length // 2,
                max_encoder_length=self.max_encoder_length,
                min_prediction_length=1,
                max_prediction_length=self.max_prediction_length,
                static_categoricals=static_categoricals,
                static_reals=static_reals,
                time_varying_known_categoricals=time_varying_known_categoricals,
                time_varying_known_reals=time_varying_known_reals,
                time_varying_unknown_reals=time_varying_unknown_reals,
                target_normalizer=GroupNormalizer(
                    groups=static_categoricals,
                    transformation="softplus",
                    center=True
                ),
                add_relative_time_idx=True,
                add_target_scales=True,
                add_encoder_length=True,
                allow_missing_timesteps=True
            )
            
            return self.training
            
        except Exception as e:
            self.logger.log_error(f"Preprocessing error: {str(e)}")
            raise
            
    def _create_dataloaders(self, training, validation=None):
        train_dataloader = training.to_dataloader(
            train=True,
            batch_size=self.batch_size,
            num_workers=4,
            pin_memory=True
        )
        
        if validation is not None:
            val_dataloader = validation.to_dataloader(
                train=False,
                batch_size=self.batch_size,
                num_workers=4,
                pin_memory=True
            )
            return train_dataloader, val_dataloader
            
        return train_dataloader, None
        
    def train(self, train_data, validation_data=None, max_epochs=100):
        try:
            training = self.preprocess_data(train_data)
            validation = self.preprocess_data(validation_data) if validation_data is not None else None
            
            train_dataloader, val_dataloader = self._create_dataloaders(training, validation)
            
            # Initialize model
            self.model = TemporalFusionTransformer.from_dataset(
                training,
                learning_rate=self.learning_rate,
                hidden_size=self.hidden_size,
                attention_head_size=self.attention_head_size,
                dropout=self.dropout,
                hidden_continuous_size=self.hidden_continuous_size,
                loss=self.loss_fn,
                logging_metrics=self.metrics
            )
            
            # Setup callbacks
            early_stop_callback = EarlyStopping(
                monitor="val_loss",
                min_delta=1e-4,
                patience=10,
                verbose=False,
                mode="min"
            )
            
            lr_monitor = LearningRateMonitor(logging_interval='epoch')
            
            trainer = Trainer(
                max_epochs=max_epochs,
                accelerator='gpu' if torch.cuda.is_available() else 'cpu',
                devices=1,
                gradient_clip_val=0.1,
                limit_train_batches=50,
                callbacks=[early_stop_callback, lr_monitor],
                enable_progress_bar=True,
                logger=True
            )
            
            # Train
            trainer.fit(
                self.model,
                train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader
            )
            
            # Store feature importance
            self.feature_importance = self.model.interpret_output(
                train_dataloader.dataset[:100],
                reduction="sum"
            )["attention"]
            
        except Exception as e:
            self.logger.log_error(f"Training error: {str(e)}")
            raise
            
    def predict(self, data, forecast_horizon=None):
        try:
            test_dataset = self.preprocess_data(data)
            test_dataloader = test_dataset.to_dataloader(
                train=False,
                batch_size=self.batch_size,
                num_workers=4,
                pin_memory=True
            )
            
            predictions = self.model.predict(
                test_dataloader,
                mode="prediction",
                return_x=True,
                trainer_kwargs=dict(accelerator='gpu' if torch.cuda.is_available() else 'cpu')
            )
            
            # Get prediction intervals
            predictions_dist = self.model.predict(
                test_dataloader,
                mode="quantiles",
                return_x=True,
                trainer_kwargs=dict(accelerator='gpu' if torch.cuda.is_available() else 'cpu')
            )
            
            results = pd.DataFrame({
                'prediction': predictions.numpy().flatten(),
                'lower_bound': predictions_dist[:, 0].numpy().flatten(),
                'upper_bound': predictions_dist[:, -1].numpy().flatten()
            }, index=data.index[-len(predictions):])
            
            return results
            
        except Exception as e:
            self.logger.log_error(f"Prediction error: {str(e)}")
            raise
            
    def _save_model_data(self, path):
        # Save model parameters and state
        model_data = {
            'max_prediction_length': self.max_prediction_length,
            'max_encoder_length': self.max_encoder_length,
            'learning_rate': self.learning_rate,
            'hidden_size': self.hidden_size,
            'attention_head_size': self.attention_head_size,
            'dropout': self.dropout,
            'hidden_continuous_size': self.hidden_continuous_size,
            'training_config': self.training.get_parameters(),
            'feature_importance': self.feature_importance
        }
        
        # Save model state
        torch.save(self.model.state_dict(), f"{path}/tft_model.pth")
        # Save other parameters
        joblib.dump(model_data, f"{path}/tft_params.pkl")
        
    def _load_model_data(self, path):
        # Load parameters
        model_data = joblib.load(f"{path}/tft_params.pkl")
        
        # Restore parameters
        self.max_prediction_length = model_data['max_prediction_length']
        self.max_encoder_length = model_data['max_encoder_length']
        self.learning_rate = model_data['learning_rate']
        self.hidden_size = model_data['hidden_size']
        self.attention_head_size = model_data['attention_head_size']
        self.dropout = model_data['dropout']
        self.hidden_continuous_size = model_data['hidden_continuous_size']
        self.feature_importance = model_data['feature_importance']
        
        # Recreate training dataset configuration
        self.training = TimeSeriesDataSet.from_parameters(model_data['training_config'])
        
        # Initialize and load model
        self.model = TemporalFusionTransformer.from_dataset(
            self.training,
            learning_rate=self.learning_rate,
            hidden_size=self.hidden_size,
            attention_head_size=self.attention_head_size,
            dropout=self.dropout,
            hidden_continuous_size=self.hidden_continuous_size,
            loss=self.loss_fn,
            logging_metrics=self.metrics
        )
        self.model.load_state_dict(torch.load(f"{path}/tft_model.pth"))