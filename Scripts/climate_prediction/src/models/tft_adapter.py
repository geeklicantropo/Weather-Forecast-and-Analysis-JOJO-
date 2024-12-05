from .base_model import BaseModelAdapter
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import QuantileLoss
from pytorch_lightning import Trainer
import pandas as pd

class TFTAdapter(BaseModelAdapter):
    def __init__(self, config):
        super().__init__(config)
        self.max_encoder_length = config['tft']['max_encoder_length']
        self.max_prediction_length = config['tft']['max_prediction_length']
    
    def preprocess(self, data):
        """Prepare data for TFT."""
        # Add time index
        data = data.copy()
        data['time_idx'] = (data.index - data.index.min()).days
        
        # Create TimeSeriesDataSet
        return TimeSeriesDataSet(
            data,
            time_idx="time_idx",
            target=self.target_variable,
            group_ids=["ESTACAO"],  # Using weather station as group
            min_encoder_length=self.max_encoder_length // 2,
            max_encoder_length=self.max_encoder_length,
            min_prediction_length=1,
            max_prediction_length=self.max_prediction_length,
            time_varying_known_reals=["month_sin", "month_cos"],  # Cyclic features
            time_varying_unknown_reals=[self.target_variable],
            target_normalizer="standard"
        )
    
    def train(self, train_data, val_data):
        """Train TFT model."""
        training_data = self.preprocess(train_data)
        validation_data = TimeSeriesDataSet.from_dataset(training_data, val_data)
        
        train_dataloader = training_data.to_dataloader(
            train=True,
            batch_size=self.config['tft']['batch_size']
        )
        val_dataloader = validation_data.to_dataloader(
            train=False,
            batch_size=self.config['tft']['batch_size']
        )
        
        # Initialize TFT model
        self.model = TemporalFusionTransformer.from_dataset(
            training_data,
            learning_rate=self.config['tft']['learning_rate'],
            hidden_size=self.config['tft']['hidden_size'],
            attention_head_size=self.config['tft']['attention_head_size'],
            dropout=self.config['tft']['dropout'],
            hidden_continuous_size=self.config['tft']['hidden_continuous_size'],
            loss=QuantileLoss(),
            log_interval=10,
            reduce_on_plateau_patience=4
        )
        
        # Train
        trainer = Trainer(
            max_epochs=self.config['tft']['epochs'],
            gpus=1 if torch.cuda.is_available() else 0
        )
        
        trainer.fit(
            self.model,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader
        )
        
        return trainer
    
    def predict(self, data, prediction_steps):
        """Generate predictions."""
        dataloader = self.preprocess(data).to_dataloader(train=False)
        predictions = self.model.predict(dataloader)
        return predictions.numpy()