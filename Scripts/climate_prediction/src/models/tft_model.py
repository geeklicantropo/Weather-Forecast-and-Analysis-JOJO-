# src/models/tft_model.py
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import QuantileLoss
from pytorch_lightning import Trainer
from .base_model import BaseModel

class TFTModel(BaseModel):
    def __init__(self, max_encoder_length=30, max_prediction_length=10):
        super().__init__(name="TFT")
        self.max_encoder_length = max_encoder_length
        self.max_prediction_length = max_prediction_length
    
    def preprocess_data(self, df):
        # Add time index
        df = df.copy()
        df['time_idx'] = (df.index - df.index.min()).days
        
        # Create training dataset
        training = TimeSeriesDataSet(
            df,
            time_idx="time_idx",
            target=self.target_variable,
            group_ids=["LATITUDE", "LONGITUDE"],  # Group by location
            min_encoder_length=self.max_encoder_length // 2,
            max_encoder_length=self.max_encoder_length,
            min_prediction_length=1,
            max_prediction_length=self.max_prediction_length,
            time_varying_known_reals=["time_idx"],
            time_varying_unknown_reals=[self.target_variable],
            target_normalizer="std"
        )
        return training
    
    def train(self, train_data):
        training = self.preprocess_data(train_data)
        train_dataloader = training.to_dataloader(train=True, batch_size=64)
        
        self.model = TemporalFusionTransformer.from_dataset(
            training,
            learning_rate=0.03,
            hidden_size=16,
            attention_head_size=1,
            dropout=0.1,
            hidden_continuous_size=8,
            loss=QuantileLoss(),
            log_interval=10,
        )
        
        trainer = Trainer(max_epochs=30)
        trainer.fit(self.model, train_dataloader)
    
    def predict(self, test_data):
        test_dataset = self.preprocess_data(test_data)
        predictions = self.model.predict(test_dataset)
        return predictions.numpy()