#src/models/lstm_model.py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import joblib
import pandas as pd
import dask.dataframe as dd
import dask.array as da
from dask.delayed import delayed
import gc
from datetime import datetime
from typing import Dict, Optional, Tuple, Any
import os
import psutil
from tqdm import tqdm

from .base_model import BaseModel
from ..utils.gpu_manager import gpu_manager

class LSTMNet(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int = 1):
        super(LSTMNet, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )
        
        # Output layers
        self.dropout = nn.Dropout(0.1)
        self.norm = nn.LayerNorm(hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x: torch.Tensor, hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # LSTM forward pass
        lstm_out, hidden = self.lstm(x, hidden)
        
        # Apply attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Add & Norm
        lstm_out = self.norm(lstm_out + attn_out)
        
        # Get last output
        out = self.dropout(lstm_out[:, -1, :])
        out = self.fc(out)
        
        return out, hidden

class LSTMModel(BaseModel):
    def __init__(self, target_variable: str, logger: Any, 
                 sequence_length: int = 30, 
                 hidden_size: int = 50,
                 num_layers: int = 2,
                 forecast_horizon: int = 24):
        super().__init__(target_variable, logger)
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.forecast_horizon = forecast_horizon
        self.scaler = MinMaxScaler()
        self.device = gpu_manager.get_device()
        self.batch_size = gpu_manager.get_optimal_batch_size()
        self.feature_names = []
        
    def _convert_to_dask(self, df: pd.DataFrame) -> dd.DataFrame:
        """Convert pandas DataFrame to Dask DataFrame if not already."""
        if isinstance(df, dd.DataFrame):
            return df
        
        # Calculate memory usage and optimal chunk size
        memory_usage = df.memory_usage(deep=True).sum()
        chunk_bytes = 128 * 1024 * 1024  # 128MB chunks
        npartitions = max(1, memory_usage // chunk_bytes)
        
        return dd.from_pandas(df, npartitions=int(npartitions))
    
    def preprocess_data(self, df: pd.DataFrame) -> TensorDataset:
        try:
            self.logger.log_info(f"Starting LSTM preprocessing for dataset of size {len(df):,} rows")
            
            # Convert to Dask DataFrame
            ddf = self._convert_to_dask(df)
            features = ddf.select_dtypes(include=[np.number])
            self.feature_names = features.columns.tolist()
            
            # Repartition for optimal batch size
            batch_size = 10000
            n_partitions = max(1, len(df) // batch_size)
            features = features.repartition(npartitions=n_partitions)
            
            all_X, all_y = [], []
            X, y = [], []
            total_sequences = 0
            
            with tqdm(total=features.npartitions, desc="Processing LSTM batches") as pbar:
                for i, chunk_df in enumerate(features.partitions):
                    chunk_data = chunk_df.compute()
                    
                    if len(chunk_data) <= self.sequence_length:
                        continue
                    
                    scaled_chunk = self.scaler.fit_transform(chunk_data)
                    
                    for j in range(len(scaled_chunk) - self.sequence_length - self.forecast_horizon + 1):
                        X.append(scaled_chunk[j:(j + self.sequence_length)])
                        y.append(scaled_chunk[
                            (j + self.sequence_length):(j + self.sequence_length + self.forecast_horizon),
                            self.feature_names.index(self.target_variable)
                        ])
                        
                        if len(X) >= batch_size:
                            all_X.append(torch.FloatTensor(np.array(X, dtype=np.float32)).to(self.device))
                            all_y.append(torch.FloatTensor(np.array(y, dtype=np.float32)).to(self.device))
                            total_sequences += len(X)
                            X, y = [], []
                    
                    pbar.update(1)
                    gc.collect()
            
            if X:
                all_X.append(torch.FloatTensor(np.array(X, dtype=np.float32)).to(self.device))
                all_y.append(torch.FloatTensor(np.array(y, dtype=np.float32)).to(self.device))
                total_sequences += len(X)
            
            X_tensor = torch.cat(all_X)
            y_tensor = torch.cat(all_y)
            
            return TensorDataset(X_tensor, y_tensor)
            
        except Exception as e:
            self.logger.error(f"Error in LSTM preprocessing: {str(e)}")
            raise
    
    def train(self, train_data: dd.DataFrame, validation_data: Optional[dd.DataFrame] = None, 
             epochs: int = 50) -> Dict:
        """Train LSTM model with processed Dask data."""
        try:
            # Process data first
            processed_train = self.preprocess_data(train_data)
            processed_val = self.preprocess_data(validation_data) if validation_data is not None else None
            
            input_size = len(self.feature_names)
            self.model = LSTMNet(
                input_size, 
                self.hidden_size, 
                self.num_layers,
                self.forecast_horizon
            ).to(self.device)
            
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=5, verbose=True
            )
            
            train_loader = DataLoader(
                processed_train,
                batch_size=self.batch_size,
                shuffle=True,
                pin_memory=True
            )
            
            training_history = {
                'train_loss': [],
                'val_loss': [],
                'best_epoch': 0
            }
            
            best_val_loss = float('inf')
            patience = 10
            patience_counter = 0
            
            for epoch in range(epochs):
                self.model.train()
                train_loss = 0
                
                for X_batch, y_batch in train_loader:
                    optimizer.zero_grad()
                    outputs, _ = self.model(X_batch)
                    loss = criterion(outputs, y_batch)
                    loss.backward()
                    
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    train_loss += loss.item()
                
                avg_train_loss = train_loss / len(train_loader)
                training_history['train_loss'].append(avg_train_loss)
                
                if processed_val is not None:
                    val_loss = self._validate(processed_val, criterion)
                    training_history['val_loss'].append(val_loss)
                    scheduler.step(val_loss)
                    
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        training_history['best_epoch'] = epoch
                        self._save_best_model()
                        patience_counter = 0
                    else:
                        patience_counter += 1
                    
                    if patience_counter >= patience:
                        self.logger.log_info(f"Early stopping at epoch {epoch}")
                        self._load_best_model()
                        break
            
            return training_history
            
        except Exception as e:
            self.logger.log_error(f"Training error: {str(e)}")
            raise
    
    def _validate(self, validation_data: TensorDataset, criterion: nn.Module) -> float:
        """Validate the model."""
        self.model.eval()
        val_loader = DataLoader(
            validation_data,
            batch_size=self.batch_size,
            pin_memory=True
        )
        val_loss = 0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs, _ = self.model(X_batch)
                val_loss += criterion(outputs, y_batch).item()
        
        return val_loss / len(val_loader)
    
    def predict(self, data: dd.DataFrame, forecast_horizon: Optional[int] = None) -> pd.DataFrame:
        """Generate predictions with uncertainty estimation using Dask."""
        self.model.eval()
        forecast_horizon = forecast_horizon or self.forecast_horizon
        
        try:
            # Process data in parallel using Dask
            dataset = self.preprocess_data(data)
            dataloader = DataLoader(dataset, batch_size=self.batch_size, pin_memory=True)
            
            predictions = []
            prediction_intervals = []
            
            with torch.no_grad():
                for X_batch, _ in dataloader:
                    mc_predictions = []
                    self.model.train()
                    
                    for _ in range(100):
                        output, _ = self.model(X_batch)
                        mc_predictions.append(output.cpu().numpy())
                    
                    mc_predictions = np.array(mc_predictions)
                    mean_prediction = np.mean(mc_predictions, axis=0)
                    std_prediction = np.std(mc_predictions, axis=0)
                    
                    predictions.append(mean_prediction)
                    prediction_intervals.append([
                        mean_prediction - 1.96 * std_prediction,
                        mean_prediction + 1.96 * std_prediction
                    ])
            
            predictions = np.vstack(predictions)
            prediction_intervals = np.array(prediction_intervals)
            
            dates = pd.date_range(
                start=data.index.compute()[-1],
                periods=len(predictions) * self.forecast_horizon,
                freq='H'
            )
            
            results = pd.DataFrame(index=dates)
            results['forecast'] = self._inverse_transform(predictions.flatten())
            results['lower_bound'] = self._inverse_transform(prediction_intervals[:, :, 0].flatten())
            results['upper_bound'] = self._inverse_transform(prediction_intervals[:, :, 1].flatten())
            
            return results
            
        except Exception as e:
            self.logger.log_error(f"Prediction error: {str(e)}")
            raise
    
    def _inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Inverse transform scaled data."""
        temp_data = np.zeros((len(data), len(self.feature_names)))
        temp_data[:, self.feature_names.index(self.target_variable)] = data
        return self.scaler.inverse_transform(temp_data)[:, self.feature_names.index(self.target_variable)]
    
    def _save_best_model(self):
        """Save best model state."""
        self.best_state = {
            'model_state_dict': self.model.state_dict(),
            'scaler_state': self.scaler
        }
    
    def _load_best_model(self):
        """Load best model state."""
        self.model.load_state_dict(self.best_state['model_state_dict'])
        self.scaler = self.best_state['scaler_state']
    
    def _save_model_data(self, path: str):
        """Save model and associated data."""
        os.makedirs(path, exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'sequence_length': self.sequence_length,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'forecast_horizon': self.forecast_horizon,
            'feature_names': self.feature_names,
            'best_state': self.best_state
        }, f"{path}/lstm_model.pth")
        joblib.dump(self.scaler, f"{path}/scaler.pkl")
    
    def _load_model_data(self, path: str):
        """Load model and associated data."""
        checkpoint = torch.load(f"{path}/lstm_model.pth")
        self.sequence_length = checkpoint['sequence_length']
        self.hidden_size = checkpoint['hidden_size']
        self.num_layers = checkpoint['num_layers']
        self.forecast_horizon = checkpoint['forecast_horizon']
        self.feature_names = checkpoint['feature_names']
        self.best_state = checkpoint['best_state']
        

        input_size = len(self.feature_names)
        self.model = LSTMNet(
            input_size,
            self.hidden_size,
            self.num_layers,
            self.forecast_horizon
        ).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.scaler = joblib.load(f"{path}/scaler.pkl")