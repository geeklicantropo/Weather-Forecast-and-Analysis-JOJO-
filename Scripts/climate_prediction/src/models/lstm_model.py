import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.preprocessing import MinMaxScaler
import joblib
import pandas as pd
import dask.dataframe as dd
import dask.array as da
from dask.delayed import delayed
import gc
from datetime import datetime
from typing import Dict, Optional, Tuple, Any, Union
import os
import psutil
from tqdm import tqdm

from .base_model import BaseModel
from ..utils.gpu_manager import gpu_manager

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
if torch.cuda.is_available():
    torch.cuda.set_device(0)

class LSTMNet(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int = 1):
        super(LSTMNet, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.num_heads = 4
        self.adjusted_hidden_size = ((hidden_size + self.num_heads - 1) // self.num_heads) * self.num_heads
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=self.adjusted_hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0
        )
        
        self.attention = nn.MultiheadAttention(
            embed_dim=self.adjusted_hidden_size,
            num_heads=self.num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        self.dropout = nn.Dropout(0.1)
        self.norm = nn.LayerNorm(self.adjusted_hidden_size)
        self.fc = nn.Linear(self.adjusted_hidden_size, output_size)
    
    def forward(self, x: torch.Tensor, hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        lstm_out, hidden = self.lstm(x, hidden)
        torch.nn.utils.clip_grad_norm_(self.lstm.parameters(), max_norm=1.0)
        
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        lstm_out = self.norm(lstm_out + attn_out)
        out = self.dropout(lstm_out[:, -1, :])
        out = self.fc(out)
        
        return out, hidden
    
class LSTMModel(BaseModel):
    def __init__(self, target_variable: str, logger: Any, 
             sequence_length: int = 30, 
             hidden_size: int = 50,
             num_layers: int = 2,
             forecast_horizon: int = 24,
             batch_size: int = 500000):
        super().__init__(target_variable, logger)
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.forecast_horizon = forecast_horizon
        self.scaler = MinMaxScaler()
        self.device = self._initialize_device()
        self.batch_size = batch_size
        self.feature_names = []
        
        self.config = {
            'gpu': {
                'optimization': {
                    'mixed_precision': False
                }
            }
        }
    
    def _initialize_device(self) -> torch.device:
        try:
            if torch.cuda.is_available():
                device = torch.device('cuda')
                torch.cuda.empty_cache()
                test_tensor = torch.zeros(1, device=device)
                del test_tensor
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                self.logger.log_info(f"Using GPU: {torch.cuda.get_device_name(0)} with {gpu_memory:.2f} GB memory")
                return device
            else:
                self.logger.log_info("CUDA not available, using CPU")
                return torch.device('cpu')
        except Exception as e:
            self.logger.log_warning(f"GPU initialization failed: {str(e)}. Using CPU")
            return torch.device('cpu')
    
    def _convert_to_dask(self, df: pd.DataFrame) -> dd.DataFrame:
        if isinstance(df, dd.DataFrame):
            return df
            
        memory_usage = df.memory_usage(deep=True).sum()
        chunk_bytes = 128 * 1024 * 1024
        npartitions = max(1, memory_usage // chunk_bytes)
        
        return dd.from_pandas(df, npartitions=int(npartitions))
    
    def preprocess_data(self, df: Union[pd.DataFrame, TensorDataset]) -> TensorDataset:
        if isinstance(df, TensorDataset):
            return df
            
        try:
            self.logger.log_info(f"Starting LSTM preprocessing for dataset of size {len(df):,} rows")
            temp_dir = "Scripts/climate_prediction/outputs/data/temp"
            os.makedirs(temp_dir, exist_ok=True)
            
            # Convert to Dask DataFrame
            ddf = self._convert_to_dask(df)

            features = ddf.select_dtypes(include=[np.number])
            self.feature_names = features.columns.tolist()
            temp_files = []
            
            with tqdm(total=ddf.npartitions, desc="Processing LSTM batches") as pbar:
                for partition_idx in range(ddf.npartitions):
                    try:
                        # Process current partition
                        partition = ddf.get_partition(partition_idx).compute()
                        scaled_data = self.scaler.fit_transform(partition[self.feature_names])
                        
                        num_sequences = len(scaled_data) - self.sequence_length - self.forecast_horizon + 1
                        if num_sequences <= 0:
                            continue
                        
                        sequence_batches = range(0, num_sequences, self.batch_size)
                        X_parts = []
                        y_parts = []
                        rows_processed = 0
                        
                        with tqdm(sequence_batches, desc=f"Processing sequences for partition {partition_idx + 1}", leave=False) as seq_pbar:
                            for batch_start in seq_pbar:
                                batch_end = min(batch_start + self.batch_size, num_sequences)
                                
                                X = np.lib.stride_tricks.as_strided(
                                    scaled_data[batch_start:batch_start + num_sequences],
                                    shape=((batch_end - batch_start), self.sequence_length, len(self.feature_names)),
                                    strides=(scaled_data.strides[0], scaled_data.strides[0], scaled_data.strides[1])
                                )
                                
                                y_indices = np.arange(self.sequence_length, self.sequence_length + self.forecast_horizon)
                                y = scaled_data[y_indices[0]:y_indices[-1] + 1, self.feature_names.index(self.target_variable)]
                                
                                X_parts.append(X)
                                y_parts.append(y)
                                rows_processed += X.shape[0]
                                
                                if rows_processed >= 500000:
                                    temp_file = os.path.join(temp_dir, f'temp_sequences_{len(temp_files)}.npz')
                                    np.savez_compressed(
                                        temp_file,
                                        X=np.concatenate(X_parts),
                                        y=np.concatenate(y_parts)
                                    )
                                    temp_files.append(temp_file)
                                    X_parts = []
                                    y_parts = []
                                    rows_processed = 0
                                    gc.collect()
                                
                                seq_pbar.set_postfix({'Memory': f'{psutil.Process().memory_info().rss/1e9:.2f}GB'})
                        
                        # Save remaining sequences
                        if X_parts:
                            temp_file = os.path.join(temp_dir, f'temp_sequences_{len(temp_files)}.npz')
                            np.savez_compressed(
                                temp_file,
                                X=np.concatenate(X_parts),
                                y=np.concatenate(y_parts)
                            )
                            temp_files.append(temp_file)
                        
                        # Cleanup
                        del scaled_data, X_parts, y_parts
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            
                    except Exception as e:
                        self.logger.log_error(f"Error processing partition {partition_idx}: {str(e)}")
                        continue
                    
                    pbar.update(1)
            
            # Combine all saved sequences
            self.logger.log_info("Combining processed sequences")
            final_dataset = None

            batch_size = 33  
            num_batches = (len(temp_files) + batch_size - 1) // batch_size
            #print("LEN(TEMP_FILES) AAAAAAAAAAAAAAAAAAAAAA")
            #print(len(temp_files))

            with tqdm(total=num_batches, desc="Loading processed sequences") as pbar:
                for i in range(num_batches):
                    batch_files = temp_files[i * batch_size : (i + 1) * batch_size]
                    X_batch = []
                    y_batch = []

                    for temp_file in batch_files:
                        data = np.load(temp_file)
                        X_batch.append(torch.FloatTensor(data['X']))
                        y_batch.append(torch.FloatTensor(data['y']))

                    X_concat = torch.cat(X_batch)
                    y_concat = torch.cat(y_batch)

                    if final_dataset is None:
                        final_dataset = TensorDataset(X_concat, y_concat)
                    else:
                        final_dataset = TensorDataset(
                            torch.cat((final_dataset.tensors[0], X_concat)),
                            torch.cat((final_dataset.tensors[1], y_concat))
                        )

                    # Clear memory
                    del X_batch, y_batch, X_concat, y_concat
                    gc.collect()

                    pbar.update(1)
            
            # Cleanup
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            os.rmdir(temp_dir)
            
            self.logger.log_info("LSTM preprocessing completed")
            return final_dataset
            
        except Exception as e:
            self.logger.log_error(f"Error in LSTM preprocessing: {str(e)}")
            raise

    def train(self, train_data: pd.DataFrame, validation_data: Optional[pd.DataFrame] = None, 
          epochs: int = 50, progress_callback: callable = None) -> Dict:
        try:
            processed_train = self.preprocess_data(train_data)
            processed_val = self.preprocess_data(validation_data) if validation_data is not None else None
            
            input_size = len(self.feature_names)
            self.model = LSTMNet(
                input_size,
                self.hidden_size,
                self.num_layers,
                self.forecast_horizon
            ).to(self.device)
            
            if torch.cuda.device_count() > 1:
                self.model = nn.DataParallel(self.model)
            
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)
            
            if self.config['gpu']['optimization']['mixed_precision']:
                scaler = torch.cuda.amp.GradScaler()
            
            train_loader = DataLoader(
                processed_train,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=False
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
                total_loss = 0
                
                for X_batch, y_batch in train_loader:
                    optimizer.zero_grad()
                    
                    if self.config['gpu']['optimization']['mixed_precision']:
                        with torch.cuda.amp.autocast():
                            outputs, _ = self.model(X_batch)
                            loss = criterion(outputs, y_batch)
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        outputs, _ = self.model(X_batch)
                        loss = criterion(outputs, y_batch)
                        loss.backward()
                        optimizer.step()
                    
                    total_loss += loss.item()
                
                avg_loss = total_loss / len(train_loader)
                training_history['train_loss'].append(avg_loss)
                
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

                if progress_callback:
                    metrics = {'loss': avg_loss, 'val_loss': val_loss if processed_val else None}
                    progress_callback(epoch, metrics)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join("Scripts/climate_prediction/outputs/models", f"lstm_model_{timestamp}")
            self._save_model_data(save_path)
            
            latest_path = os.path.join("Scripts/climate_prediction/outputs/models", "lstm_model_latest")
            if os.path.exists(latest_path):
                os.remove(latest_path)
            os.symlink(save_path, latest_path)
            
            return training_history
            
        except Exception as e:
            self.logger.log_error(f"Training error: {str(e)}")
            raise

    def _validate(self, validation_data: TensorDataset, criterion: nn.Module) -> float:
        self.model.eval()
        val_loader = DataLoader(
            validation_data,
            batch_size=self.batch_size,
            num_workers=0,
            pin_memory=False
        )
        
        total_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                outputs, _ = self.model(X_batch)
                total_loss += criterion(outputs, y_batch).item()
        
        return total_loss / len(val_loader)

    def predict(self, data: dd.DataFrame, forecast_horizon: Optional[int] = None) -> pd.DataFrame:
        self.model.eval()
        forecast_horizon = forecast_horizon or self.forecast_horizon
        
        try:
            dataset = self.preprocess_data(data)
            dataloader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                num_workers=0,
                pin_memory=False
            )
            
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
                start=data.index[-1],
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
        if hasattr(self, 'best_state'):
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
            'best_state': self.best_state if hasattr(self, 'best_state') else None
        }, f"{path}/lstm_model.pth")
        joblib.dump(self.scaler, f"{path}/scaler.pkl")
    
    def _load_model_data(self, path: str):
        """Load model and associated data."""
        checkpoint = torch.load(f"{path}/lstm_model.pth", map_location=self.device)
        
        self.sequence_length = checkpoint['sequence_length']
        self.hidden_size = checkpoint['hidden_size']
        self.num_layers = checkpoint['num_layers']
        self.forecast_horizon = checkpoint['forecast_horizon']
        self.feature_names = checkpoint['feature_names']
        if 'best_state' in checkpoint:
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
