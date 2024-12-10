import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Dataset, IterableDataset
from sklearn.preprocessing import MinMaxScaler
import joblib
import pandas as pd
import dask.dataframe as dd
import dask.array as da
from dask.delayed import delayed
import gc
from datetime import datetime
from typing import Dict, Optional, Tuple, Any, Union, List
import os
from pathlib import Path
import psutil
from tqdm import tqdm

from .base_model import BaseModel
from ..utils.gpu_manager import gpu_manager
from ..data_processing.sequence_dataset import SequenceDataset

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
        self.temp_files = []
        self.model_artifacts = []
        
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
    
    def preprocess_data(self, df: Union[pd.DataFrame, TensorDataset, SequenceDataset]) -> Union[TensorDataset, SequenceDataset]:
        """Preprocess data for LSTM model."""
        if isinstance(df, (TensorDataset, SequenceDataset)):
            return df
            
        if df is None:
            return None
            
        try:
            self.logger.log_info(f"Starting LSTM preprocessing for dataset of size {len(df):,} rows")
            temp_dir = Path("Scripts/climate_prediction/outputs/data/temp")
            temp_dir.mkdir(parents=True, exist_ok=True)
            
            # Convert to Dask DataFrame if needed
            if isinstance(df, pd.DataFrame):
                memory_usage = df.memory_usage(deep=True).sum()
                chunk_bytes = 128 * 1024 * 1024
                npartitions = max(1, memory_usage // chunk_bytes)
                ddf = dd.from_pandas(df, npartitions=int(npartitions))
            else:
                ddf = df

            features = ddf.select_dtypes(include=[np.number])
            self.feature_names = features.columns.tolist()
            sequence_files = []
            
            with tqdm(total=ddf.npartitions, desc="Processing LSTM batches") as pbar:
                for partition_idx in range(ddf.npartitions):
                    try:
                        partition = ddf.get_partition(partition_idx).compute()
                        scaled_data = self.scaler.fit_transform(partition[self.feature_names])
                        
                        num_sequences = len(scaled_data) - self.sequence_length + 1
                        if num_sequences <= 0:
                            continue
                        
                        sequence_batches = range(0, num_sequences, self.batch_size)
                        current_sequences = []
                        current_size = 0
                        max_size = 1024 * 1024 * 1024  # 1GB buffer
                        
                        with tqdm(sequence_batches, desc=f"Creating sequences for partition {partition_idx + 1}", leave=False) as seq_pbar:
                            for batch_start in seq_pbar:
                                batch_end = min(batch_start + self.batch_size, num_sequences)
                                
                                X = np.lib.stride_tricks.as_strided(
                                    scaled_data[batch_start:batch_start + num_sequences],
                                    shape=((batch_end - batch_start), self.sequence_length, len(self.feature_names)),
                                    strides=(scaled_data.strides[0], scaled_data.strides[0], scaled_data.strides[1])
                                )
                                
                                current_sequences.append(X)
                                current_size += X.nbytes
                                
                                if current_size >= max_size:
                                    self._save_sequences_batch(
                                        current_sequences, 
                                        temp_dir, 
                                        len(sequence_files)
                                    )
                                    sequence_files.append(
                                        temp_dir / f'sequences_{len(sequence_files)}.npz'
                                    )
                                    current_sequences = []
                                    current_size = 0
                                    gc.collect()
                                
                                seq_pbar.set_postfix({'Memory': f'{psutil.Process().memory_info().rss/1e9:.2f}GB'})
                        
                        # Save any remaining sequences
                        if current_sequences:
                            self._save_sequences_batch(
                                current_sequences, 
                                temp_dir, 
                                len(sequence_files)
                            )
                            sequence_files.append(
                                temp_dir / f'sequences_{len(sequence_files)}.npz'
                            )
                        
                        # Cleanup
                        del scaled_data, current_sequences
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            
                    except Exception as e:
                        self.logger.log_error(f"Error processing partition {partition_idx}: {str(e)}")
                        continue
                    
                    pbar.update(1)
            
            # Create and return streaming dataset
            return SequenceDataset(
                str(temp_dir),
                self.sequence_length,
                self.batch_size,
                device=self.device
            )
            
        except Exception as e:
            self.logger.log_error(f"Error in LSTM preprocessing: {str(e)}")
            raise
            
        finally:
            try:
                self._cleanup_temp_files(temp_dir)
            except Exception as e:
                self.logger.log_warning(f"Cleanup warning: {str(e)}")

    def _save_sequences_batch(self, sequences: List[np.ndarray], temp_dir: Path, batch_idx: int):
        """Save sequence batch to compressed npz file."""
        sequences_array = np.concatenate(sequences, axis=0)
        file_path = temp_dir / f'sequences_{batch_idx}.npz'
        np.savez_compressed(file_path, X=sequences_array.astype(np.float32))
        self.temp_files.append(file_path)

    def _cleanup_temp_files(self, temp_dir: Path):
        """Clean up temporary files and force garbage collection."""
        try:
            # Remove temporary npz files
            for file in self.temp_files:
                try:
                    if os.path.exists(file):
                        os.remove(file)
                except Exception as e:
                    self.logger.log_warning(f"Failed to delete {file}: {str(e)}")
            
            # Remove memmap file if exists
            memmap_file = temp_dir / 'sequences.mmap'
            if memmap_file.exists():
                try:
                    memmap_file.unlink()
                except Exception as e:
                    self.logger.log_warning(f"Failed to delete memmap file: {str(e)}")
            
            # Try to remove temp directory
            try:
                if temp_dir.exists():
                    temp_dir.rmdir()
            except Exception as e:
                self.logger.log_warning(f"Failed to remove temp directory: {str(e)}")
            
            # Clear lists
            self.temp_files = []
            
            # Force garbage collection
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            self.logger.log_error(f"Error during cleanup: {str(e)}")

    def cleanup(self):
        """Cleanup all resources used by the model."""
        try:
            # Clear GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Remove temp files
            for file in self.temp_files:
                try:
                    if os.path.exists(file):
                        os.remove(file)
                except Exception as e:
                    self.logger.log_warning(f"Failed to remove temp file {file}: {str(e)}")
            
            # Clear model artifacts
            for artifact in self.model_artifacts:
                if isinstance(artifact, (torch.Tensor, nn.Module)):
                    del artifact
            
            # Reset lists
            self.temp_files = []
            self.model_artifacts = []
            
            # Force garbage collection
            gc.collect()
            
        except Exception as e:
            self.logger.log_error(f"Error during cleanup: {str(e)}")

    def train(self, train_data: pd.DataFrame, validation_data: Optional[pd.DataFrame] = None, 
          epochs: int = 50, progress_callback: callable = None) -> Dict:
        """Train model with memory management."""
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
                scaler = torch.amp.GradScaler()
            
            train_loader = DataLoader(
                processed_train,
                batch_size=self.batch_size,
                num_workers=4,
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
                
                for X_batch, y_batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}', leave=False):
                    optimizer.zero_grad()
                    
                    if self.config['gpu']['optimization']['mixed_precision']:
                        with torch.amp.autocast():
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
            
        finally:
            # Cleanup resources
            self.cleanup()
            if processed_train:
                del processed_train
            if processed_val:
                del processed_val
            gc.collect()

    def _validate(self, validation_data: TensorDataset, criterion: nn.Module) -> float:
        self.model.eval()
        val_loader = DataLoader(
            validation_data,
            batch_size=None,  # Batch size handled by dataset
            num_workers=4,
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
        try:
            dataset = self.preprocess_data(data)
            dataloader = DataLoader(
                dataset,
                batch_size=None,  # Batch size handled by dataset
                num_workers=4,
                pin_memory=False
            )
            
            predictions = []
            prediction_intervals = []
            
            with torch.no_grad():
                for X_batch, _ in tqdm(dataloader, desc="Generating predictions"):
                    mc_predictions = []
                    self.model.train()  # Enable dropout for MC sampling
                    
                    for _ in range(100):  # Monte Carlo samples
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
        
        finally:
            self.cleanup()
    
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