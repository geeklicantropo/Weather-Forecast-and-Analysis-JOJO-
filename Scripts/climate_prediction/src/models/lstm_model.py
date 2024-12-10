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
import multiprocessing as mp

from .base_model import BaseModel
from ..utils.gpu_manager import gpu_manager
from ..data_processing.sequence_dataset import SequenceDataset
from torch.optim import lr_scheduler

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def setup_gpu():
    if torch.cuda.is_available():
        try:
            # Initialize CUDA first
            device = torch.device('cuda')
            torch.cuda.set_device(device)
            
            # Test GPU memory
            test_tensor = torch.zeros((1,), device=device)
            del test_tensor
            torch.cuda.empty_cache()
            
            # Now safe to set memory fraction
            torch.cuda.set_per_process_memory_fraction(0.7)
            torch.backends.cudnn.benchmark = True
            return device
        except Exception:
            return torch.device('cpu')
    return torch.device('cpu')

device = setup_gpu()

mp.set_start_method('spawn', force=True)

class LSTMNet(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int = 1):
        super(LSTMNet, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Adjust hidden size for multi-head attention
        self.num_heads = 4
        self.adjusted_hidden_size = ((hidden_size + self.num_heads - 1) // self.num_heads) * self.num_heads
        
        # Use smaller hidden size and add dropout
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=self.adjusted_hidden_size // 2,  
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0,
            bidirectional=True  # Use bidirectional LSTM
        )
        
        self.attention = nn.MultiheadAttention(
            embed_dim=self.adjusted_hidden_size,
            num_heads=self.num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        self.dropout = nn.Dropout(0.2)
        self.norm = nn.LayerNorm(self.adjusted_hidden_size)
        self.fc = nn.Linear(self.adjusted_hidden_size, output_size)
    
    def forward(self, x: torch.Tensor, hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # Apply gradient clipping
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.5)
        
        lstm_out, hidden = self.lstm(x, hidden)
        
        # Apply attention mechanism
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        lstm_out = self.norm(lstm_out + attn_out)
        
        # Use only last output
        out = self.dropout(lstm_out[:, -1, :])
        out = self.fc(out)
        
        return out, hidden
    
class LSTMModel(BaseModel):
    def __init__(self, target_variable: str, logger: Any, 
             sequence_length: int = 30, 
             hidden_size: int = 32,  
             num_layers: int = 2,
             forecast_horizon: int = 30,  
             batch_size: int = 128):  
        super().__init__(target_variable, logger)
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.forecast_horizon = forecast_horizon
        self.scaler = MinMaxScaler()  # Data scaler
        self.device = self._initialize_device()
        self.batch_size = batch_size
        self.feature_names = []
        self.temp_files = []
        self.model_artifacts = []
        
        # Separate GradScaler for mixed precision training
        self.grad_scaler = torch.amp.GradScaler()
        
    def _initialize_device(self) -> torch.device:
        try:
            if torch.cuda.is_available():
                device = torch.device('cuda')
                torch.cuda.empty_cache()
                
                # Test GPU memory
                test_tensor = torch.zeros(1, device=device)
                del test_tensor
                torch.cuda.empty_cache()
                
                return device
            else:
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
        """Preprocess data for LSTM model using streaming approach."""
        if isinstance(df, (TensorDataset, SequenceDataset)):
            return df
                
        if df is None:
            return None
                
        try:
            self.logger.log_info(f"Starting LSTM preprocessing for dataset of size {len(df):,} rows")
            temp_dir = Path("Scripts/climate_prediction/outputs/data/temp")
            sequence_chunks_dir = temp_dir / 'sequence_chunks'
            sequence_chunks_dir.mkdir(parents=True, exist_ok=True)
            
            # Calculate total expected partitions and chunks
            if isinstance(df, pd.DataFrame):
                memory_usage = df.memory_usage(deep=True).sum()
            else:  
                memory_usage = df.memory_usage(deep=True).sum().compute()
                
            chunk_bytes = 128 * 1024 * 1024  # 128MB chunks
            npartitions = max(1, int(memory_usage // chunk_bytes))
            
            # Check existing chunks and get completion status
            existing_chunks = sorted(list(sequence_chunks_dir.glob('chunk_*.npz')))
            completed_partitions = set()
            completed_batches = set()
            
            for chunk_file in existing_chunks:
                parts = chunk_file.stem.split('_')
                partition_idx = int(parts[1])
                batch_idx = int(parts[2])
                completed_batches.add((partition_idx, batch_idx))
                if not any(p > partition_idx for p, _ in completed_batches):
                    completed_partitions.add(partition_idx)

            if len(completed_partitions) == npartitions:
                self.logger.log_info(f"Found complete set of {len(existing_chunks)} sequence chunks")
                return SequenceDataset(
                    str(temp_dir),
                    self.sequence_length,
                    self.batch_size,
                    device=self.device
                )
                
            # Convert to Dask DataFrame if needed
            if isinstance(df, pd.DataFrame):
                ddf = dd.from_pandas(df, npartitions=npartitions)
            else:
                ddf = df.repartition(npartitions=npartitions)

            features = ddf.select_dtypes(include=[np.number])
            self.feature_names = features.columns.tolist()
                
            # Process missing partitions
            with tqdm(total=npartitions, desc="Processing LSTM batches", initial=len(completed_partitions)) as pbar:
                for partition_idx in range(npartitions):
                    if partition_idx in completed_partitions:
                        continue
                        
                    try:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            
                        partition = ddf.get_partition(partition_idx).compute()
                        scaled_data = self.scaler.fit_transform(partition[self.feature_names])
                            
                        num_sequences = len(scaled_data) - self.sequence_length + 1
                        if num_sequences <= 0:
                            continue
                            
                        batch_size = min(self.batch_size, 100000)
                        sequence_batches = range(0, num_sequences, batch_size)
                            
                        with tqdm(sequence_batches, desc=f"Creating sequences for partition {partition_idx + 1}", 
                                leave=False) as seq_pbar:
                            for batch_idx, batch_start in enumerate(seq_pbar):
                                if (partition_idx, batch_idx) in completed_batches:
                                    seq_pbar.update(1)
                                    continue
                                    
                                batch_end = min(batch_start + batch_size, num_sequences)
                                    
                                sequences = np.lib.stride_tricks.as_strided(
                                    scaled_data[batch_start:batch_start + num_sequences],
                                    shape=((batch_end - batch_start), self.sequence_length, len(self.feature_names)),
                                    strides=(scaled_data.strides[0], scaled_data.strides[0], scaled_data.strides[1])
                                )
                                    
                                chunk_file = sequence_chunks_dir / f'chunk_{partition_idx:04d}_{batch_idx:04d}.npz'
                                np.savez_compressed(chunk_file, X=sequences.astype(np.float32))
                                    
                                seq_pbar.set_postfix({
                                    'Memory': f'{psutil.Process().memory_info().rss/1e9:.2f}GB',
                                    'GPU': f'{torch.cuda.memory_allocated()/1e9:.2f}GB' if torch.cuda.is_available() else 'N/A'
                                })
                                    
                                del sequences
                                gc.collect()
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                            
                        del scaled_data
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            
                    except Exception as e:
                        self.logger.log_error(f"Error processing partition {partition_idx}: {str(e)}")
                        continue
                            
                    pbar.update(1)
                
            # Final verification
            final_chunks = list(sequence_chunks_dir.glob('chunk_*.npz'))
            if not final_chunks:
                raise RuntimeError(f"No sequence chunks were created in {sequence_chunks_dir}")
                    
            self.logger.log_info(f"Created/Found total of {len(final_chunks)} sequence chunks")
                
            return SequenceDataset(
                str(temp_dir),
                self.sequence_length,
                self.batch_size,
                device=self.device
            )
                
        except Exception as e:
            self.logger.log_error(f"Error in LSTM preprocessing: {str(e)}")
            raise
            
    def _save_sequences_batch(self, sequences: List[np.ndarray], temp_dir: Path, batch_idx: int):
        """Save sequence batch to compressed npz file."""
        sequences_array = np.concatenate(sequences, axis=0)
        file_path = temp_dir / f'sequences_{batch_idx}.npz'
        np.savez_compressed(file_path, X=sequences_array.astype(np.float32))
        self.temp_files.append(file_path)

    def _cleanup_temp_files(self, temp_dir: Path):
        """Clean up temporary files while preserving sequence chunks."""
        try:
            # Remove temporary npz files (except sequence chunks)
            for file in self.temp_files:
                if file.parent.name != 'sequence_chunks':  # Preserve sequence chunks
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
            
            # Clear lists
            self.temp_files = []
            
            # Force garbage collection
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            self.logger.log_error(f"Error during cleanup: {str(e)}")

    def cleanup(self):
        """Clean up resources while preserving sequence chunks."""
        try:
            if hasattr(self, 'model'):
                del self.model
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            gc.collect()
            
        except Exception as e:
            self.logger.log_error(f"Error during cleanup: {str(e)}")

    def train(self, train_data: pd.DataFrame, validation_data: Optional[pd.DataFrame] = None, 
              epochs: int = 30, progress_callback: callable = None) -> Dict:
        try:
            # Initialize progress bar
            with tqdm(total=3, desc="Training LSTM Model", leave=True) as main_pbar:
                # Data preprocessing
                main_pbar.set_description("Preprocessing data")
                processed_train = self.preprocess_data(train_data)
                processed_val = self.preprocess_data(validation_data) if validation_data is not None else None
                main_pbar.update(1)

                # Model initialization
                main_pbar.set_description("Initializing model")
                input_size = len(self.feature_names)
                self.model = LSTMNet(
                    input_size,
                    self.hidden_size,
                    self.num_layers,
                    self.forecast_horizon
                ).to(self.device)
                
                criterion = nn.MSELoss().to(self.device)
                optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
                scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

                main_pbar.update(1)

                train_loader = DataLoader(
                    processed_train,
                    batch_size=None,  #Batch size handled by dataset
                    num_workers=2,
                    pin_memory=True if torch.cuda.is_available() else False
                )
                
                training_history = {
                    'train_loss': [],
                    'val_loss': [],
                    'best_epoch': 0
                }
                
                best_val_loss = float('inf')
                patience = 10
                patience_counter = 0
                
                # Training progress
                main_pbar.set_description("Training epochs")
                with tqdm(total=epochs, desc="Epochs", position=0, leave=True) as epoch_pbar:
                    for epoch in range(epochs):
                        self.model.train()
                        total_loss = 0
                        batches_processed = 0
                        
                        # Enable automatic mixed precision
                        with torch.amp.autocast(device_type=self.device.type):
                            for X_batch, y_batch in train_loader:
                                X_batch = X_batch.to(self.device)
                                y_batch = y_batch.to(self.device)
                                
                                optimizer.zero_grad(set_to_none=True)
                                outputs, _ = self.model(X_batch)
                                loss = criterion(outputs, y_batch)
                                
                                # Use gradient scaler
                                self.grad_scaler.scale(loss).backward()
                                self.grad_scaler.step(optimizer)
                                self.grad_scaler.update()
                                
                                current_loss = loss.item()
                                total_loss += current_loss
                                batches_processed += 1
                                
                                # Free memory
                                del X_batch, y_batch, outputs, loss
                                if batches_processed % 5 == 0:
                                    torch.cuda.empty_cache()
                        
                        avg_loss = total_loss / batches_processed
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
                            
                        epoch_pbar.update(1)
                
                main_pbar.update(1)
                return training_history
                
        except Exception as e:
            self.logger.log_error(f"Training error: {str(e)}")
            raise
        
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.cleanup()

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