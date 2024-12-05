# src/models/lstm_model.py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import joblib
import pandas as pd
from .base_model import BaseModel
from ..utils.gpu_manager import gpu_manager

class LSTMNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size=1):
        super(LSTMNet, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, hidden=None):
        lstm_out, hidden = self.lstm(x, hidden)
        out = self.dropout(lstm_out[:, -1, :])
        out = self.fc(out)
        return out, hidden

class LSTMModel(BaseModel):
    def __init__(self, target_variable, logger, sequence_length=30, hidden_size=50, 
                 num_layers=2, forecast_horizon=24):
        super().__init__(target_variable, logger)
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.forecast_horizon = forecast_horizon
        self.scaler = MinMaxScaler()
        self.device = gpu_manager.get_device()
        self.batch_size = gpu_manager.get_optimal_batch_size()
        
    def preprocess_data(self, df):
        features = df.select_dtypes(include=[np.number]).copy()
        self.feature_names = features.columns.tolist()
        
        # Scale features
        scaled_data = self.scaler.fit_transform(features)
        sequences = []
        targets = []
        
        # Multi-step sequence creation
        for i in range(len(scaled_data) - self.sequence_length - self.forecast_horizon + 1):
            sequences.append(scaled_data[i:(i + self.sequence_length)])
            targets.append(scaled_data[
                (i + self.sequence_length):(i + self.sequence_length + self.forecast_horizon),
                features.columns.get_loc(self.target_variable)
            ])
        
        X = torch.FloatTensor(np.array(sequences)).to(self.device)
        y = torch.FloatTensor(np.array(targets)).to(self.device)
        
        return TensorDataset(X, y)
    
    def train(self, train_data, validation_data=None, epochs=50):
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
            train_data, 
            batch_size=self.batch_size, 
            shuffle=True,
            pin_memory=True
        )
        
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
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            
            # Validation phase
            if validation_data is not None:
                val_loss = self._validate(validation_data, criterion)
                scheduler.step(val_loss)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self._save_best_model()
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    self.logger.log_info(f"Early stopping at epoch {epoch}")
                    self._load_best_model()
                    break
                
                self.logger.log_info(
                    f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, Val Loss = {val_loss:.4f}"
                )
    
    def _validate(self, validation_data, criterion):
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
    
    def predict(self, data, forecast_horizon=None):
        self.model.eval()
        forecast_horizon = forecast_horizon or self.forecast_horizon
        predictions = []
        prediction_intervals = []
        
        with torch.no_grad():
            for sequence in data:
                sequence = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
                
                # Monte Carlo Dropout for uncertainty estimation
                mc_predictions = []
                self.model.train()  # Enable dropout
                
                for _ in range(100):  # Number of Monte Carlo samples
                    output, _ = self.model(sequence)
                    mc_predictions.append(output.cpu().numpy())
                
                mc_predictions = np.array(mc_predictions)
                mean_prediction = np.mean(mc_predictions, axis=0)
                std_prediction = np.std(mc_predictions, axis=0)
                
                predictions.append(mean_prediction)
                prediction_intervals.append([
                    mean_prediction - 1.96 * std_prediction,  # Lower bound (95% CI)
                    mean_prediction + 1.96 * std_prediction   # Upper bound (95% CI)
                ])
        
        # Inverse transform predictions and intervals
        predictions = np.array(predictions).reshape(-1, self.forecast_horizon)
        prediction_intervals = np.array(prediction_intervals)
        
        results = pd.DataFrame()
        results['forecast'] = self._inverse_transform(predictions)
        results['lower_bound'] = self._inverse_transform(prediction_intervals[:, :, 0])
        results['upper_bound'] = self._inverse_transform(prediction_intervals[:, :, 1])
        
        return results
    
    def _inverse_transform(self, data):
        temp_data = np.zeros((len(data), len(self.feature_names)))
        temp_data[:, self.feature_names.index(self.target_variable)] = data.flatten()
        return self.scaler.inverse_transform(temp_data)[:, self.feature_names.index(self.target_variable)]
    
    def _save_best_model(self):
        self.best_state = {
            'model_state_dict': self.model.state_dict(),
            'scaler_state': self.scaler
        }
    
    def _load_best_model(self):
        self.model.load_state_dict(self.best_state['model_state_dict'])
        self.scaler = self.best_state['scaler_state']
    
    def _save_model_data(self, path):
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
    
    def _load_model_data(self, path):
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