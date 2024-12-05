# src/models/lstm_model.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from .base_model import BaseModel

class LSTMModel(BaseModel):
    def __init__(self, sequence_length=10, hidden_size=50, num_layers=2):
        super().__init__(name="LSTM")
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.scaler = MinMaxScaler()
        
    def preprocess_data(self, df):
        scaled_data = self.scaler.fit_transform(df[[self.target_variable]])
        sequences = []
        targets = []
        
        for i in range(len(scaled_data) - self.sequence_length):
            sequences.append(scaled_data[i:i+self.sequence_length])
            targets.append(scaled_data[i+self.sequence_length])
        
        return torch.FloatTensor(sequences), torch.FloatTensor(targets)
    
    def train(self, train_data, epochs=20, batch_size=32):
        X_train, y_train = self.preprocess_data(train_data)
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        self.model = nn.LSTM(
            input_size=1,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True
        )
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters())
        
        for epoch in range(epochs):
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs, _ = self.model(batch_X)
                loss = criterion(outputs[:, -1, :], batch_y)
                loss.backward()
                optimizer.step()
    
    def predict(self, test_data):
        self.model.eval()
        X_test, _ = self.preprocess_data(test_data)
        with torch.no_grad():
            predictions, _ = self.model(X_test)
            predictions = predictions[:, -1, :]
        return self.scaler.inverse_transform(predictions.numpy())