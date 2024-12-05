from .base_model import BaseModelAdapter
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import numpy as np

class LSTMAdapter(BaseModelAdapter):
    def __init__(self, config):
        super().__init__(config)
        self.sequence_length = config['lstm']['sequence_length']
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self._build_model()
    
    def _build_model(self):
        """Build LSTM model architecture."""
        self.model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(self.sequence_length, 1)),
            LSTM(50),
            Dense(1)
        ])
        self.model.compile(optimizer=Adam(learning_rate=self.config['lstm']['learning_rate']),
                          loss='mse')
    
    def preprocess(self, data):
        """Prepare data for LSTM."""
        values = self.scaler.fit_transform(data[[self.target_variable]].values)
        X, y = [], []
        for i in range(len(values) - self.sequence_length):
            X.append(values[i:(i + self.sequence_length), 0])
            y.append(values[i + self.sequence_length, 0])
        return np.array(X), np.array(y)
    
    def train(self, train_data, val_data):
        """Train LSTM model."""
        X_train, y_train = self.preprocess(train_data)
        X_val, y_val = self.preprocess(val_data)
        
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        X_val = np.reshape(X_val, (X_val.shape[0], X_val.shape[1], 1))
        
        history = self.model.fit(
            X_train, y_train,
            epochs=self.config['lstm']['epochs'],
            batch_size=self.config['lstm']['batch_size'],
            validation_data=(X_val, y_val),
            verbose=1
        )
        return history
    
    def predict(self, data, prediction_steps):
        """Generate predictions."""
        X_test, _ = self.preprocess(data)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        predictions = self.model.predict(X_test)
        return self.scaler.inverse_transform(predictions.reshape(-1, 1))