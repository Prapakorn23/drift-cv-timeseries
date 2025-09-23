import numpy as np
import pandas as pd
from typing import Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, SimpleRNN, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

class SequenceGenerator:
    """Class to create sequence data for RNN-based models and handle scaling."""
    def __init__(self, sequence_length: int = 30):
        self.sequence_length = sequence_length
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        
    def create_sequences(self, X: pd.DataFrame, y: pd.Series, fit_scalers: bool = True):
        X_scaled = self.scaler_X.fit_transform(X) if fit_scalers else self.scaler_X.transform(X)
        y_scaled = self.scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten() if fit_scalers else self.scaler_y.transform(y.values.reshape(-1, 1)).flatten()
        
        X_seq, y_seq = [], []
        for i in range(self.sequence_length, len(X_scaled)):
            X_seq.append(X_scaled[i-self.sequence_length:i])
            y_seq.append(y_scaled[i])
        
        return np.array(X_seq), np.array(y_seq)
    
    def inverse_transform_y(self, y_scaled):
        return self.scaler_y.inverse_transform(y_scaled.reshape(-1, 1)).flatten()

class RNNRegressor:
    """Universal RNN Regressor supporting SimpleRNN, LSTM, and GRU models."""
    def __init__(self, model_type: str = 'LSTM', sequence_length: int = 30, units: int = 128, dropout_rate: float = 0.3, learning_rate: float = 0.0005, epochs: int = 200, batch_size: int = 32, verbose: int = 0):
        self.model_type = model_type.upper()
        self.sequence_length = sequence_length
        self.units = units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.model = None
        self.seq_generator = SequenceGenerator(sequence_length)
        if self.model_type not in ['RNN', 'LSTM', 'GRU']:
            raise ValueError("model_type must be 'RNN', 'LSTM', or 'GRU'")
        
    def _get_layer_type(self):
        if self.model_type == 'RNN': return SimpleRNN
        elif self.model_type == 'LSTM': return LSTM
        elif self.model_type == 'GRU': return GRU
        
    def _build_model(self, input_shape):
        LayerType = self._get_layer_type()
        model = Sequential([
            LayerType(self.units, return_sequences=True, input_shape=input_shape),
            Dropout(self.dropout_rate),
            LayerType(self.units // 2, return_sequences=True), # Add layer
            Dropout(self.dropout_rate),
            LayerType(self.units // 4),
            Dropout(self.dropout_rate),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mse', metrics=['mae'])
        return model
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        X_seq, y_seq = self.seq_generator.create_sequences(X, y, fit_scalers=True)
        if len(X_seq) == 0: raise ValueError("Not enough data to create sequences")
        self.model = self._build_model((X_seq.shape[1], X_seq.shape[2]))
        early_stopping = EarlyStopping(monitor='loss', patience=20, restore_best_weights=True) # Increase patience
        self.model.fit(X_seq, y_seq, epochs=self.epochs, batch_size=self.batch_size, callbacks=[early_stopping], verbose=self.verbose)
        return self
    
    def predict(self, X: pd.DataFrame):
        if self.model is None: raise ValueError("Model not fitted yet")
        # Ensure X is correctly transformed
        X_scaled = self.seq_generator.scaler_X.transform(X)
        X_seq, _ = self.seq_generator.create_sequences(X, pd.Series([0] * len(X)), fit_scalers=False)
        if len(X_seq) == 0: return np.array([])
        y_pred_scaled = self.model.predict(X_seq, verbose=0)
        y_pred = self.seq_generator.inverse_transform_y(y_pred_scaled)
        return y_pred

class LinearRegressionModel:
    """Simple linear regression wrapper for compatibility."""
    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept
        self.model = LinearRegression(fit_intercept=fit_intercept)
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series):
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten()
        self.model.fit(X_scaled, y_scaled)
        self.is_fitted = True

    def predict(self, X: pd.DataFrame):
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        X_scaled = self.scaler_X.transform(X)
        y_pred_scaled = self.model.predict(X_scaled)
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        return y_pred
