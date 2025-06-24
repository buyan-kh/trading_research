import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class AdvancedLSTMTrainer:
    def __init__(self, data, look_back=60, features=None):
        self.data = data
        self.look_back = look_back
        self.features = features or ['Close', 'Volume', 'High', 'Low', 'Open']
        # Filter features that actually exist in data
        self.features = [f for f in self.features if f in data.columns]
        self.scaler = RobustScaler()  # More robust to outliers
        self.model = None
        self.history = None
        self.X_train = None
        self.X_val = None
        self.y_train = None
        self.y_val = None

    def prepare_data(self, validation_split=0.2):
        """Prepare multivariate time series data with proper scaling"""
        # Select available features
        feature_data = self.data[self.features].values
        
        # Scale the data
        scaled_data = self.scaler.fit_transform(feature_data)
        
        # Create sequences
        X, y = [], []
        for i in range(self.look_back, len(scaled_data)):
            X.append(scaled_data[i-self.look_back:i])  # All features
            y.append(scaled_data[i, 0])  # Predict Close price (first feature)
        
        X, y = np.array(X), np.array(y)
        
        # Split data chronologically (no shuffling for time series)
        split_idx = int(len(X) * (1 - validation_split))
        self.X_train, self.X_val = X[:split_idx], X[split_idx:]
        self.y_train, self.y_val = y[:split_idx], y[split_idx:]
        
        print(f"Training shape: {self.X_train.shape}, Validation shape: {self.X_val.shape}")
        print(f"Using features: {self.features}")
        
        return self.X_train, self.y_train

    def build_model(self, units=[128, 64, 32], dropout_rate=0.3, bidirectional=True):
        """Build advanced LSTM model with modern architecture"""
        input_shape = (self.look_back, len(self.features))
        
        self.model = Sequential()
        
        # First LSTM layer
        if bidirectional:
            self.model.add(Bidirectional(LSTM(units[0], return_sequences=True), 
                                        input_shape=input_shape))
        else:
            self.model.add(LSTM(units[0], return_sequences=True, input_shape=input_shape))
        
        self.model.add(BatchNormalization())
        self.model.add(Dropout(dropout_rate))
        
        # Second LSTM layer
        if bidirectional:
            self.model.add(Bidirectional(LSTM(units[1], return_sequences=True)))
        else:
            self.model.add(LSTM(units[1], return_sequences=True))
        
        self.model.add(BatchNormalization())
        self.model.add(Dropout(dropout_rate))
        
        # Third LSTM layer
        if bidirectional:
            self.model.add(Bidirectional(LSTM(units[2], return_sequences=False)))
        else:
            self.model.add(LSTM(units[2], return_sequences=False))
        
        self.model.add(BatchNormalization())
        self.model.add(Dropout(dropout_rate))
        
        # Dense layers
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dropout(dropout_rate/2))
        self.model.add(Dense(32, activation='relu'))
        self.model.add(Dropout(dropout_rate/2))
        self.model.add(Dense(1))
        
        # Compile with advanced optimizer
        optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
        self.model.compile(optimizer=optimizer, loss='huber', metrics=['mae', 'mse'])
        
        print(f"Model built with input shape: {input_shape}")
        print(f"Total parameters: {self.model.count_params():,}")

    def train_model(self, epochs=100, batch_size=32, patience=15, save_best=True):
        """Train model with advanced callbacks and monitoring"""
        if self.X_train is None:
            raise ValueError("Data not prepared. Call prepare_data() first.")
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=patience//2,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        if save_best:
            callbacks.append(
                ModelCheckpoint(
                    'best_lstm_model.h5',
                    monitor='val_loss',
                    save_best_only=True,
                    verbose=1
                )
            )
        
        # Train the model
        print("Starting training...")
        self.history = self.model.fit(
            self.X_train, self.y_train,
            validation_data=(self.X_val, self.y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        print("Training completed!")
        return self.history

    def predict(self, X_test=None, return_confidence=False):
        """Make predictions with optional confidence intervals"""
        if X_test is None:
            X_test = self.X_val
        
        predictions = self.model.predict(X_test, verbose=0)
        
        if return_confidence:
            # Monte Carlo Dropout for uncertainty estimation
            mc_predictions = []
            for _ in range(100):
                mc_pred = self.model.predict(X_test, verbose=0)
                mc_predictions.append(mc_pred)
            
            mc_predictions = np.array(mc_predictions)
            mean_pred = np.mean(mc_predictions, axis=0)
            std_pred = np.std(mc_predictions, axis=0)
            
            return mean_pred, std_pred
        
        return predictions
    
    def evaluate_model(self):
        """Evaluate model performance with multiple metrics"""
        if self.X_val is None:
            raise ValueError("No validation data available")
        
        predictions = self.predict(self.X_val)
        
        # Calculate metrics
        mae = np.mean(np.abs(predictions.flatten() - self.y_val))
        rmse = np.sqrt(np.mean((predictions.flatten() - self.y_val) ** 2))
        
        # Directional accuracy
        actual_direction = np.sign(np.diff(self.y_val))
        pred_direction = np.sign(np.diff(predictions.flatten()))
        directional_accuracy = np.mean(actual_direction == pred_direction)
        
        results = {
            'mae': mae,
            'rmse': rmse,
            'directional_accuracy': directional_accuracy
        }
        
        print(f"Model Performance:")
        print(f"MAE: {mae:.6f}")
        print(f"RMSE: {rmse:.6f}")
        print(f"Directional Accuracy: {directional_accuracy:.2%}")
        
        return results
    
    def plot_training_history(self):
        """Plot training history"""
        if self.history is None:
            print("No training history available")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss
        ax1.plot(self.history.history['loss'], label='Training Loss')
        ax1.plot(self.history.history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        
        # MAE
        ax2.plot(self.history.history['mae'], label='Training MAE')
        ax2.plot(self.history.history['val_mae'], label='Validation MAE')
        ax2.set_title('Model MAE')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MAE')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def predict_future(self, n_steps=5):
        """Predict multiple steps into the future"""
        if self.X_val is None:
            raise ValueError("No data available for prediction")
        
        # Start with the last sequence from validation data
        last_sequence = self.X_val[-1].copy()
        predictions = []
        
        for _ in range(n_steps):
            # Predict next value
            next_pred = self.model.predict(last_sequence.reshape(1, -1, len(self.features)), verbose=0)
            predictions.append(next_pred[0, 0])
            
            # Update sequence: remove first element, add prediction
            last_sequence = np.roll(last_sequence, -1, axis=0)
            last_sequence[-1, 0] = next_pred[0, 0]  # Update Close price
        
        return np.array(predictions)
    
    def save_model(self, filepath='lstm_model.h5'):
        """Save the trained model"""
        if self.model is None:
            raise ValueError("No model to save")
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='lstm_model.h5'):
        """Load a pre-trained model"""
        from tensorflow.keras.models import load_model
        self.model = load_model(filepath)
        print(f"Model loaded from {filepath}")