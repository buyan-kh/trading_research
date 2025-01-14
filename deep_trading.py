import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, Input, concatenate
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import talib

class DeepTradingBot:
    def __init__(self, sequence_length=60):
        self.sequence_length = sequence_length
        self.price_scaler = MinMaxScaler()
        self.feature_scaler = MinMaxScaler()
        self.models = {
            'lstm': None,
            'cnn_lstm': None,
            'attention': None
        }
        
    def create_features(self, df):
        df = df.copy()
        
        # Technical indicators
        df['RSI'] = talib.RSI(df['close'])
        df['MACD'], _, _ = talib.MACD(df['close'])
        df['BB_upper'], _, df['BB_lower'] = talib.BBANDS(df['close'])
        df['ATR'] = talib.ATR(df['high'], df['low'], df['close'])
        
        # Price-based features
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log1p(df['returns'])
        df['volatility'] = df['returns'].rolling(20).std()
        
        # Volume features
        df['volume_ma'] = df['volume'].rolling(20).mean()
        df['volume_std'] = df['volume'].rolling(20).std()
        
        # Moving averages
        for window in [7, 14, 21, 50]:
            df[f'MA_{window}'] = df['close'].rolling(window).mean()
            df[f'price_ma_ratio_{window}'] = df['close'] / df[f'MA_{window}']
        
        return df.fillna(0)
    
    def prepare_sequences(self, data):
        sequences = []
        targets = []
        
        for i in range(len(data) - self.sequence_length):
            sequences.append(data[i:(i + self.sequence_length)])
            targets.append(data['returns'].iloc[i + self.sequence_length])
            
        return np.array(sequences), np.array(targets)
    
    def build_lstm_model(self, input_shape):
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=input_shape),
            Dropout(0.3),
            LSTM(64, return_sequences=True),
            Dropout(0.3),
            LSTM(32),
            Dense(16, activation='relu'),
            Dense(1, activation='tanh')
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return model
    
    def build_cnn_lstm_model(self, input_shape):
        model = Sequential([
            Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
            Conv1D(filters=32, kernel_size=3, activation='relu'),
            LSTM(64, return_sequences=True),
            Dropout(0.3),
            LSTM(32),
            Dense(16, activation='relu'),
            Dense(1, activation='tanh')
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return model
    
    def build_attention_model(self, input_shape):
        inputs = Input(shape=input_shape)
        
        # Attention mechanism
        attention = Dense(1, activation='tanh')(inputs)
        attention = Dense(1, activation='softmax')(attention)
        attention_mul = concatenate([inputs, attention], axis=2)
        
        # LSTM layers
        lstm_out = LSTM(64, return_sequences=True)(attention_mul)
        lstm_out = Dropout(0.3)(lstm_out)
        lstm_out = LSTM(32)(lstm_out)
        
        # Output layers
        dense_out = Dense(16, activation='relu')(lstm_out)
        outputs = Dense(1, activation='tanh')(dense_out)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return model
    
    def train(self, df, epochs=100, batch_size=32):
        # Prepare features
        features_df = self.create_features(df)
        
        # Scale data
        price_data = self.price_scaler.fit_transform(features_df[['close']])
        feature_data = self.feature_scaler.fit_transform(features_df.drop(['close'], axis=1))
        
        # Combine scaled data
        scaled_data = np.hstack((price_data, feature_data))
        
        # Prepare sequences
        X, y = self.prepare_sequences(scaled_data)
        
        # Build and train models
        input_shape = (self.sequence_length, X.shape[2])
        
        self.models['lstm'] = self.build_lstm_model(input_shape)
        self.models['cnn_lstm'] = self.build_cnn_lstm_model(input_shape)
        self.models['attention'] = self.build_attention_model(input_shape)
        
        for name, model in self.models.items():
            print(f"\nTraining {name} model...")
            model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=0.2)
    
    def predict(self, current_data):
        # Prepare features
        features = self.create_features(current_data)
        
        # Scale data
        price_data = self.price_scaler.transform(features[['close']])
        feature_data = self.feature_scaler.transform(features.drop(['close'], axis=1))
        
        # Combine scaled data
        scaled_data = np.hstack((price_data, feature_data))
        
        # Get last sequence
        last_sequence = scaled_data[-self.sequence_length:]
        last_sequence = last_sequence.reshape(1, self.sequence_length, -1)
        
        # Get predictions from all models
        predictions = {}
        for name, model in self.models.items():
            predictions[name] = model.predict(last_sequence)[0][0]
        
        # Ensemble prediction
        ensemble_pred = np.mean(list(predictions.values()))
        
        return {
            'ensemble': ensemble_pred,
            'individual': predictions
        }
    
    def generate_trading_signal(self, predictions, threshold=0.1):
        ensemble_pred = predictions['ensemble']
        
        if ensemble_pred > threshold:
            return 'BUY'
        elif ensemble_pred < -threshold:
            return 'SELL'
        return 'HOLD'

# Example usage
if __name__ == "__main__":
    # Load your data
    df = pd.read_csv('your_price_data.csv')
    
    # Initialize and train the bot
    bot = DeepTradingBot(sequence_length=60)
    bot.train(df, epochs=100)
    
    # Get predictions for recent data
    recent_data = df.tail(61)  # sequence_length + 1
    predictions = bot.predict(recent_data)
    signal = bot.generate_trading_signal(predictions)
    
    print(f"Predictions: {predictions}")
    print(f"Trading Signal: {signal}") 