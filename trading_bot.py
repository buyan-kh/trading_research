import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import talib

class TradingBot:
    def __init__(self):
        self.rf_model = RandomForestClassifier()
        self.gb_model = GradientBoostingClassifier()
        self.xgb_model = xgb.XGBClassifier()
        self.lstm_model = None
        self.scaler = StandardScaler()
        
    def create_features(self, df):
        df['RSI'] = talib.RSI(df['close'])
        df['MACD'], df['MACD_signal'], _ = talib.MACD(df['close'])
        df['BB_upper'], df['BB_middle'], df['BB_lower'] = talib.BBANDS(df['close'])
        df['ATR'] = talib.ATR(df['high'], df['low'], df['close'])
        df['OBV'] = talib.OBV(df['close'], df['volume'])
        
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(window=20).std()
        
        for window in [7, 14, 21]:
            df[f'MA_{window}'] = df['close'].rolling(window=window).mean()
            df[f'std_{window}'] = df['close'].rolling(window=window).std()
        
        return df
    
    def create_lstm_model(self, input_shape):
        model = Sequential([
            LSTM(100, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model
    
    def prepare_data(self, df):
        features = self.create_features(df)
        features = features.dropna()
        
        X = features[['RSI', 'MACD', 'BB_upper', 'BB_lower', 'ATR', 'OBV', 
                     'volatility', 'MA_7', 'MA_14', 'MA_21']]
        y = (features['close'].shift(-1) > features['close']).astype(int)[:-1]
        
        return X[:-1], y
    
    def train_models(self, df):
        X, y = self.prepare_data(df)
        X_scaled = self.scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)
        
        # Train traditional ML models
        self.rf_model.fit(X_train, y_train)
        self.gb_model.fit(X_train, y_train)
        self.xgb_model.fit(X_train, y_train)
        
        # Prepare and train LSTM
        X_lstm = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
        self.lstm_model = self.create_lstm_model((1, X_scaled.shape[1]))
        self.lstm_model.fit(X_lstm, y, epochs=50, batch_size=32, validation_split=0.2)
        
    def predict(self, current_data):
        features = self.create_features(current_data)
        X = features[['RSI', 'MACD', 'BB_upper', 'BB_lower', 'ATR', 'OBV', 
                     'volatility', 'MA_7', 'MA_14', 'MA_21']].iloc[-1:]
        X_scaled = self.scaler.transform(X)
        
        # Get predictions from all models
        rf_pred = self.rf_model.predict_proba(X_scaled)[0][1]
        gb_pred = self.gb_model.predict_proba(X_scaled)[0][1]
        xgb_pred = self.xgb_model.predict_proba(X_scaled)[0][1]
        lstm_pred = self.lstm_model.predict(X_scaled.reshape(1, 1, -1))[0][0]
        
        # Ensemble prediction
        ensemble_pred = np.mean([rf_pred, gb_pred, xgb_pred, lstm_pred])
        
        return {
            'ensemble_prediction': ensemble_pred,
            'individual_predictions': {
                'random_forest': rf_pred,
                'gradient_boosting': gb_pred,
                'xgboost': xgb_pred,
                'lstm': lstm_pred
            }
        }
    
    def calculate_position_size(self, confidence, account_balance, risk_per_trade=0.02):
        return account_balance * risk_per_trade * confidence
    
    def generate_signals(self, current_data, account_balance):
        predictions = self.predict(current_data)
        confidence = predictions['ensemble_prediction']
        
        # Generate trading signal
        if confidence > 0.7:
            position_size = self.calculate_position_size(confidence, account_balance)
            return {'action': 'BUY', 'size': position_size, 'confidence': confidence}
        elif confidence < 0.3:
            position_size = self.calculate_position_size(1 - confidence, account_balance)
            return {'action': 'SELL', 'size': position_size, 'confidence': 1 - confidence}
        else:
            return {'action': 'HOLD', 'size': 0, 'confidence': confidence}

# Example usage
if __name__ == "__main__":
    # Sample data loading (replace with your data source)
    df = pd.read_csv('your_price_data.csv')
    
    bot = TradingBot()
    bot.train_models(df)
    
    # Get current market data
    current_data = df.tail(100)  # Replace with real-time data
    account_balance = 10000
    
    # Generate trading signals
    signals = bot.generate_signals(current_data, account_balance)
    print(f"Trading Signal: {signals}") 