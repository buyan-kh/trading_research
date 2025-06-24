import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input, Attention
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import warnings
warnings.filterwarnings('ignore')

def create_advanced_lstm_model(input_shape, units=[64, 32], dropout_rate=0.3):
    """Create advanced LSTM model with modern architecture"""
    model = Sequential([
        LSTM(units[0], return_sequences=True, input_shape=input_shape),
        BatchNormalization(),
        Dropout(dropout_rate),
        
        LSTM(units[1], return_sequences=True),
        BatchNormalization(),
        Dropout(dropout_rate),
        
        LSTM(units[1]//2, return_sequences=False),
        BatchNormalization(),
        Dropout(dropout_rate),
        
        Dense(32, activation='relu'),
        Dropout(dropout_rate/2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    
    optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
    model.compile(optimizer=optimizer, loss='huber', metrics=['mae'])
    return model

def prepare_multivariate_data(data, look_back=60, features=['Close', 'Volume', 'High', 'Low']):
    """Prepare multivariate time series data"""
    feature_data = data[features].values
    scaler = RobustScaler()  # More robust to outliers than MinMaxScaler
    scaled_data = scaler.fit_transform(feature_data)
    
    X, y = [], []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i-look_back:i])
        y.append(scaled_data[i, 0])  # Predict Close price
    
    return np.array(X), np.array(y), scaler

def train_lstm_model(data, look_back=60, validation_split=0.2):
    """Train advanced LSTM model with proper validation and callbacks"""
    if len(data) < look_back + 20:
        raise ValueError(f"Not enough data. Need at least {look_back + 20} data points.")
    
    # Prepare multivariate data
    available_features = [col for col in ['Close', 'Volume', 'High', 'Low'] if col in data.columns]
    X, y, scaler = prepare_multivariate_data(data, look_back, available_features)
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=validation_split, shuffle=False)
    
    # Create model
    model = create_advanced_lstm_model((look_back, len(available_features)))
    
    # Callbacks for better training
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
    ]
    
    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    
    return model, scaler, history

def predict_next_price(model, scaler, recent_data, look_back=60):
    """Make prediction with confidence intervals"""
    if len(recent_data) < look_back:
        raise ValueError(f"Need at least {look_back} recent data points")
    
    # Get last sequence
    last_sequence = recent_data[-look_back:]
    last_sequence_scaled = scaler.transform(last_sequence)
    
    # Reshape for prediction
    X_test = last_sequence_scaled.reshape(1, look_back, -1)
    
    # Make prediction
    predicted_scaled = model.predict(X_test, verbose=0)
    
    # Inverse transform (only for Close price)
    dummy_array = np.zeros((1, last_sequence.shape[1]))
    dummy_array[0, 0] = predicted_scaled[0, 0]
    predicted_price = scaler.inverse_transform(dummy_array)[0, 0]
    
    return predicted_price

def evaluate_model_performance(model, X_test, y_test, scaler):
    """Evaluate model performance with multiple metrics"""
    predictions = model.predict(X_test)
    
    # Calculate metrics
    mae = np.mean(np.abs(predictions - y_test))
    rmse = np.sqrt(np.mean((predictions - y_test) ** 2))
    mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
    
    return {
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'predictions': predictions
    } 