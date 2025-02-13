import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

def create_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))  # Prediction of the next price
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_lstm_model(data):
    # Prepare the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

    # Create training data
    training_data_len = int(np.ceil(len(scaled_data) * 0.8))
    train_data = scaled_data[0:training_data_len, :]

    x_train, y_train = [], []
    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i, 0])
        y_train.append(train_data[i, 0])
    x_train, y_train = np.array(x_train), np.array(y_train)

    # Reshape for LSTM
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # Create and train the model
    model = create_lstm_model((x_train.shape[1], 1))
    model.fit(x_train, y_train, batch_size=1, epochs=10)

    return model, scaler

def predict_next_price(model, scaler, last_60_days):
    last_60_days_scaled = scaler.transform(last_60_days)
    X_test = []
    X_test.append(last_60_days_scaled)
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    predicted_price = model.predict(X_test)
    predicted_price = scaler.inverse_transform(predicted_price)
    return predicted_price[0][0] 