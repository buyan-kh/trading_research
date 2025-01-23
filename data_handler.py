import pandas as pd
import yfinance as yf

def load_data(ticker='XAUUSD=X', start='2020-01-01', end='2023-01-01', interval='1d'):
    data = yf.download(ticker, start=start, end=end, interval=interval)
    data['Return'] = data['Close'].pct_change()
    return data

def feature_engineering(data):
    data['SMA_10'] = data['Close'].rolling(window=10).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['RSI'] = 100 - (100 / (1 + data['Return'].rolling(window=14).mean() / data['Return'].rolling(window=14).std()))
    data.dropna(inplace=True)
    return data 