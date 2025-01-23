import pandas as pd
import yfinance as yf

class DataHandler:
    def __init__(self, ticker='XAUUSD=X', start='2020-01-01', end='2023-01-01', interval='1d'):
        self.ticker = ticker
        self.start = start
        self.end = end
        self.interval = interval
        self.data = None

    def load_data(self):
        self.data = yf.download(self.ticker, start=self.start, end=self.end, interval=self.interval)
        self.data['Return'] = self.data['Close'].pct_change()

    def feature_engineering(self):
        self.data['SMA_10'] = self.data['Close'].rolling(window=10).mean()
        self.data['SMA_50'] = self.data['Close'].rolling(window=50).mean()
        self.data['RSI'] = 100 - (100 / (1 + self.data['Return'].rolling(window=14).mean() / self.data['Return'].rolling(window=14).std()))
        self.data.dropna(inplace=True)
        return self.data 