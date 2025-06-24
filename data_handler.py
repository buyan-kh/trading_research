import pandas as pd
import yfinance as yf
import talib

class DataHandler:
    def __init__(self, ticker='AAPL', start='2020-01-01', end='2023-01-01', interval='1d'):
        self.ticker = ticker
        self.start = start
        self.end = end
        self.interval = interval
        self.data = None

    def load_data(self):
        self.data = yf.download(self.ticker, start=self.start, end=self.end, interval=self.interval)
        self.data['Return'] = self.data['Close'].pct_change()

    def feature_engineering(self):
        # Check if 'Volume' column exists before calculating OBV
        if 'Volume' not in self.data.columns:
            self.data['Volume'] = 0  # or handle appropriately

        self.data['SMA_10'] = self.data['Close'].rolling(window=10).mean()
        self.data['SMA_50'] = self.data['Close'].rolling(window=50).mean()
        self.data['RSI'] = talib.RSI(self.data['Close'], timeperiod=14)
        self.data['MACD'], self.data['MACD_signal'], _ = talib.MACD(self.data['Close'])
        self.data['BB_upper'], self.data['BB_middle'], self.data['BB_lower'] = talib.BBANDS(self.data['Close'])
        self.data['OBV'] = talib.OBV(self.data['Close'], self.data['Volume'])
        self.data.dropna(inplace=True)
        return self.data 