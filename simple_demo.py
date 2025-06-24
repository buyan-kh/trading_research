#!/usr/bin/env python3

import sys
import os
sys.path.append('/Users/buyantogtokh/bot/trading/venv/lib/python3.12/site-packages')

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def fetch_and_analyze_stock(ticker='AAPL', period='2y'):
    """Fetch stock data and perform simple analysis"""
    print(f"Fetching {ticker} data...")
    
    # Download stock data
    stock = yf.Ticker(ticker)
    data = stock.history(period=period)
    
    if data.empty:
        return f"Failed to fetch data for {ticker}"
    
    print(f"Data shape: {data.shape}")
    print(f"Date range: {data.index[0].date()} to {data.index[-1].date()}")
    
    # Basic stats
    current_price = data['Close'].iloc[-1]
    price_change = (current_price - data['Close'].iloc[-2]) / data['Close'].iloc[-2] * 100
    
    print(f"Current price: ${current_price:.2f}")
    print(f"Daily change: {price_change:.2f}%")
    
    # Simple technical indicators
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['Returns'] = data['Close'].pct_change()
    
    # Create target (1 if next day goes up, 0 if down)
    data['Target'] = np.where(data['Returns'].shift(-1) > 0, 1, 0)
    
    # Features for ML model
    features = ['Open', 'High', 'Low', 'Volume']
    data = data.dropna()
    
    if len(data) < 100:
        return f"Not enough data for analysis ({len(data)} rows)"
    
    X = data[features]
    y = data['Target'][:-1]  # Remove last row since it has no future target
    X = X[:-1]  # Align with y
    
    # Train simple model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Make prediction for tomorrow
    tomorrow_features = data[features].iloc[-1:].values
    tomorrow_pred = model.predict(tomorrow_features)[0]
    tomorrow_prob = model.predict_proba(tomorrow_features)[0]
    
    print(f"\n=== {ticker} Analysis Results ===")
    print(f"Model accuracy: {accuracy:.2f}")
    print(f"Tomorrow prediction: {'UP' if tomorrow_pred == 1 else 'DOWN'}")
    print(f"Confidence: {max(tomorrow_prob):.2f}")
    
    # Simple visualization
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(data.index[-100:], data['Close'].iloc[-100:], label='Close Price')
    plt.plot(data.index[-100:], data['SMA_20'].iloc[-100:], label='SMA 20', alpha=0.7)
    plt.plot(data.index[-100:], data['SMA_50'].iloc[-100:], label='SMA 50', alpha=0.7)
    plt.title(f'{ticker} Price & Moving Averages (Last 100 days)')
    plt.legend()
    plt.xticks(rotation=45)
    
    plt.subplot(2, 2, 2)
    plt.plot(data.index[-100:], data['Volume'].iloc[-100:])
    plt.title('Volume')
    plt.xticks(rotation=45)
    
    plt.subplot(2, 2, 3)
    plt.hist(data['Returns'].dropna(), bins=50, alpha=0.7)
    plt.title('Daily Returns Distribution')
    plt.xlabel('Returns')
    
    plt.subplot(2, 2, 4)
    cumulative_returns = (1 + data['Returns']).cumprod()
    plt.plot(data.index, cumulative_returns)
    plt.title('Cumulative Returns')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(f'{ticker}_analysis.png', dpi=150, bbox_inches='tight')
    print(f"\nChart saved as {ticker}_analysis.png")
    
    return f"Analysis complete for {ticker}"

if __name__ == "__main__":
    # Demo with multiple stocks
    stocks = ['AAPL', 'GOOGL', 'TSLA']
    
    for stock in stocks:
        try:
            result = fetch_and_analyze_stock(stock)
            print(f"{result}\n" + "="*50)
        except Exception as e:
            print(f"Error analyzing {stock}: {e}")
            continue
    
    print("\nDemo complete! Check the generated PNG files for visualizations.")