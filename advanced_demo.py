#!/usr/bin/env python3
"""
Advanced Trading Bot Demo
Showcases enhanced LSTM models, technical analysis, and portfolio management
"""

import sys
import os
sys.path.append('/Users/buyantogtokh/bot/trading/venv/lib/python3.12/site-packages')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Import our enhanced modules
from stocks.utils_enhanced import (
    fetch_stock_data, analyze_stock_trends, 
    get_trading_signals, calculate_portfolio_metrics
)
from lstm_trainer_advanced import AdvancedLSTMTrainer

def demo_enhanced_analysis(ticker='AAPL'):
    """Demonstrate enhanced technical analysis"""
    print(f"🔍 Enhanced Analysis for {ticker}")
    print("=" * 50)
    
    # Fetch data with enhanced indicators
    data = fetch_stock_data(ticker, period='1y')
    if data.empty:
        print(f"❌ Could not fetch data for {ticker}")
        return
    
    current_price = data['Close'].iloc[-1]
    print(f"Current Price: ${current_price:.2f}")
    
    # Technical analysis
    analysis = analyze_stock_trends(data)
    print(f"\n📊 Technical Analysis:")
    for key, value in analysis.items():
        print(f"  {key.replace('_', ' ').title()}: {value}")
    
    # Trading signals
    signals = get_trading_signals(data)
    print(f"\n🚦 Trading Signals ({len(signals)} found):")
    for signal in signals:
        icon = "🟢" if signal['type'] == 'BUY' else "🔴"
        print(f"  {icon} {signal['type']} - {signal['indicator']}: {signal['reason']}")
    
    # Portfolio metrics
    metrics = calculate_portfolio_metrics(data)
    print(f"\n💰 Portfolio Performance:")
    for key, value in metrics.items():
        print(f"  {key.replace('_', ' ').title()}: {value}")
    
    return data, analysis, signals, metrics

def demo_advanced_lstm(ticker='AAPL'):
    """Demonstrate advanced LSTM model"""
    print(f"\n🧠 Advanced LSTM Training for {ticker}")
    print("=" * 50)
    
    # Fetch data
    data = fetch_stock_data(ticker, period='2y')
    if data.empty or len(data) < 200:
        print(f"❌ Need more data for LSTM training (have {len(data)}, need 200+)")
        return None
    
    print(f"✅ Data loaded: {len(data)} days")
    
    # Initialize advanced LSTM trainer
    trainer = AdvancedLSTMTrainer(data, look_back=60)
    
    # Prepare data
    print("📊 Preparing multivariate data...")
    X_train, y_train = trainer.prepare_data(validation_split=0.2)
    
    # Build model
    print("🏗️ Building advanced LSTM model...")
    trainer.build_model(units=[128, 64, 32], dropout_rate=0.3, bidirectional=True)
    
    # Train model
    print("🎯 Training model (this may take a few minutes)...")
    history = trainer.train_model(epochs=50, batch_size=32, patience=10)
    
    # Evaluate model
    print("📈 Evaluating model performance...")
    results = trainer.evaluate_model()
    
    # Make future predictions
    print("🔮 Making future predictions...")
    future_predictions = trainer.predict_future(n_steps=5)
    
    print(f"\n📊 Model Results:")
    print(f"  Directional Accuracy: {results['directional_accuracy']:.2%}")
    print(f"  MAE: {results['mae']:.6f}")
    print(f"  RMSE: {results['rmse']:.6f}")
    
    print(f"\n🔮 5-Day Predictions:")
    current_price = data['Close'].iloc[-1]
    for i, pred in enumerate(future_predictions, 1):
        # Inverse transform prediction
        dummy_array = np.zeros((1, len(trainer.features)))
        dummy_array[0, 0] = pred
        pred_price = trainer.scaler.inverse_transform(dummy_array)[0, 0]
        change_pct = ((pred_price - current_price) / current_price) * 100
        print(f"  Day {i}: ${pred_price:.2f} ({change_pct:+.2f}%)")
        current_price = pred_price
    
    return trainer, results

def demo_portfolio_analysis(tickers=['AAPL', 'GOOGL', 'MSFT', 'TSLA']):
    """Demonstrate portfolio analysis"""
    print(f"\n💼 Portfolio Analysis")
    print("=" * 50)
    
    portfolio_data = []
    total_investment = 0
    total_final_value = 0
    
    for ticker in tickers:
        print(f"📊 Analyzing {ticker}...")
        try:
            data = fetch_stock_data(ticker, period='1y')
            if not data.empty:
                metrics = calculate_portfolio_metrics(data, initial_investment=10000)
                portfolio_data.append({
                    'ticker': ticker,
                    'metrics': metrics
                })
                total_investment += 10000
                total_final_value += metrics['final_value']
                
        except Exception as e:
            print(f"❌ Error analyzing {ticker}: {e}")
            continue
    
    # Portfolio summary
    portfolio_return = ((total_final_value - total_investment) / total_investment) * 100
    
    print(f"\n💰 Portfolio Summary:")
    print(f"  Total Investment: ${total_investment:,.2f}")
    print(f"  Total Final Value: ${total_final_value:,.2f}")
    print(f"  Portfolio Return: {portfolio_return:+.2f}%")
    
    print(f"\n📊 Individual Stock Performance:")
    for item in portfolio_data:
        metrics = item['metrics']
        print(f"  {item['ticker']}: {metrics['total_return']:+.2f}% "
              f"(Sharpe: {metrics['sharpe_ratio']:.2f}, "
              f"Max DD: {metrics['max_drawdown']:.2f}%)")
    
    return portfolio_data

def main():
    """Run complete advanced demo"""
    print("🚀 Advanced Trading Bot Demo")
    print("=" * 60)
    
    # Demo 1: Enhanced Technical Analysis
    demo_enhanced_analysis('AAPL')
    
    # Demo 2: Advanced LSTM (optional - computationally intensive)
    train_lstm = input("\n🤖 Train LSTM model? (takes 2-5 minutes) [y/N]: ").lower() == 'y'
    if train_lstm:
        demo_advanced_lstm('AAPL')
    
    # Demo 3: Portfolio Analysis
    demo_portfolio_analysis(['AAPL', 'GOOGL', 'MSFT', 'TSLA'])
    
    print(f"\n🎉 Demo Complete!")
    print(f"📁 Check for generated files: training_history.png, best_lstm_model.h5")
    print(f"🌐 Start Django server with: python stock_analysis/manage.py runserver")

if __name__ == "__main__":
    main()