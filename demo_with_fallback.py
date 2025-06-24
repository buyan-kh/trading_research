#!/usr/bin/env python3
"""
Trading Bot Demo with Fallback Data
Works even when Yahoo Finance API is down
"""

import sys
import os
sys.path.append('/Users/buyantogtokh/bot/trading/venv/lib/python3.12/site-packages')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Import our enhanced modules
from stocks.utils_enhanced import (
    fetch_stock_data, analyze_stock_trends, 
    get_trading_signals, calculate_portfolio_metrics,
    generate_sample_data
)

def test_data_fetch():
    """Test data fetching with fallback"""
    print("🔍 Testing Data Sources")
    print("=" * 40)
    
    tickers = ['AAPL', 'GOOGL', 'MSFT', 'TSLA']
    working_data = {}
    
    for ticker in tickers:
        print(f"\n📊 Testing {ticker}...")
        data = fetch_stock_data(ticker, period='6mo')
        
        if not data.empty:
            working_data[ticker] = data
            current_price = data['Close'].iloc[-1]
            price_change = data['Close'].iloc[-1] - data['Close'].iloc[-2]
            price_change_pct = (price_change / data['Close'].iloc[-2]) * 100
            
            print(f"  ✅ Success: ${current_price:.2f} ({price_change_pct:+.2f}%)")
        else:
            print(f"  ❌ Failed to fetch data")
    
    return working_data

def demo_with_sample_data(ticker='AAPL'):
    """Run demo with guaranteed sample data"""
    print(f"\n🎯 Demo Analysis for {ticker}")
    print("=" * 50)
    
    # Get data (will fallback to sample if needed)
    data = fetch_stock_data(ticker, period='1y')
    
    current_price = data['Close'].iloc[-1]
    price_change = data['Close'].iloc[-1] - data['Close'].iloc[-2]
    price_change_pct = (price_change / data['Close'].iloc[-2]) * 100
    
    print(f"Current Price: ${current_price:.2f}")
    print(f"Daily Change: {price_change:+.2f} ({price_change_pct:+.2f}%)")
    
    # Technical analysis
    analysis = analyze_stock_trends(data)
    print(f"\n📊 Technical Analysis:")
    if analysis:
        for key, value in analysis.items():
            print(f"  • {key.replace('_', ' ').title()}: {value}")
    else:
        print("  • No analysis available")
    
    # Trading signals
    signals = get_trading_signals(data)
    print(f"\n🚦 Trading Signals ({len(signals)} found):")
    if signals:
        for signal in signals:
            icon = "🟢" if signal['type'] == 'BUY' else "🔴"
            print(f"  {icon} {signal['type']} - {signal['indicator']}: {signal['reason']}")
    else:
        print("  • No signals generated")
    
    # Portfolio metrics
    metrics = calculate_portfolio_metrics(data)
    print(f"\n💰 Performance Metrics:")
    if metrics:
        print(f"  • Total Return: {metrics.get('total_return', 0):.2f}%")
        print(f"  • Annualized Return: {metrics.get('annualized_return', 0):.2f}%")
        print(f"  • Volatility: {metrics.get('volatility', 0):.2f}%")
        print(f"  • Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
        print(f"  • Max Drawdown: {metrics.get('max_drawdown', 0):.2f}%")
        print(f"  • Win Rate: {metrics.get('win_rate', 0):.2f}%")
    
    return data, analysis, signals, metrics

def create_simple_chart(data, ticker):
    """Create a simple matplotlib chart"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Price chart
    ax1.plot(data.index, data['Close'], label='Close Price', linewidth=2)
    if 'SMA_20' in data.columns:
        ax1.plot(data.index, data['SMA_20'], label='SMA 20', alpha=0.7)
    if 'SMA_50' in data.columns:
        ax1.plot(data.index, data['SMA_50'], label='SMA 50', alpha=0.7)
    
    ax1.set_title(f'{ticker} - Price Analysis')
    ax1.set_ylabel('Price ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Volume chart
    ax2.bar(data.index, data['Volume'], alpha=0.6, color='lightblue')
    ax2.set_title('Volume')
    ax2.set_ylabel('Volume')
    ax2.set_xlabel('Date')
    
    plt.tight_layout()
    plt.savefig(f'{ticker}_analysis_fallback.png', dpi=150, bbox_inches='tight')
    print(f"📊 Chart saved as {ticker}_analysis_fallback.png")
    plt.show()

def demo_portfolio_fallback(tickers=['AAPL', 'GOOGL', 'MSFT', 'TSLA']):
    """Portfolio demo with fallback data"""
    print(f"\n💼 Portfolio Analysis (with fallback)")
    print("=" * 50)
    
    portfolio_data = []
    total_investment = 0
    total_final_value = 0
    
    for ticker in tickers:
        print(f"\n📊 Analyzing {ticker}...")
        data = fetch_stock_data(ticker, period='1y')
        
        if not data.empty:
            metrics = calculate_portfolio_metrics(data, initial_investment=10000)
            portfolio_data.append({
                'ticker': ticker,
                'current_price': data['Close'].iloc[-1],
                'metrics': metrics
            })
            total_investment += 10000
            total_final_value += metrics['final_value']
        else:
            print(f"  ❌ Skipping {ticker} - no data")
    
    if portfolio_data:
        portfolio_return = ((total_final_value - total_investment) / total_investment) * 100
        
        print(f"\n💰 Portfolio Summary:")
        print(f"  Total Investment: ${total_investment:,.2f}")
        print(f"  Total Final Value: ${total_final_value:,.2f}")
        print(f"  Portfolio Return: {portfolio_return:+.2f}%")
        
        print(f"\n📊 Individual Performance:")
        for item in portfolio_data:
            metrics = item['metrics']
            print(f"  {item['ticker']}: ${item['current_price']:.2f} | "
                  f"Return: {metrics['total_return']:+.2f}% | "
                  f"Sharpe: {metrics['sharpe_ratio']:.2f}")
    
    return portfolio_data

def main():
    """Main demo with fallback support"""
    print("🚀 Trading Bot Demo (Fallback Mode)")
    print("=" * 60)
    
    # Test data sources first
    working_data = test_data_fetch()
    
    if working_data:
        ticker = list(working_data.keys())[0]
        print(f"\n✅ Using real data from working ticker: {ticker}")
    else:
        ticker = 'AAPL'
        print(f"\n📊 Yahoo Finance unavailable - using sample data for {ticker}")
    
    # Run analysis
    data, analysis, signals, metrics = demo_with_sample_data(ticker)
    
    # Create chart
    create_simple_chart(data, ticker)
    
    # Portfolio analysis
    demo_portfolio_fallback(['AAPL', 'GOOGL', 'MSFT', 'TSLA'])
    
    print(f"\n🎉 Demo Complete!")
    print(f"📁 Files created: {ticker}_analysis_fallback.png")
    print(f"🌐 Start web server: python stock_analysis/manage.py runserver")
    print(f"💡 Tip: The web app will also use fallback data if needed")

if __name__ == "__main__":
    main()