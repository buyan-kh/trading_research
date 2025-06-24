import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import talib
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

def fetch_stock_data(ticker, period='1y', interval='1d'):
    """Fetch stock data with error handling and data validation"""
    try:
        # Try different approaches to fetch data
        stock = yf.Ticker(ticker)
        
        # Method 1: Use history with auto_adjust
        data = stock.history(period=period, interval=interval, auto_adjust=True, prepost=True)
        
        # Method 2: If empty, try download method
        if data.empty:
            print(f"Trying alternative method for {ticker}...")
            data = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
        
        # Method 3: If still empty, try shorter period
        if data.empty and period != '6mo':
            print(f"Trying shorter period for {ticker}...")
            data = stock.history(period='6mo', interval=interval, auto_adjust=True)
        
        # Method 4: Try basic period
        if data.empty:
            print(f"Trying basic period for {ticker}...")
            data = stock.history(period='1mo', interval=interval)
        
        if data.empty:
            raise ValueError(f"No data found for ticker: {ticker}")
        
        # Clean the data
        data = data.dropna()
        if len(data) < 10:
            raise ValueError(f"Insufficient data for {ticker}: only {len(data)} rows")
        
        print(f"✅ Fetched {len(data)} days of data for {ticker}")
        
        # Add basic technical indicators
        data = add_technical_indicators(data)
        return data
        
    except Exception as e:
        print(f"❌ Error fetching data for {ticker}: {e}")
        # Return sample data for demo purposes
        return generate_sample_data(ticker)

def add_technical_indicators(data):
    """Add comprehensive technical indicators to stock data"""
    if data.empty or len(data) < 50:
        return data
    
    try:
        # Price-based indicators
        data['SMA_20'] = data['Close'].rolling(window=20).mean()
        data['SMA_50'] = data['Close'].rolling(window=50).mean()
        data['SMA_200'] = data['Close'].rolling(window=200).mean()
        
        data['EMA_12'] = data['Close'].ewm(span=12).mean()
        data['EMA_26'] = data['Close'].ewm(span=26).mean()
        
        # Bollinger Bands
        data['BB_Middle'] = data['Close'].rolling(window=20).mean()
        bb_std = data['Close'].rolling(window=20).std()
        data['BB_Upper'] = data['BB_Middle'] + (bb_std * 2)
        data['BB_Lower'] = data['BB_Middle'] - (bb_std * 2)
        data['BB_Width'] = data['BB_Upper'] - data['BB_Lower']
        data['BB_Position'] = (data['Close'] - data['BB_Lower']) / (data['BB_Upper'] - data['BB_Lower'])
        
        # Volume indicators
        data['Volume_SMA'] = data['Volume'].rolling(window=20).mean()
        data['Volume_Ratio'] = data['Volume'] / data['Volume_SMA']
        
        # Volatility
        data['Returns'] = data['Close'].pct_change()
        data['Volatility'] = data['Returns'].rolling(window=20).std() * np.sqrt(252)
        
        # Support and Resistance levels
        data['Resistance'] = data['High'].rolling(window=20).max()
        data['Support'] = data['Low'].rolling(window=20).min()
        
        # Price momentum
        data['Price_Change'] = data['Close'].pct_change()
        data['Price_Change_5d'] = data['Close'].pct_change(5)
        data['Price_Change_20d'] = data['Close'].pct_change(20)
        
        # Advanced indicators using TA-Lib if available
        try:
            close_prices = data['Close'].values.astype(float)
            high_prices = data['High'].values.astype(float)
            low_prices = data['Low'].values.astype(float)
            volume_values = data['Volume'].values.astype(float)
            
            data['RSI'] = talib.RSI(close_prices, timeperiod=14)
            data['MACD'], data['MACD_Signal'], data['MACD_Hist'] = talib.MACD(close_prices)
            data['Stoch_K'], data['Stoch_D'] = talib.STOCH(high_prices, low_prices, close_prices)
            data['Williams_R'] = talib.WILLR(high_prices, low_prices, close_prices)
            data['CCI'] = talib.CCI(high_prices, low_prices, close_prices)
            data['ADX'] = talib.ADX(high_prices, low_prices, close_prices)
            data['OBV'] = talib.OBV(close_prices, volume_values)
            data['ATR'] = talib.ATR(high_prices, low_prices, close_prices)
            
        except Exception as e:
            print(f"TA-Lib indicators failed: {e}")
            # Fallback manual calculations
            data['RSI'] = calculate_rsi(data['Close'])
            data['MACD'] = data['EMA_12'] - data['EMA_26']
            data['MACD_Signal'] = data['MACD'].ewm(span=9).mean()
        
    except Exception as e:
        print(f"Error calculating technical indicators: {e}")
    
    return data

def calculate_rsi(prices, period=14):
    """Calculate RSI manually if TA-Lib not available"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def create_candlestick_chart(data, ticker='Stock'):
    """Create comprehensive candlestick chart with technical indicators"""
    if data.empty:
        return "<p>No data available for chart</p>"
    
    # Create subplots
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=('Price & Volume', 'RSI', 'MACD', 'Volume'),
        row_heights=[0.5, 0.2, 0.2, 0.1]
    )
    
    # Main candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='Price'
        ),
        row=1, col=1
    )
    
    # Add moving averages
    if 'SMA_20' in data.columns:
        fig.add_trace(
            go.Scatter(x=data.index, y=data['SMA_20'], name='SMA 20', line=dict(color='orange')),
            row=1, col=1
        )
    
    if 'SMA_50' in data.columns:
        fig.add_trace(
            go.Scatter(x=data.index, y=data['SMA_50'], name='SMA 50', line=dict(color='blue')),
            row=1, col=1
        )
    
    # Bollinger Bands
    if all(col in data.columns for col in ['BB_Upper', 'BB_Lower', 'BB_Middle']):
        fig.add_trace(
            go.Scatter(x=data.index, y=data['BB_Upper'], name='BB Upper', 
                      line=dict(color='gray', dash='dash')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=data.index, y=data['BB_Lower'], name='BB Lower', 
                      line=dict(color='gray', dash='dash'), fill='tonexty'),
            row=1, col=1
        )
    
    # RSI
    if 'RSI' in data.columns:
        fig.add_trace(
            go.Scatter(x=data.index, y=data['RSI'], name='RSI', line=dict(color='purple')),
            row=2, col=1
        )
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    
    # MACD
    if all(col in data.columns for col in ['MACD', 'MACD_Signal']):
        fig.add_trace(
            go.Scatter(x=data.index, y=data['MACD'], name='MACD', line=dict(color='blue')),
            row=3, col=1
        )
        fig.add_trace(
            go.Scatter(x=data.index, y=data['MACD_Signal'], name='Signal', line=dict(color='red')),
            row=3, col=1
        )
        if 'MACD_Hist' in data.columns:
            colors = ['green' if x > 0 else 'red' for x in data['MACD_Hist']]
            fig.add_trace(
                go.Bar(x=data.index, y=data['MACD_Hist'], name='MACD Hist', 
                      marker_color=colors, opacity=0.6),
                row=3, col=1
            )
    
    # Volume
    fig.add_trace(
        go.Bar(x=data.index, y=data['Volume'], name='Volume', marker_color='lightblue'),
        row=4, col=1
    )
    
    # Update layout
    fig.update_layout(
        title=f'{ticker} - Technical Analysis',
        yaxis_title='Price',
        xaxis_rangeslider_visible=False,
        height=800,
        showlegend=True
    )
    
    return fig.to_html(full_html=False, include_plotlyjs='cdn')

def analyze_stock_trends(data):
    """Analyze stock trends and provide insights"""
    if data.empty or len(data) < 20:
        return {}
    
    analysis = {}
    current_price = data['Close'].iloc[-1]
    
    # Price trend analysis
    if 'SMA_20' in data.columns and 'SMA_50' in data.columns:
        sma_20 = data['SMA_20'].iloc[-1]
        sma_50 = data['SMA_50'].iloc[-1]
        
        if current_price > sma_20 > sma_50:
            analysis['trend'] = 'Strong Uptrend'
        elif current_price > sma_20 and sma_20 < sma_50:
            analysis['trend'] = 'Weak Uptrend'
        elif current_price < sma_20 < sma_50:
            analysis['trend'] = 'Strong Downtrend'
        else:
            analysis['trend'] = 'Weak Downtrend'
    
    # RSI analysis
    if 'RSI' in data.columns:
        rsi = data['RSI'].iloc[-1]
        if rsi > 70:
            analysis['rsi_signal'] = 'Overbought'
        elif rsi < 30:
            analysis['rsi_signal'] = 'Oversold'
        else:
            analysis['rsi_signal'] = 'Neutral'
        analysis['rsi_value'] = rsi
    
    # Bollinger Bands analysis
    if all(col in data.columns for col in ['BB_Upper', 'BB_Lower', 'BB_Position']):
        bb_pos = data['BB_Position'].iloc[-1]
        if bb_pos > 0.8:
            analysis['bb_signal'] = 'Near Upper Band'
        elif bb_pos < 0.2:
            analysis['bb_signal'] = 'Near Lower Band'
        else:
            analysis['bb_signal'] = 'Middle Range'
    
    # Volume analysis
    if 'Volume_Ratio' in data.columns:
        vol_ratio = data['Volume_Ratio'].iloc[-1]
        if vol_ratio > 1.5:
            analysis['volume_signal'] = 'High Volume'
        elif vol_ratio < 0.5:
            analysis['volume_signal'] = 'Low Volume'
        else:
            analysis['volume_signal'] = 'Normal Volume'
    
    # Volatility analysis
    if 'Volatility' in data.columns:
        volatility = data['Volatility'].iloc[-1]
        analysis['volatility'] = f"{volatility:.2%}"
        if volatility > 0.4:
            analysis['volatility_signal'] = 'High Volatility'
        elif volatility < 0.15:
            analysis['volatility_signal'] = 'Low Volatility'
        else:
            analysis['volatility_signal'] = 'Normal Volatility'
    
    return analysis

def get_trading_signals(data):
    """Generate trading signals based on technical indicators"""
    signals = []
    
    if data.empty or len(data) < 20:
        return signals
    
    latest = data.iloc[-1]
    
    # RSI signals
    if 'RSI' in data.columns:
        rsi = latest['RSI']
        if rsi < 30:
            signals.append({'type': 'BUY', 'indicator': 'RSI', 'reason': f'Oversold (RSI: {rsi:.1f})'})
        elif rsi > 70:
            signals.append({'type': 'SELL', 'indicator': 'RSI', 'reason': f'Overbought (RSI: {rsi:.1f})'})
    
    # Moving Average signals
    if all(col in data.columns for col in ['SMA_20', 'SMA_50']):
        if data['SMA_20'].iloc[-1] > data['SMA_50'].iloc[-1] and data['SMA_20'].iloc[-2] <= data['SMA_50'].iloc[-2]:
            signals.append({'type': 'BUY', 'indicator': 'MA Cross', 'reason': 'Golden Cross (SMA 20 > SMA 50)'})
        elif data['SMA_20'].iloc[-1] < data['SMA_50'].iloc[-1] and data['SMA_20'].iloc[-2] >= data['SMA_50'].iloc[-2]:
            signals.append({'type': 'SELL', 'indicator': 'MA Cross', 'reason': 'Death Cross (SMA 20 < SMA 50)'})
    
    # MACD signals
    if all(col in data.columns for col in ['MACD', 'MACD_Signal']):
        if data['MACD'].iloc[-1] > data['MACD_Signal'].iloc[-1] and data['MACD'].iloc[-2] <= data['MACD_Signal'].iloc[-2]:
            signals.append({'type': 'BUY', 'indicator': 'MACD', 'reason': 'MACD Bullish Crossover'})
        elif data['MACD'].iloc[-1] < data['MACD_Signal'].iloc[-1] and data['MACD'].iloc[-2] >= data['MACD_Signal'].iloc[-2]:
            signals.append({'type': 'SELL', 'indicator': 'MACD', 'reason': 'MACD Bearish Crossover'})
    
    # Bollinger Bands signals
    if all(col in data.columns for col in ['BB_Upper', 'BB_Lower']):
        close = latest['Close']
        if close <= latest['BB_Lower']:
            signals.append({'type': 'BUY', 'indicator': 'Bollinger', 'reason': 'Price at Lower Bollinger Band'})
        elif close >= latest['BB_Upper']:
            signals.append({'type': 'SELL', 'indicator': 'Bollinger', 'reason': 'Price at Upper Bollinger Band'})
    
    return signals

def generate_sample_data(ticker='DEMO'):
    """Generate realistic sample stock data for demo purposes"""
    print(f"📊 Generating sample data for {ticker} (Yahoo Finance unavailable)")
    
    # Generate 252 trading days (1 year)
    dates = pd.date_range(end=pd.Timestamp.now().date(), periods=252, freq='B')
    
    # Starting price based on ticker
    start_prices = {
        'AAPL': 150.0, 'GOOGL': 2500.0, 'MSFT': 300.0, 'TSLA': 200.0, 
        'AMZN': 3000.0, 'NVDA': 400.0, 'META': 250.0
    }
    start_price = start_prices.get(ticker.upper(), 100.0)
    
    # Generate realistic price movements
    np.random.seed(42)  # For reproducible results
    returns = np.random.normal(0.001, 0.02, len(dates))  # Daily returns
    returns[::10] += np.random.normal(0, 0.05, len(returns[::10]))  # Add some volatility
    
    # Calculate prices
    prices = [start_price]
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    # Generate OHLCV data
    data = []
    for i, (date, close) in enumerate(zip(dates, prices)):
        # Generate realistic OHLC from close price
        volatility = 0.02 + 0.01 * np.sin(i / 50)  # Varying volatility
        high = close * (1 + abs(np.random.normal(0, volatility/2)))
        low = close * (1 - abs(np.random.normal(0, volatility/2)))
        open_price = close * np.random.normal(1, volatility/4)
        
        # Ensure OHLC relationship
        high = max(high, open_price, close)
        low = min(low, open_price, close)
        
        # Generate volume (higher volume on big moves)
        base_volume = 1000000
        volume = base_volume * (1 + abs(returns[i]) * 10) * np.random.uniform(0.5, 2.0)
        
        data.append({
            'Open': open_price,
            'High': high,
            'Low': low,
            'Close': close,
            'Volume': int(volume)
        })
    
    df = pd.DataFrame(data, index=dates)
    df.index.name = 'Date'
    
    print(f"✅ Generated {len(df)} days of sample data for {ticker}")
    return df

def calculate_portfolio_metrics(data, initial_investment=10000):
    """Calculate portfolio performance metrics"""
    if data.empty or 'Returns' not in data.columns:
        return {}
    
    returns = data['Returns'].dropna()
    
    # Basic metrics
    total_return = (data['Close'].iloc[-1] / data['Close'].iloc[0] - 1) * 100
    annualized_return = ((data['Close'].iloc[-1] / data['Close'].iloc[0]) ** (252 / len(data)) - 1) * 100
    volatility = returns.std() * np.sqrt(252) * 100
    
    # Sharpe ratio (assuming 0% risk-free rate)
    sharpe_ratio = annualized_return / volatility if volatility != 0 else 0
    
    # Maximum drawdown
    cumulative_returns = (1 + returns).cumprod()
    peak = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - peak) / peak
    max_drawdown = drawdown.min() * 100
    
    # Win rate
    positive_returns = returns[returns > 0]
    win_rate = len(positive_returns) / len(returns) * 100
    
    return {
        'total_return': round(total_return, 2),
        'annualized_return': round(annualized_return, 2),
        'volatility': round(volatility, 2),
        'sharpe_ratio': round(sharpe_ratio, 2),
        'max_drawdown': round(max_drawdown, 2),
        'win_rate': round(win_rate, 2),
        'final_value': round(initial_investment * (1 + total_return/100), 2)
    }

def create_correlation_heatmap(tickers, period='1y'):
    """Create correlation heatmap for multiple stocks"""
    try:
        data = {}
        for ticker in tickers:
            stock_data = fetch_stock_data(ticker, period)
            if not stock_data.empty:
                data[ticker] = stock_data['Close']
        
        if not data:
            return "<p>No data available for correlation analysis</p>"
        
        df = pd.DataFrame(data)
        correlation_matrix = df.corr()
        
        fig = px.imshow(
            correlation_matrix,
            title="Stock Correlation Heatmap",
            color_continuous_scale='RdBu',
            aspect='auto'
        )
        
        return fig.to_html(full_html=False, include_plotlyjs='cdn')
        
    except Exception as e:
        return f"<p>Error creating correlation heatmap: {e}</p>"