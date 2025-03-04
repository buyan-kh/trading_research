import pandas as pd
import numpy as np
import plotly.graph_objects as go
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LSTM
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import talib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from data_handler import load_data, feature_engineering

def identify_trade_signals(df):
    """Identify long trade signals based on previous lower high"""
    trade_signals = []
    
    for i in range(1, len(df)):
        # Check if the current price breaks above the previous lower high
        if df['lower_high'].iloc[i] and df['lower_high'].iloc[i-1] == False:
            entry_price = df['High'].iloc[i]  # Entry at the break of the lower high
            previous_low = df['Low'].iloc[i-1]  # Previous low
            
            # Calculate take profit
            take_profit = entry_price + (entry_price - previous_low) * 2
            
            # Calculate stop loss (e.g., 1% below entry price)
            stop_loss = entry_price * 0.99
            
            trade_signals.append({
                'entry_date': df.index[i],
                'entry_price': entry_price,
                'take_profit': take_profit,
                'stop_loss': stop_loss
            })
    
    return trade_signals

def backtest_trades(df, trade_signals, initial_balance=10000, lot_size=1):
    """
    Backtest trading strategy with enhanced win percentage tracking
    """
    results = []
    current_balance = initial_balance
    
    # Tracking variables for win/loss analysis
    total_trades = 0
    winning_trades = 0
    losing_trades = 0
    
    # Profit/loss tracking
    total_profit = 0
    
    for signal in trade_signals:
        total_trades += 1
        
        # Simulate trade execution
        entry_price = signal['entry_price']
        take_profit = signal['take_profit']
        stop_loss = signal['stop_loss']
        
        # Simulate price movement to determine trade outcome
        exit_price = None
        exit_date = None
        for j in range(df.index.get_loc(signal['entry_date']), len(df)):
            if df['Low'].iloc[j] <= stop_loss:
                exit_price = stop_loss
                exit_date = df.index[j]
                losing_trades += 1
                break
            elif df['High'].iloc[j] >= take_profit:
                exit_price = take_profit
                exit_date = df.index[j]
                winning_trades += 1
                break
        
        if exit_price is None:  # If neither stop loss nor take profit was hit
            exit_price = df['Close'].iloc[-1]  # Close at the last price
            exit_date = df.index[-1]
        
        profit = (exit_price - entry_price) * lot_size
        current_balance += profit
        total_profit += profit
        
        # Store trade result
        results.append({
            'entry_date': signal['entry_date'],
            'entry_price': entry_price,
            'exit_price': exit_price,
            'exit_date': exit_date,
            'profit': profit,
            'is_winning_trade': profit > 0
        })
    
    # Calculate percentages
    win_percentage = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
    loss_percentage = (losing_trades / total_trades) * 100 if total_trades > 0 else 0
    
    # Print detailed trade analysis
    print("\nTrade Analysis:")
    print(f"Total Trades: {total_trades}")
    print(f"Winning Trades: {winning_trades} ({win_percentage:.2f}%)")
    print(f"Losing Trades: {losing_trades} ({loss_percentage:.2f}%)")
    print(f"Total Profit: ${total_profit:.2f}")
    print(f"Final Account Balance: ${current_balance:.2f}")
    
    return results

def calculate_fibonacci_levels(df, period=50):
    """Calculate Fibonacci retracement levels based on the highest and lowest prices over a specified period"""
    if len(df) < period:
        raise ValueError("DataFrame must have at least 'period' rows.")
    
    # Calculate the highest and lowest prices over the specified period
    highest_price = df['High'].rolling(window=period).max()
    lowest_price = df['Low'].rolling(window=period).min()
    
    # Calculate Fibonacci levels
    fibonacci_0 = highest_price
    fibonacci_1 = highest_price - (highest_price - lowest_price) * 0.236  # 23.6% level
    fibonacci_2 = highest_price - (highest_price - lowest_price) * 0.618  # 61.8% level
    
    # Add Fibonacci levels to the DataFrame
    df['Fibonacci_0'] = fibonacci_0
    df['Fibonacci_1'] = fibonacci_1
    df['Fibonacci_2'] = fibonacci_2
    
    return df

def count_cycles(df, interval='4H'):
    """Count cycles based on the specified interval (4H or 1D)"""
    # Ensure the interval string includes a number
    if interval == 'D':
        interval = '1D'
    
    df['Cycle'] = (df.index.to_series().diff() > pd.Timedelta(interval)).cumsum()
    cycle_counts = df['Cycle'].value_counts().sort_index()
    return cycle_counts

def analyze_retracements(df):
    """Analyze retracements based on Fibonacci levels"""
    retracement_analysis = []
    
    for i in range(1, len(df)):
        if df['Close'].iloc[i] < df['Fibonacci_0'].iloc[i] and df['Close'].iloc[i] > df['Fibonacci_1'].iloc[i]:
            retracement_analysis.append({
                'date': df.index[i],
                'retracement_level': '23.6%',
                'price': df['Close'].iloc[i]
            })
        elif df['Close'].iloc[i] < df['Fibonacci_1'].iloc[i] and df['Close'].iloc[i] > df['Fibonacci_2'].iloc[i]:
            retracement_analysis.append({
                'date': df.index[i],
                'retracement_level': '61.8%',
                'price': df['Close'].iloc[i]
            })
    
    return retracement_analysis

def calculate_derivative(df):
    """Calculate the derivative (rate of change) of the closing price."""
    df['Price_Derivative'] = df['Close'].diff() / df.index.to_series().diff().dt.total_seconds()
    return df

def calculate_integral(df, period=50):
    """Calculate the integral (cumulative sum) of the closing price over a specified period."""
    df['Price_Integral'] = df['Close'].rolling(window=period).sum()
    return df

def plot_chart_with_trades(df, trade_signals, results, cycle_counts, retracement_analysis):
    """Create an interactive plot with trade signals, results, Fibonacci levels, and retracement analysis."""
    fig = go.Figure(data=[go.Candlestick(x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name='GBP/USD')])

    # Add Fibonacci levels
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Fibonacci_0'],
        mode='lines',
        line=dict(color='blue', width=1, dash='dash'),
        name='Fibonacci Level 0'
    ))
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Fibonacci_1'],
        mode='lines',
        line=dict(color='orange', width=1, dash='dash'),
        name='Fibonacci Level 1 (23.6%)'
    ))
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Fibonacci_2'],
        mode='lines',
        line=dict(color='red', width=1, dash='dash'),
        name='Fibonacci Level 2 (61.8%)'
    ))

    # Add trade signals
    for signal in trade_signals:
        fig.add_trace(go.Scatter(
            x=[signal['entry_date'], signal['entry_date']],
            y=[signal['entry_price'], signal['take_profit']],
            mode='lines+text',
            line=dict(color='purple', width=2, dash='dash'),
            name='Trade Entry',
            text=[f"Entry: {signal['entry_price']:.4f}", f"TP: {signal['take_profit']:.4f}"],
            textposition="top right"
        ))

    # Add results to the chart
    for result in results:
        fig.add_trace(go.Scatter(
            x=[result['entry_date'], result['exit_date']],
            y=[result['entry_price'], result['exit_price']],
            mode='lines+text',
            line=dict(color='green' if result['is_winning_trade'] else 'red', width=2),
            name='Trade Result',
            text=[f"Profit: {result['profit']:.4f}"],
            textposition="top right"
        ))

    # Add retracement analysis to the chart
    for retracement in retracement_analysis:
        fig.add_trace(go.Scatter(
            x=[retracement['date']],
            y=[retracement['price']],
            mode='markers+text',
            marker=dict(color='yellow', size=10),
            name=f'Retracement {retracement["retracement_level"]}',
            text=[f"Retracement: {retracement['retracement_level']}"],
            textposition="top center"
        ))

    # Add cycle counts to the chart
    for cycle, count in cycle_counts.items():
        fig.add_annotation(
            x=df.index[0] + pd.Timedelta(hours=cycle * 4),  # Adjust for cycle timing
            y=df['High'].max(),
            text=f'Cycle {cycle}: {count}',
            showarrow=True,
            arrowhead=2,
            ax=0,
            ay=-40,
            bgcolor='lightblue'
        )

    fig.update_layout(
        title='GBP/USD Price Action with Trade Signals, Fibonacci Levels, and Retracement Analysis',
        yaxis_title='Price (USD)',
        xaxis_title='Date',
        template='plotly_dark'
    )

    fig.show()

def create_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))  # Output layer for regression
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def prepare_data(df, feature_col, target_col, time_steps=60):
    data = df[[feature_col, target_col]].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    X, y = [], []
    for i in range(time_steps, len(scaled_data)):
        X.append(scaled_data[i-time_steps:i, 0])
        y.append(scaled_data[i, 1])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X, y, scaler

def create_features_with_indicators(df):
    df['lag_1'] = df['Close'].shift(1)
    df['lag_2'] = df['Close'].shift(2)
    df['moving_avg_3'] = df['Close'].rolling(window=3).mean()
    df['moving_avg_5'] = df['Close'].rolling(window=5).mean()
    df['volatility'] = df['Close'].rolling(window=5).std()
    df['momentum'] = df['Close'] - df['Close'].shift(3)
    
    # Add technical indicators
    df['RSI'] = talib.RSI(df['Close'], timeperiod=14)
    df['MACD'], df['MACD_signal'], _ = talib.MACD(df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
    df['BB_upper'], df['BB_middle'], df['BB_lower'] = talib.BBANDS(df['Close'], timeperiod=20)
    
    df.dropna(inplace=True)
    return df

def main():
    print("Fetching GBP/USD data for the last year at 1-hour intervals...")
    df = get_gbpusd_data()
    df = identify_all_points(df)
    
    # Calculate Fibonacci levels
    df = calculate_fibonacci_levels(df, period=50)
    
    # Count cycles for 4H and daily intervals
    cycle_counts_4h = count_cycles(df, interval='4H')
    cycle_counts_daily = count_cycles(df, interval='D')
    
    # Analyze retracements
    retracement_analysis = analyze_retracements(df)
    
    # Calculate derivatives and integrals
    df = calculate_derivative(df)
    df = calculate_integral(df, period=50)
    
    # Identify trade signals
    trade_signals = identify_trade_signals(df)
    
    # Backtest the trades with an initial balance of $10,000 and lot size of 1
    results = backtest_trades(df, trade_signals, initial_balance=10000, lot_size=1)
    
    # Plot the chart with trade signals, results, and analysis
    plot_chart_with_trades(df, trade_signals, results, cycle_counts_4h, retracement_analysis)

if __name__ == "__main__":
    main()

# Load and prepare data
data = load_data()
data = feature_engineering(data)

# Create target variable
data['Target'] = np.where(data['Return'].shift(-1) > 0, 1, 0)

# Features and target
features = ['SMA_10', 'SMA_50', 'RSI']
X = data[features]
y = data['Target']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Backtesting
data['Prediction'] = model.predict(X)
data['Strategy_Return'] = data['Return'] * data['Prediction'].shift(1)

# Calculate cumulative returns
data['Cumulative_Market_Return'] = (1 + data['Return']).cumprod()
data['Cumulative_Strategy_Return'] = (1 + data['Strategy_Return']).cumprod()

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(data['Cumulative_Market_Return'], label='Market Return')
plt.plot(data['Cumulative_Strategy_Return'], label='Strategy Return')
plt.legend()
plt.show()
