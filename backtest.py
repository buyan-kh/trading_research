import pandas as pd
import plotly.graph_objects as go

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
        for j in range(df.index.get_loc(signal['entry_date']), len(df)):
            if df['Low'].iloc[j] <= stop_loss:
                exit_price = stop_loss
                losing_trades += 1
                break
            elif df['High'].iloc[j] >= take_profit:
                exit_price = take_profit
                winning_trades += 1
                break
        
        if exit_price is None:  # If neither stop loss nor take profit was hit
            exit_price = df['Close'].iloc[-1]  # Close at the last price
        
        profit = (exit_price - entry_price) * lot_size
        current_balance += profit
        total_profit += profit
        
        # Store trade result
        results.append({
            'entry_date': signal['entry_date'],
            'entry_price': entry_price,
            'exit_price': exit_price,
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
    """Calculate Fibonacci retracement levels based on the highest and lowest prices over a specified period."""
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

def plot_chart_with_trades(df, trade_signals, results):
    """Create an interactive plot with trade signals and results, including Fibonacci levels."""
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

    fig.update_layout(
        title='GBP/USD Price Action with Trade Signals and Fibonacci Levels',
        yaxis_title='Price (USD)',
        xaxis_title='Date',
        template='plotly_dark'
    )

    fig.show()

def main():
    print("Fetching GBP/USD data for the last year at 1-hour intervals...")
    df = get_gbpusd_data()
    df = identify_all_points(df)
    
    # Calculate Fibonacci levels
    df = calculate_fibonacci_levels(df, period=50)
    
    # Identify trade signals
    trade_signals = identify_trade_signals(df)
    
    # Backtest the trades with an initial balance of $10,000 and lot size of 1
    results = backtest_trades(df, trade_signals, initial_balance=10000, lot_size=1)
    
    # Plot the chart with trade signals and results
    plot_chart_with_trades(df, trade_signals, results)

if __name__ == "__main__":
    main()
