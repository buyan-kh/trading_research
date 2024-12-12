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
            
            trade_signals.append({
                'entry_date': df.index[i],
                'entry_price': entry_price,
                'take_profit': take_profit
            })
    
    return trade_signals

def backtest_trades(df, trade_signals, initial_balance=10000, lot_size=1):
    """
    Backtest trading strategy with enhanced win percentage tracking
    
    Parameters:
    - df: DataFrame with price data
    - trade_signals: List of trade signals
    - initial_balance: Starting account balance
    - lot_size: Size of each trade
    
    Returns:
    - Detailed results including win/loss percentages
    """
    results = []
    current_balance = initial_balance
    
    # Tracking variables for win/loss analysis
    total_trades = 0
    winning_trades = 0
    losing_trades = 0
    
    # Profit/loss tracking
    total_profit = 0
    max_profit = float('-inf')
    max_loss = float('inf')
    
    for signal in trade_signals:
        total_trades += 1
        
        # Simulate trade execution
        entry_price = signal['entry_price']
        stop_loss = signal['stop_loss']
        take_profit = signal['take_profit']
        trade_type = signal['type']
        
        # Calculate potential profit/loss
        if trade_type == 'long':
            trade_result = take_profit - entry_price
        else:  # short
            trade_result = entry_price - take_profit
        
        # Determine trade outcome
        if trade_result > 0:
            winning_trades += 1
            profit = abs(trade_result) * lot_size
            current_balance += profit
            total_profit += profit
            max_profit = max(max_profit, profit)
        else:
            losing_trades += 1
            loss = abs(trade_result) * lot_size
            current_balance -= loss
            total_profit -= loss
            max_loss = min(max_loss, loss)
        
        # Store trade result
        results.append({
            'type': trade_type,
            'entry_price': entry_price,
            'profit': trade_result * lot_size,
            'is_winning_trade': trade_result > 0
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
    print(f"Max Single Trade Profit: ${max_profit:.2f}")
    print(f"Max Single Trade Loss: ${max_loss:.2f}")
    print(f"Final Account Balance: ${current_balance:.2f}")
    
    return results

def plot_chart_with_trades(df, trade_signals, results):
    """Create an interactive plot with trade signals and results"""
    fig = go.Figure(data=[go.Candlestick(x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name='GBP/USD')])

    # Add swing highs (red)
    swing_highs = df[df['swing_high']]
    fig.add_scatter(
        x=swing_highs.index,
        y=swing_highs['High'],
        mode='markers',
        marker=dict(symbol='triangle-down', size=10, color='red'),
        name='Swing Highs'
    )
    
    # Add swing lows (green)
    swing_lows = df[df['swing_low']]
    fig.add_scatter(
        x=swing_lows.index,
        y=swing_lows['Low'],
        mode='markers',
        marker=dict(symbol='triangle-up', size=10, color='green'),
        name='Swing Lows'
    )
    
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
            line=dict(color='green', width=2),
            name='Trade Result',
            text=[f"Profit: {result['profit']:.4f}"],
            textposition="top right"
        ))

    fig.update_layout(
        title='GBP/USD Price Action with Trade Signals',
        yaxis_title='Price (USD)',
        xaxis_title='Date',
        template='plotly_dark'
    )

    fig.show()
