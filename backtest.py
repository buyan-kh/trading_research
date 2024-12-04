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

def backtest_trades(df, trade_signals):
    """Backtest the identified trades and calculate results"""
    results = []
    
    for signal in trade_signals:
        # Check if the take profit is hit in the future
        for j in range(df.index.get_loc(signal['entry_date']), len(df)):
            if df['High'].iloc[j] >= signal['take_profit']:
                results.append({
                    'entry_date': signal['entry_date'],
                    'entry_price': signal['entry_price'],
                    'take_profit': signal['take_profit'],
                    'exit_date': df.index[j],
                    'exit_price': signal['take_profit'],
                    'profit': signal['take_profit'] - signal['entry_price']
                })
                break  # Exit the loop once the take profit is hit
    
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
