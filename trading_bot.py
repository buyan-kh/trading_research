import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

def get_gbpusd_data(period="1mo", interval="1h"):
    """Get GBP/USD historical data from Yahoo Finance"""
    ticker = yf.Ticker("GBPUSD=X")
    df = ticker.history(period=period, interval=interval)
    return df

def identify_all_points(df, window=5):
    """Identify swing points and between points"""
    # Initialize all point types
    df['swing_high'] = False
    df['swing_low'] = False
    df['between_high'] = False
    df['between_low'] = False
    df['lower_high'] = False  # Add this for tracking lower highs
    
    # First identify swing points
    for i in range(window, len(df) - window):
        current_high = df['High'].iloc[i]
        current_low = df['Low'].iloc[i]
        
        # Check surrounding bars
        left_bars = df.iloc[i-window:i]
        right_bars = df.iloc[i+1:i+window+1]
        
        # Swing high
        if (current_high > left_bars['High'].max() and 
            current_high > right_bars['High'].max()):
            df.loc[df.index[i], 'swing_high'] = True
        
        # Swing low
        if (current_low < left_bars['Low'].min() and 
            current_low < right_bars['Low'].min()):
            df.loc[df.index[i], 'swing_low'] = True
            df.loc[df.index[i], 'lower_high'] = True  # Mark as lower high
    
    # Now identify between points
    for i in range(window, len(df) - window):
        # Find between high points (highest point between two swing lows)
        if i > 0 and i < len(df) - 1:
            prev_sl_idx = df[:i][df['swing_low']].index[-1] if any(df[:i]['swing_low']) else None
            next_sl_idx = df[i:][df['swing_low']].index[0] if any(df[i:]['swing_low']) else None
            
            if prev_sl_idx is not None and next_sl_idx is not None:
                current_section = df.loc[prev_sl_idx:next_sl_idx]
                if df['High'].iloc[i] == current_section['High'].max():
                    df.loc[df.index[i], 'between_high'] = True
        
        # Find between low points (lowest point between two swing highs)
        if i > 0 and i < len(df) - 1:
            prev_sh_idx = df[:i][df['swing_high']].index[-1] if any(df[:i]['swing_high']) else None
            next_sh_idx = df[i:][df['swing_high']].index[0] if any(df[i:]['swing_high']) else None
            
            if prev_sh_idx is not None and next_sh_idx is not None:
                current_section = df.loc[prev_sh_idx:next_sh_idx]
                if df['Low'].iloc[i] == current_section['Low'].min():
                    df.loc[df.index[i], 'between_low'] = True

    return df

def price_to_pips(price_diff):
    """Convert price difference to pips"""
    return round(price_diff * 10000)

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

def main():
    print("Fetching GBP/USD data...")
    df = get_gbpusd_data()
    df = identify_all_points(df)
    
    # Identify trade signals
    trade_signals = identify_trade_signals(df)
    
    # Backtest the trades
    results = backtest_trades(df, trade_signals)
    
    # Plot the chart with trade signals and results
    plot_chart_with_trades(df, trade_signals, results)

    # Print backtest results
    print("\nBacktest Results:")
    print("-----------------")
    print(f"Total Trades: {len(results)}")
    total_profit = sum(result['profit'] for result in results)
    print(f"Total Profit: {total_profit:.4f} USD")
    win_rate = (len(results) / len(trade_signals)) * 100 if trade_signals else 0
    print(f"Win Rate: {win_rate:.2f}%")

if __name__ == "__main__":
    main() 