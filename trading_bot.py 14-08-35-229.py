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

def analyze_between_points(df):
    """Analyze pip differences between points and add annotations"""
    # Get between points with their prices
    between_highs = df[df['between_high']][['High']]
    between_lows = df[df['between_low']][['Low']]
    
    # Store annotations for plotting
    annotations = []
    high_diffs = []
    low_diffs = []
    
    # Calculate pip differences for between highs
    for i in range(1, len(between_highs)):
        current_price = between_highs['High'].iloc[i]
        prev_price = between_highs['High'].iloc[i-1]
        price_diff = abs(current_price - prev_price)
        pips = price_to_pips(price_diff)
        
        high_diffs.append({
            'start_date': between_highs.index[i-1],
            'end_date': between_highs.index[i],
            'pips': pips
        })
        
        # Create annotation for chart
        mid_point = between_highs.index[i-1] + (between_highs.index[i] - between_highs.index[i-1])/2
        annotations.append(dict(
            x=mid_point,
            y=max(current_price, prev_price),
            text=f"{pips}p",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor='orange',
            font=dict(size=10, color='orange'),
            ax=0,
            ay=-40
        ))
    
    # Calculate pip differences for between lows
    for i in range(1, len(between_lows)):
        current_price = between_lows['Low'].iloc[i]
        prev_price = between_lows['Low'].iloc[i-1]
        price_diff = abs(current_price - prev_price)
        pips = price_to_pips(price_diff)
        
        low_diffs.append({
            'start_date': between_lows.index[i-1],
            'end_date': between_lows.index[i],
            'pips': pips
        })
        
        # Create annotation for chart
        mid_point = between_lows.index[i-1] + (between_lows.index[i] - between_lows.index[i-1])/2
        annotations.append(dict(
            x=mid_point,
            y=min(current_price, prev_price),
            text=f"{pips}p",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor='blue',
            font=dict(size=10, color='blue'),
            ax=0,
            ay=40
        ))
    
    return annotations, high_diffs, low_diffs

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

def plot_chart_with_trades(df, trade_signals):
    """Create an interactive plot with trade signals"""
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
    
    # Add between highs (orange)
    between_highs = df[df['between_high']]
    fig.add_scatter(
        x=between_highs.index,
        y=between_highs['High'],
        mode='markers',
        marker=dict(symbol='triangle-down', size=8, color='orange'),
        name='Between Highs'
    )
    
    # Add between lows (blue)
    between_lows = df[df['between_low']]
    fig.add_scatter(
        x=between_lows.index,
        y=between_lows['Low'],
        mode='markers',
        marker=dict(symbol='triangle-up', size=8, color='blue'),
        name='Between Lows'
    )

    # Plot trade signals
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

    # Add pip difference annotations
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
    
    # Plot the chart with trade signals
    plot_chart_with_trades(df, trade_signals)

if __name__ == "__main__":
    main() 