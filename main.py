import yfinance as yf
import pandas as pd
from backtest import (
    calculate_fibonacci_levels,
    count_cycles,
    analyze_retracements,
    calculate_derivative,
    calculate_integral,
    identify_trade_signals,
    backtest_trades,
    plot_chart_with_trades
)

def get_gbpusd_data(period="1y", interval="1h"):
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