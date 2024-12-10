import pandas as pd

def calculate_daily_midpoint(df):
    """Calculate the daily midpoint, double it, and double the difference between high and low"""
    # Ensure the DataFrame has a DateTime index
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DateTimeIndex.")

    # Group by date and calculate the highest and lowest prices
    daily_highs = df.resample('D').max()['High']
    daily_lows = df.resample('D').min()['Low']
    
    # Calculate the midpoint for each day
    daily_midpoints = (daily_highs + daily_lows) / 2
    
    # Multiply the midpoint by 2
    daily_midpoints_doubled = daily_midpoints * 2
    
    # Calculate double the difference between high and low
    daily_differences_doubled = (daily_highs - daily_lows) * 2
    
    return daily_midpoints, daily_midpoints_doubled, daily_differences_doubled

def print_daily_midpoints(df):
    """Print the daily midpoints and their doubled values"""
    midpoints, doubled_midpoints, doubled_differences = calculate_daily_midpoint(df)
    
    print("Daily Midpoints and Doubled Values:")
    print("-------------------------------------")
    for date in midpoints.index:
        print(f"Date: {date.date()}, Midpoint: {midpoints[date]:.4f}, Doubled: {doubled_midpoints[date]:.4f}, Doubled Difference: {doubled_differences[date]:.4f}")