from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json

try:
    from .utils_enhanced import (
        fetch_stock_data, create_candlestick_chart, 
        analyze_stock_trends, get_trading_signals,
        calculate_portfolio_metrics, create_correlation_heatmap
    )
except ImportError:
    from .utils import fetch_stock_data, create_candlestick_chart
    
try:
    from .lstm_model import train_lstm_model, predict_next_price
except ImportError:
    train_lstm_model = None
    predict_next_price = None

def stock_view(request):
    ticker = request.GET.get('ticker', 'AAPL')
    period = request.GET.get('period', '6mo')
    interval = request.GET.get('interval', '1d')
    
    # Fetch stock data
    data = fetch_stock_data(ticker, period)
    
    if data.empty:
        return render(request, 'stocks/stock_view.html', {
            'error': f'No data found for ticker: {ticker}'
        })

    # Create enhanced chart
    chart = create_candlestick_chart(data, ticker)
    
    # Get current price and basic info
    current_price = data['Close'].iloc[-1]
    price_change = data['Close'].iloc[-1] - data['Close'].iloc[-2]
    price_change_pct = (price_change / data['Close'].iloc[-2]) * 100
    
    # Analyze trends and get signals
    analysis = {}
    signals = []
    portfolio_metrics = {}
    
    try:
        analysis = analyze_stock_trends(data)
        signals = get_trading_signals(data)
        portfolio_metrics = calculate_portfolio_metrics(data)
    except:
        pass  # Fallback gracefully if enhanced features not available

    # LSTM predictions (if available)
    predicted_price = None
    if train_lstm_model and predict_next_price and len(data) >= 60:
        try:
            model, scaler, _ = train_lstm_model(data)
            recent_data = data[['Close', 'Volume', 'High', 'Low']].tail(60).values
            predicted_price = predict_next_price(model, scaler, recent_data)
        except Exception as e:
            print(f"LSTM prediction failed: {e}")

    # Prepare data table (last 20 rows)
    data_table = data.tail(20)[['Open', 'High', 'Low', 'Close', 'Volume']].round(2)
    
    context = {
        'ticker': ticker.upper(),
        'current_price': round(current_price, 2),
        'price_change': round(price_change, 2),
        'price_change_pct': round(price_change_pct, 2),
        'data_table': data_table.to_html(classes='table table-striped table-sm'),
        'chart': chart,
        'predicted_price': round(predicted_price, 2) if predicted_price else None,
        'analysis': analysis,
        'signals': signals,
        'portfolio_metrics': portfolio_metrics,
        'period': period,
    }
    return render(request, 'stocks/stock_view.html', context)

def index(request):
    """Landing page with popular stocks"""
    popular_stocks = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
    stocks_data = []
    
    for ticker in popular_stocks:
        try:
            data = fetch_stock_data(ticker, period='5d')
            if not data.empty:
                current_price = data['Close'].iloc[-1]
                price_change = data['Close'].iloc[-1] - data['Close'].iloc[-2]
                price_change_pct = (price_change / data['Close'].iloc[-2]) * 100
                
                stocks_data.append({
                    'ticker': ticker,
                    'price': round(current_price, 2),
                    'change': round(price_change, 2),
                    'change_pct': round(price_change_pct, 2)
                })
        except:
            continue
    
    context = {
        'stocks_data': stocks_data,
        'default_ticker': 'AAPL'
    }
    return render(request, 'stocks/index.html', context)

def portfolio_view(request):
    """Portfolio analysis view"""
    tickers = request.GET.get('tickers', 'AAPL,GOOGL,MSFT').split(',')
    period = request.GET.get('period', '1y')
    
    portfolio_data = []
    total_value = 0
    
    for ticker in tickers[:10]:  # Limit to 10 stocks
        ticker = ticker.strip().upper()
        try:
            data = fetch_stock_data(ticker, period)
            if not data.empty:
                metrics = calculate_portfolio_metrics(data)
                portfolio_data.append({
                    'ticker': ticker,
                    'metrics': metrics
                })
                total_value += metrics.get('final_value', 10000)
        except:
            continue
    
    # Create correlation heatmap
    correlation_chart = ""
    try:
        correlation_chart = create_correlation_heatmap([item['ticker'] for item in portfolio_data], period)
    except:
        pass
    
    context = {
        'portfolio_data': portfolio_data,
        'total_value': round(total_value, 2),
        'correlation_chart': correlation_chart,
        'tickers': ','.join(tickers),
        'period': period
    }
    return render(request, 'stocks/portfolio.html', context)

@csrf_exempt
def api_stock_data(request):
    """API endpoint for real-time stock data"""
    if request.method == 'GET':
        ticker = request.GET.get('ticker', 'AAPL')
        period = request.GET.get('period', '5d')
        
        try:
            data = fetch_stock_data(ticker, period)
            if data.empty:
                return JsonResponse({'error': 'No data found'}, status=404)
            
            # Return latest data point
            latest = data.iloc[-1]
            response_data = {
                'ticker': ticker.upper(),
                'price': round(latest['Close'], 2),
                'open': round(latest['Open'], 2),
                'high': round(latest['High'], 2),
                'low': round(latest['Low'], 2),
                'volume': int(latest['Volume']),
                'timestamp': latest.name.isoformat()
            }
            
            # Add technical indicators if available
            if 'RSI' in data.columns:
                response_data['rsi'] = round(latest['RSI'], 2)
            if 'MACD' in data.columns:
                response_data['macd'] = round(latest['MACD'], 4)
            
            return JsonResponse(response_data)
            
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    
    return JsonResponse({'error': 'Method not allowed'}, status=405)