from django.shortcuts import render
from .utils import fetch_stock_data, create_candlestick_chart
from .lstm_model import train_lstm_model, predict_next_price

def stock_view(request):
    ticker = request.GET.get('ticker', 'AAPL')
    period = request.GET.get('period', '1mo')
    interval = request.GET.get('interval', '1d')
    
    # Fetch stock data
    data = fetch_stock_data(ticker, period, interval)
    
    if data.empty:
        return render(request, 'stocks/stock_view.html', {'error': 'No data found for the given ticker.'})

    chart = create_candlestick_chart(data)

    # Train LSTM model and make predictions
    model, scaler = train_lstm_model(data)
    last_60_days = data['Close'][-60:].values.reshape(-1, 1)
    predicted_price = predict_next_price(model, scaler, last_60_days)

    context = {
        'ticker': ticker,
        'data': data.to_html(classes='table table-striped'),
        'chart': chart,
        'predicted_price': predicted_price,
    }
    return render(request, 'stocks/stock_view.html', context)

def index(request):
    return render(request, 'stocks/stock_view.html', {'ticker': 'AAPL', 'data': '', 'chart': ''})