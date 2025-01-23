from django.shortcuts import render
from .utils import fetch_stock_data, create_candlestick_chart

def stock_view(request):
    ticker = request.GET.get('ticker', 'AAPL')
    period = request.GET.get('period', '1mo')
    interval = request.GET.get('interval', '1d')
    data = fetch_stock_data(ticker, period, interval)
    chart = create_candlestick_chart(data)
    context = {
        'ticker': ticker,
        'data': data.to_html(classes='table table-striped'),
        'chart': chart,
    }
    return render(request, 'stocks/stock_view.html', context)