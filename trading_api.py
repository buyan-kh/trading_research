class TradingAPI:
    def __init__(self, api_key, api_secret):
        self.api_key = api_key
        self.api_secret = api_secret

    def connect(self):
        print("Connecting to trading platform...")

    def place_order(self, symbol, quantity, order_type='market'):
        print(f"Placing {order_type} order for {quantity} of {symbol}")

    def get_account_balance(self):
        print("Retrieving account balance...")
        return 10000