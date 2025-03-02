from data_handler import DataHandler
from model_trainer import ModelTrainer
from lstm_trainer import LSTMTrainer
from trading_api import TradingAPI

def main():
    # Load and prepare data
    data_handler = DataHandler()
    data_handler.load_data()
    data = data_handler.feature_engineering()

    # Train model and backtest
    model_trainer = ModelTrainer(data)
    model_trainer.train_model()
    model_trainer.backtest()
    model_trainer.calculate_sharpe_ratio()
    model_trainer.calculate_drawdown()
    model_trainer.plot_results()

    # LSTM Model
    lstm_trainer = LSTMTrainer(data)
    X, y = lstm_trainer.prepare_data()
    lstm_trainer.build_model(X.shape[1])
    lstm_trainer.train_model(X, y)

    # Trading API
    trading_api = TradingAPI(api_key='your_api_key', api_secret='your_api_secret')
    trading_api.connect()
    trading_api.get_account_balance()
    trading_api = TradingAPI(api_key='your_api_key', api_secret='your_api_secret')

if __name__ == "__main__":
    main() 
if __name__ == "__main__":
    main()
