from data_handler import DataHandler
from model_trainer import ModelTrainer

def main():
    # Load and prepare data
    data_handler = DataHandler()
    data_handler.load_data()
    data = data_handler.feature_engineering()

    # Train model and backtest
    model_trainer = ModelTrainer(data)
    model_trainer.train_model()
    model_trainer.backtest()
    model_trainer.plot_results()

if __name__ == "__main__":
    main() 