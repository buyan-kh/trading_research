import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

class ModelTrainer:
    def __init__(self, data):
        self.data = data
        self.features = [col for col in ['SMA_10', 'SMA_50', 'RSI', 'MACD', 'BB_upper', 'BB_middle', 'BB_lower', 'OBV'] if col in self.data.columns]
        self.X = self.data[self.features]
        self.y = np.where(self.data['Return'].shift(-1) > 0, 1, 0)

    def train_model(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10]
        }
        grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, scoring='accuracy')
        grid_search.fit(X_train, y_train)
        self.model = grid_search.best_estimator_
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f'Accuracy: {accuracy:.2f}')
        print(f'Best Parameters: {grid_search.best_params_}')

    def backtest(self, transaction_cost=0.001, stop_loss=0.02, take_profit=0.05):
        self.data['Prediction'] = self.model.predict(self.X)
        self.data['Strategy_Return'] = self.data['Return'] * self.data['Prediction'].shift(1)
        self.data['Strategy_Return'] -= transaction_cost * np.abs(self.data['Prediction'].diff())

        # Implement stop-loss and take-profit
        self.data['Strategy_Return'] = np.where(self.data['Return'] < -stop_loss, -stop_loss, self.data['Strategy_Return'])
        self.data['Strategy_Return'] = np.where(self.data['Return'] > take_profit, take_profit, self.data['Strategy_Return'])

        self.data['Cumulative_Market_Return'] = (1 + self.data['Return']).cumprod()
        self.data['Cumulative_Strategy_Return'] = (1 + self.data['Strategy_Return']).cumprod()

    def calculate_sharpe_ratio(self):
        excess_return = self.data['Strategy_Return'] - 0.0001  # Assuming a risk-free rate of 0.01% per day
        sharpe_ratio = np.sqrt(252) * excess_return.mean() / excess_return.std()
        print(f'Sharpe Ratio: {sharpe_ratio:.2f}')

    def calculate_drawdown(self):
        cum_return = self.data['Cumulative_Strategy_Return']
        drawdown = cum_return / cum_return.cummax() - 1
        max_drawdown = drawdown.min()
        print(f'Max Drawdown: {max_drawdown:.2%}')

    def plot_results(self):
        plt.figure(figsize=(12, 6))
        plt.plot(self.data['Cumulative_Market_Return'], label='Market Return')
        plt.plot(self.data['Cumulative_Strategy_Return'], label='Strategy Return')
        plt.legend()
        plt.show() 