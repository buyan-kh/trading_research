import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

class ModelTrainer:
    def __init__(self, data):
        self.data = data
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.features = ['SMA_10', 'SMA_50', 'RSI']
        self.X = self.data[self.features]
        self.y = np.where(self.data['Return'].shift(-1) > 0, 1, 0)

    def train_model(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f'Accuracy: {accuracy:.2f}')

    def backtest(self):
        self.data['Prediction'] = self.model.predict(self.X)
        self.data['Strategy_Return'] = self.data['Return'] * self.data['Prediction'].shift(1)
        self.data['Cumulative_Market_Return'] = (1 + self.data['Return']).cumprod()
        self.data['Cumulative_Strategy_Return'] = (1 + self.data['Strategy_Return']).cumprod()

    def plot_results(self):
        plt.figure(figsize=(12, 6))
        plt.plot(self.data['Cumulative_Market_Return'], label='Market Return')
        plt.plot(self.data['Cumulative_Strategy_Return'], label='Strategy Return')
        plt.legend()
        plt.show() 