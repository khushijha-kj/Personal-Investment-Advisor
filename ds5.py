import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, SimpleRNN, GRU
import json
import os
import joblib
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

class Predictor:
    def __init__(self, config_file='ds_config.json'):
        self.config = self.load_config(config_file)
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.models_dir = 'models'
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)

    def load_config(self, config_file):
        with open(config_file, 'r') as file:
            return json.load(file)

    def get_stock_data(self, symbol):
        data = yf.download(symbol, period="5y", interval="1d")
        return data['Close'].values.reshape(-1, 1)

    def prepare_data(self, data, steps):
        X, y = [], []
        for i in range(steps, len(data)):
            X.append(data[i-steps:i, 0])
            y.append(data[i, 0])
        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        return X, y

    def train_lstm(self, X_train, y_train):
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
        model.add(LSTM(50, return_sequences=False))
        model.add(Dense(25))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, y_train, batch_size=self.config['LSTM']['batch_size'], epochs=self.config['LSTM']['epochs'])
        return model

    def train_rnn(self, X_train, y_train):
        model = Sequential()
        model.add(SimpleRNN(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
        model.add(SimpleRNN(50, return_sequences=False))
        model.add(Dense(25))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, y_train, batch_size=self.config['RNN']['batch_size'], epochs=self.config['RNN']['epochs'])
        return model

    def train_gru(self, X_train, y_train):
        model = Sequential()
        model.add(GRU(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
        model.add(GRU(50, return_sequences=False))
        model.add(Dense(25))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, y_train, batch_size=self.config['GRU']['batch_size'], epochs=self.config['GRU']['epochs'])
        return model

    def train_linear_regression(self, X_train, y_train):
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        model = LinearRegression()
        model.fit(X_train_flat, y_train)
        return model

    def train_random_forest(self, X_train, y_train):
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        model = RandomForestRegressor(n_estimators=self.config['RandomForest']['n_estimators'])
        model.fit(X_train_flat, y_train)
        return model

    def save_model(self, model, symbol, algo):
        model_path = os.path.join(self.models_dir, f"{symbol}_{algo}")
        if algo in ['LSTM', 'RNN', 'GRU']:
            model.save(model_path + ".keras")
        else:
            joblib.dump(model, model_path + ".joblib")
        return model_path

    def load_model(self, symbol, algo):
        model_path = os.path.join(self.models_dir, f"{symbol}_{algo}")
        if algo in ['LSTM', 'RNN', 'GRU']:
            if os.path.exists(model_path + ".keras"):
                model_time = os.path.getmtime(model_path + ".keras")
                if (datetime.now() - datetime.fromtimestamp(model_time)).days < 7:  # Model is less than 7 days old
                    return load_model(model_path + ".keras")
        else:
            joblib_model_path = model_path + ".joblib"
            if os.path.exists(joblib_model_path):
                model_time = os.path.getmtime(joblib_model_path)
                if (datetime.now() - datetime.fromtimestamp(model_time)).days < 7:  # Model is less than 7 days old
                    return joblib.load(joblib_model_path)
        return None

    def predict(self, symbol, steps=None):
        if steps is None:
            steps = self.config['LSTM']['steps']
        data = self.get_stock_data(symbol)
        scaled_data = self.scaler.fit_transform(data)
        X, y = self.prepare_data(scaled_data, steps)

        predictions = {}
        confidences = {}
        models = {}

        for algo in ['LSTM', 'RNN', 'GRU', 'LinearRegression', 'RandomForest']:
            model = self.load_model(symbol, algo)
            if model is None:
                if algo == 'LSTM':
                    model = self.train_lstm(X, y)
                elif algo == 'RNN':
                    model = self.train_rnn(X, y)
                elif algo == 'GRU':
                    model = self.train_gru(X, y)
                elif algo == 'LinearRegression':
                    model = self.train_linear_regression(X, y)
                elif algo == 'RandomForest':
                    model = self.train_random_forest(X, y)
                self.save_model(model, symbol, algo)
            
            models[algo] = model

            if algo in ['LSTM', 'RNN', 'GRU']:
                pred = model.predict(X[-1].reshape(1, steps, 1))
            else:
                pred = model.predict(X[-1].reshape(1, -1))
            
            predictions[algo] = self.scaler.inverse_transform(pred.reshape(-1, 1))[0][0]
            
            # Calculate a simple confidence score based on recent performance
            recent_predictions = model.predict(X[-30:])
            recent_mse = mean_squared_error(y[-30:], recent_predictions)
            confidences[algo] = max(0, 100 - recent_mse)  # Simple scaling, can be improved

        return predictions, confidences, models

    def future_predictions(self, symbol, models):
        data = self.get_stock_data(symbol)
        scaled_data = self.scaler.transform(data)
        steps = self.config['LSTM']['steps']

        future_dates = [
            datetime.now() + timedelta(days=1),
            datetime.now() + timedelta(weeks=1),
            datetime.now() + timedelta(days=30),
            datetime.now() + timedelta(days=90)
        ]

        future_predictions = {algo: [] for algo in models.keys()}

        for _ in range(90):  # Predict for the next 90 days
            X_future = scaled_data[-steps:].reshape(1, steps, 1)
            
            for algo, model in models.items():
                if algo in ['LSTM', 'RNN', 'GRU']:
                    pred = model.predict(X_future)
                else:
                    pred = model.predict(X_future.reshape(1, -1))
                
                future_predictions[algo].append(self.scaler.inverse_transform(pred.reshape(-1, 1))[0][0])
            
            # Add the prediction to the data for the next iteration
            scaled_data = np.vstack((scaled_data, pred))

        return future_predictions, future_dates

    def visualize_predictions(self, symbol, current_price, predictions, future_predictions, future_dates):
        plt.figure(figsize=(15, 10))
        
        # Plot historical data
        data = self.get_stock_data(symbol)
        plt.plot(data, label='Historical Data', color='black')
        
        # Plot current price
        plt.scatter(len(data) - 1, current_price, color='blue', s=100, zorder=5, label='Current Price')
        
        # Plot predictions for each model
        colors = {'LSTM': 'red', 'RNN': 'green', 'GRU': 'purple', 'LinearRegression': 'orange', 'RandomForest': 'brown'}
        for algo, pred in predictions.items():
            plt.scatter(len(data), pred, color=colors[algo], s=100, zorder=5, label=f'{algo} Prediction')
        
        # Plot future predictions
        for algo, preds in future_predictions.items():
            future_dates_num = [i for i in range(len(data), len(data) + len(preds))]
            plt.plot(future_dates_num, preds, color=colors[algo], linestyle='--', alpha=0.7)
        
        # Mark specific future predictions
        for i, date in enumerate(future_dates):
            for algo, preds in future_predictions.items():
                plt.scatter(len(data) + i, preds[i], color=colors[algo], s=100, zorder=5)
                plt.annotate(f"{date.strftime('%Y-%m-%d')}", (len(data) + i, preds[i]), textcoords="offset points", xytext=(0,10), ha='center')
        
        plt.title(f'Stock Price Predictions for {symbol}')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{symbol}_predictions.png')
        plt.close()

    def predict_and_visualize(self, symbol):
        current_data = self.get_stock_data(symbol)
        current_price = current_data[-1][0]
        
        predictions, confidences, models = self.predict(symbol)
        future_predictions, future_dates = self.future_predictions(symbol, models)
        
        self.visualize_predictions(symbol, current_price, predictions, future_predictions, future_dates)
        
        return predictions, confidences, future_predictions, future_dates

# Example usage
if __name__ == "__main__":
    predictor = Predictor()
    symbol = "NVDA"
    predictions, confidences, future_predictions, future_dates = predictor.predict_and_visualize(symbol)
    
    print(f"Current predictions for {symbol}:")
    for algo, pred in predictions.items():
        print(f"{algo}: ${pred:.2f} (Confidence: {confidences[algo]:.2f}%)")
    
    print("\nFuture predictions:")
    for i, date in enumerate(future_dates):
        print(f"\n{date.strftime('%Y-%m-%d')}:")
        for algo, preds in future_predictions.items():
            print(f"  {algo}: ${preds[i]:.2f}")
    
    print(f"\nPrediction graph saved as {symbol}_predictions.png")