import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, SimpleRNN, GRU, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import json
import os
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
from datetime import timedelta
from statsmodels.tsa.arima.model import ARIMA

class ImprovedPredictor:
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
        data.index = pd.to_datetime(data.index)
        data.index.freq = data.index.inferred_freq  # Set the frequency
        return data[['Close', 'Open', 'High', 'Low', 'Volume']]

    def prepare_data(self, data, steps):
        scaled_data = self.scaler.fit_transform(data)
        X, y = [], []
        for i in range(steps, len(scaled_data)):
            X.append(scaled_data[i-steps:i])
            y.append(scaled_data[i, 0])  # Predicting 'Close' price
        return np.array(X), np.array(y)

    def train_lstm(self, X_train, y_train):
        model = Sequential([
            LSTM(100, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
            Dropout(0.2),
            LSTM(100, return_sequences=False),
            Dropout(0.2),
            Dense(50),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        model.fit(X_train, y_train, validation_split=0.2, batch_size=self.config['LSTM']['batch_size'], 
                  epochs=self.config['LSTM']['epochs'], callbacks=[early_stopping])
        return model

    def train_rnn(self, X_train, y_train):
        model = Sequential([
            SimpleRNN(100, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
            Dropout(0.2),
            SimpleRNN(100, return_sequences=False),
            Dropout(0.2),
            Dense(50),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        model.fit(X_train, y_train, validation_split=0.2, batch_size=self.config['RNN']['batch_size'], 
                  epochs=self.config['RNN']['epochs'], callbacks=[early_stopping])
        return model

    def train_gru(self, X_train, y_train):
        model = Sequential([
            GRU(100, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
            Dropout(0.2),
            GRU(100, return_sequences=False),
            Dropout(0.2),
            Dense(50),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        model.fit(X_train, y_train, validation_split=0.2, batch_size=self.config['GRU']['batch_size'], 
                  epochs=self.config['GRU']['epochs'], callbacks=[early_stopping])
        return model

    def train_linear_regression(self, X_train, y_train):
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        model = LinearRegression()
        model.fit(X_train_flat, y_train)
        return model

    def train_random_forest(self, X_train, y_train):
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        model = RandomForestRegressor(n_estimators=self.config['RandomForest']['n_estimators'], random_state=42)
        model.fit(X_train_flat, y_train)
        return model

    def train_arima(self, data):
        model = ARIMA(data, order=(5,1,0))
        return model.fit()

    def save_model(self, model, symbol, algo):
        model_path = os.path.join(self.models_dir, f"{symbol}_{algo}")
        if algo in ['LSTM', 'RNN', 'GRU']:
            model.save(model_path + ".keras")
        elif algo == 'ARIMA':
            joblib.dump(model, model_path + ".joblib")
        else:
            joblib.dump(model, model_path + ".joblib")
        return model_path

    def load_model(self, symbol, algo):
        model_path = os.path.join(self.models_dir, f"{symbol}_{algo}")
        if algo in ['LSTM', 'RNN', 'GRU']:
            if os.path.exists(model_path + ".keras"):
                return load_model(model_path + ".keras")
        else:
            joblib_model_path = model_path + ".joblib"
            if os.path.exists(joblib_model_path):
                return joblib.load(joblib_model_path)
        return None

    def calculate_confidence(self, y_true, y_pred):
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        
        # Normalize metrics
        max_price = np.max(y_true)
        normalized_mae = 1 - (mae / max_price)
        normalized_rmse = 1 - (rmse / max_price)
        
        # Combine metrics for overall confidence
        confidence = (normalized_mae + normalized_rmse + r2) / 3 * 100
        return max(0, min(confidence, 100))  # Ensure confidence is between 0 and 100

    def predict_future(self, symbol, short_term_days=30, long_term_days=365):
        data = self.get_stock_data(symbol)
        X, y = self.prepare_data(data, self.config['LSTM']['steps'])

        predictions = {}
        confidences = {}
        last_date = data.index[-1]

        # Split data for training and testing
        split = int(0.8 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        for algo in ['LSTM', 'RNN', 'GRU', 'LinearRegression', 'RandomForest', 'ARIMA']:
            model = self.load_model(symbol, algo)
            if model is None:
                print(f"Model for {algo} not found. Please train the model first.")
                continue

            if algo in ['LSTM', 'RNN', 'GRU']:
                short_term_pred = []
                long_term_pred = []
                last_sequence = X[-1]
                for _ in range(max(short_term_days, long_term_days)):
                    next_pred = model.predict(last_sequence.reshape(1, X.shape[1], X.shape[2]))
                    last_sequence = np.roll(last_sequence, -1, axis=0)
                    last_sequence[-1] = next_pred
                    if _ < short_term_days:
                        short_term_pred.append(next_pred[0, 0])
                    long_term_pred.append(next_pred[0, 0])
                
                short_term_pred = self.scaler.inverse_transform(np.hstack([np.array(short_term_pred).reshape(-1, 1), np.zeros((len(short_term_pred), 4))]))[:, 0]
                long_term_pred = self.scaler.inverse_transform(np.hstack([np.array(long_term_pred).reshape(-1, 1), np.zeros((len(long_term_pred), 4))]))[:, 0]
                
                # Calculate confidence
                y_pred = model.predict(X_test)
                y_pred = self.scaler.inverse_transform(np.hstack([y_pred, np.zeros((y_pred.shape[0], 4))]))[:, 0]
            
            elif algo in ['LinearRegression', 'RandomForest']:
                last_sequence = X[-1].flatten()
                short_term_pred = []
                long_term_pred = []
                for _ in range(max(short_term_days, long_term_days)):
                    next_pred = model.predict(last_sequence.reshape(1, -1))
                    last_sequence = np.roll(last_sequence, -X.shape[2])
                    last_sequence[-1] = next_pred
                    if _ < short_term_days:
                        short_term_pred.append(next_pred[0])
                    long_term_pred.append(next_pred[0])
                
                short_term_pred = self.scaler.inverse_transform(np.hstack([np.array(short_term_pred).reshape(-1, 1), np.zeros((len(short_term_pred), 4))]))[:, 0]
                long_term_pred = self.scaler.inverse_transform(np.hstack([np.array(long_term_pred).reshape(-1, 1), np.zeros((len(long_term_pred), 4))]))[:, 0]
                
                # Calculate confidence
                y_pred = model.predict(X_test.reshape(X_test.shape[0], -1))
                y_pred = self.scaler.inverse_transform(np.hstack([y_pred.reshape(-1, 1), np.zeros((y_pred.shape[0], 4))]))[:, 0]
            
            elif algo == 'ARIMA':
                forecast = model.forecast(steps=max(short_term_days, long_term_days))
                short_term_pred = forecast[:short_term_days].values
                long_term_pred = forecast.values
                
                # Calculate confidence
                y_pred = model.forecast(steps=len(y_test)).values

            predictions[algo] = {
                'short_term': short_term_pred,
                'long_term': long_term_pred
            }
            
            y_true = self.scaler.inverse_transform(np.hstack([y_test.reshape(-1, 1), np.zeros((y_test.shape[0], 4))]))[:, 0]
            confidences[algo] = self.calculate_confidence(y_true, y_pred)

        # Calculate ensemble predictions
        ensemble_short_term = np.mean([pred['short_term'] for pred in predictions.values()], axis=0)
        ensemble_long_term = np.mean([pred['long_term'] for pred in predictions.values()], axis=0)
        predictions['Ensemble'] = {
            'short_term': ensemble_short_term,
            'long_term': ensemble_long_term
        }
        confidences['Ensemble'] = np.mean(list(confidences.values()))

        return predictions, confidences, last_date
    def save_prediction_graph(self, symbol, predictions, last_date, short_term_days=30, long_term_days=365):
        data = self.get_stock_data(symbol)
        actual_prices = data['Close'].values

        plt.figure(figsize=(15, 10))
        plt.plot(data.index, actual_prices, label='Actual Price', color='black')

        colors = plt.cm.rainbow(np.linspace(0, 1, len(predictions)))
        for (algo, pred), color in zip(predictions.items(), colors):
            short_term_dates = pd.date_range(start=last_date + timedelta(days=1), periods=short_term_days)
            long_term_dates = pd.date_range(start=last_date + timedelta(days=1), periods=long_term_days)
            
            plt.plot(short_term_dates, pred['short_term'], label=f'{algo} Short-term', color=color, linestyle='--')
            plt.plot(long_term_dates, pred['long_term'], label=f'{algo} Long-term', color=color, linestyle=':')

        plt.title(f'Stock Price Predictions for {symbol}')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.tight_layout()
        plt.savefig(f'{symbol}_predictions.png')
        plt.close()

    def predict_and_visualize(self, symbol, short_term_days=30, long_term_days=365):
        predictions, confidences, last_date = self.predict_future(symbol, short_term_days, long_term_days)
        self.save_prediction_graph(symbol, predictions, last_date, short_term_days, long_term_days)
        
        results = {
            'predictions': predictions,
            'confidences': confidences,
            'last_date': last_date
        }
        
        return results
# Example usage
if __name__ == "__main__":
    predictor = ImprovedPredictor()
    symbol = "MSFT"
    results = predictor.predict_and_visualize(symbol)
    
    print(f"Predictions for {symbol} have been generated and saved as '{symbol}_predictions.png'")
    print("\nConfidences:")
    for algo, confidence in results['confidences'].items():
        print(f"{algo}: {confidence:.2f}%")
    
    print("\nShort-term predictions (first 5 days):")
    for algo, pred in results['predictions'].items():
        print(f"{algo}: {pred['short_term'][:5]}")
    
    print("\nLong-term predictions (last 5 days):")
    for algo, pred in results['predictions'].items():
        print(f"{algo}: {pred['long_term'][-5:]}")