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
import time
from joblib import Parallel, delayed

class ImprovedPredictor:
    def __init__(self, config_file='ds_config.json'):
        self.config = self.load_config(config_file)
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.models_dir = 'models'
        self.loaded_models = {}
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)

    def load_config(self, config_file):
        with open(config_file, 'r') as file:
            return json.load(file)

    def get_stock_data(self, symbol):
        data = yf.download(symbol, period="5y", interval="1d")
        data.index = pd.to_datetime(data.index)
        data.index.freq = data.index.inferred_freq  # Set the frequency
        return data[['Close', 'Open', 'High', 'Low', 'Volume']].astype('float32')

    def prepare_data(self, data, steps):
        scaled_data = self.scaler.fit_transform(data)
        X, y = [], []
        for i in range(steps, len(scaled_data)):
            X.append(scaled_data[i-steps:i])
            y.append(scaled_data[i, 0])  # Predicting 'Close' price
        return np.array(X), np.array(y)

    def is_model_recent(self, model_path, max_age_hours=12):
        if os.path.exists(model_path):
            model_age = time.time() - os.path.getmtime(model_path)
            return model_age < max_age_hours * 3600
        return False

    def load_model_cached(self, symbol, algo):
        key = f"{symbol}_{algo}"
        if key in self.loaded_models:
            return self.loaded_models[key]
        
        model_path = os.path.join(self.models_dir, f"{symbol}_{algo}")
        if algo in ['LSTM', 'RNN', 'GRU'] and self.is_model_recent(model_path + ".keras"):
            model = load_model(model_path + ".keras")
        elif self.is_model_recent(model_path + ".joblib"):
            model = joblib.load(model_path + ".joblib")
        else:
            model = None
        
        if model:
            self.loaded_models[key] = model
        return model

    def train_lstm(self, X_train, y_train):
        print('Training LSTM model...')
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

    def train_and_save_model(self,  X_train, y_train, symbol, algo):
        print(f"Training {symbol} {algo} model...")
        if algo == 'LSTM':
            model = self.train_lstm(X_train, y_train)
        elif algo == 'RNN':
            model = self.train_rnn(X_train, y_train)
        elif algo == 'GRU':
            model = self.train_gru(X_train, y_train)
        elif algo == 'LinearRegression':
            model = self.train_linear_regression(X_train, y_train)
        elif algo == 'RandomForest':
            model = self.train_random_forest(X_train, y_train)
        elif algo == 'ARIMA':
            model = self.train_arima(X_train)
        if algo in ['LSTM', 'RNN', 'GRU']:
            model_path = f"models/{symbol}_{algo}.keras"
            model.save(model_path)
        elif algo == 'LinearRegression':
            model_path = f"models/{symbol}_LinearRegression.joblib"
            #save model with joblib
            
        elif algo == 'RandomForest':
            model_path = f"models/{symbol}_RandomForest.joblib"
            self.save_model(model, model_path)
        elif algo == 'ARIMA':
            model_path = f"models/{symbol}_ARIMA.joblib"
            self.save_model(model, model_path)
    def predict_in_batches(self, model, last_sequence, steps, X_shape):
        predictions = []
        for _ in range(steps // 30 + 1):  # Predict in batches of 30
            batch_pred = model.predict(np.tile(last_sequence, (30, 1, 1)))
            predictions.extend(batch_pred)
            last_sequence = np.roll(last_sequence, -1, axis=0)
            last_sequence[-1] = batch_pred[-1]
        return predictions[:steps]

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

    def filter_models_by_confidence(self, predictions, confidences, threshold=50):
        filtered_predictions = {algo: pred for algo, pred in predictions.items() if confidences[algo] > threshold}
        filtered_confidences = {algo: conf for algo, conf in confidences.items() if conf > threshold}
        return filtered_predictions, filtered_confidences

    def weighted_ensemble(self, predictions, confidences):
        weighted_preds = np.average([pred['short_term'] for pred in predictions.values()], 
                                    axis=0, 
                                    weights=list(confidences.values()))
        return weighted_preds

    
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
            model = self.load_model_cached(symbol, algo)
            if model is None:
                print(f"Training {symbol} {algo} model...")
                model = self.train_and_save_model(X_train, y_train, symbol, algo)
                model = self.load_model_cached(symbol, algo)

            if algo in ['LSTM', 'RNN', 'GRU']:
                y_pred = model.predict(X_test)
                y_pred = self.scaler.inverse_transform(np.hstack([y_pred, np.zeros((y_pred.shape[0], 4))]))[:, 0]

                last_sequence = X[-1]
                short_term_pred = []
                long_term_pred = []
                for _ in range(max(short_term_days, long_term_days)):
                    next_pred = model.predict(last_sequence.reshape(1, X.shape[1], X.shape[2]))
                    last_sequence = np.roll(last_sequence, -1, axis=0)
                    last_sequence[-1] = next_pred
                    if _ < short_term_days:
                        short_term_pred.append(next_pred[0, 0])
                    long_term_pred.append(next_pred[0, 0])

                short_term_pred = self.scaler.inverse_transform(np.hstack([np.array(short_term_pred).reshape(-1, 1), np.zeros((len(short_term_pred), 4))]))[:, 0]
                long_term_pred = self.scaler.inverse_transform(np.hstack([np.array(long_term_pred).reshape(-1, 1), np.zeros((len(long_term_pred), 4))]))[:, 0]

            elif algo in ['LinearRegression', 'RandomForest']:
                X_test_flat = X_test.reshape(X_test.shape[0], -1)
                y_pred = model.predict(X_test_flat)
                
                last_sequence_flat = X[-1].reshape(1, -1)
                short_term_pred = model.predict(np.tile(last_sequence_flat, (short_term_days, 1)))
                long_term_pred = model.predict(np.tile(last_sequence_flat, (long_term_days, 1)))

            elif algo == 'ARIMA':
                model = self.train_arima(data['Close'].values)
                y_pred = model.forecast(steps=len(X_test))
                short_term_pred = model.forecast(steps=short_term_days)
                long_term_pred = model.forecast(steps=long_term_days)

            # Calculate confidence
            confidence = self.calculate_confidence(y_test, y_pred)
            predictions[algo] = {'short_term': short_term_pred, 'long_term': long_term_pred}
            confidences[algo] = confidence

        # Filter models by confidence threshold
        predictions, confidences = self.filter_models_by_confidence(predictions, confidences)

        if not predictions:
            raise ValueError("No models passed the confidence threshold. Can't compute ensemble.")

        # Ensemble predictions
        ensemble_short_term = self.weighted_ensemble(predictions, confidences)
        ensemble_long_term = np.mean([pred['long_term'] for pred in predictions.values()], axis=0)

        return predictions, confidences, ensemble_short_term, ensemble_long_term, last_date

    def save_prediction_graph(self, symbol, predictions, confidences, ensemble_short_term, ensemble_long_term, last_date, short_term_days, long_term_days):
        plt.figure(figsize=(15, 10))
        
        short_term_dates = pd.date_range(start=last_date + timedelta(days=1), periods=short_term_days)
        long_term_dates = pd.date_range(start=last_date + timedelta(days=1), periods=long_term_days)
        
        colors = plt.cm.rainbow(np.linspace(0, 1, len(predictions)))
        
        for (algo, pred), color in zip(predictions.items(), colors):
            plt.plot(short_term_dates, pred['short_term'], label=f'{algo} Short-term (Conf: {confidences[algo]:.2f}%)', color=color, linestyle='--')
            plt.plot(long_term_dates, pred['long_term'], label=f'{algo} Long-term', color=color, linestyle=':')
        
        plt.plot(short_term_dates, ensemble_short_term, label='Ensemble Short-term', color='black', linewidth=2)
        plt.plot(long_term_dates, ensemble_long_term, label='Ensemble Long-term', color='black', linewidth=2, linestyle='--')
        
        plt.title(f'{symbol} Stock Price Prediction')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'{symbol}_prediction.png')
        plt.close()

    def predict_and_visualize(self, symbol, short_term_days=30, long_term_days=365):
        predictions, confidences, ensemble_short_term, ensemble_long_term, last_date = self.predict_future(symbol, short_term_days, long_term_days)
        self.save_prediction_graph(symbol, predictions, confidences, ensemble_short_term, ensemble_long_term, last_date, short_term_days, long_term_days)
        return {
            'predictions': predictions,
            'confidences': confidences,
            'ensemble_short_term': ensemble_short_term.tolist(),
            'ensemble_long_term': ensemble_long_term.tolist(),
            'last_date': last_date.strftime('%Y-%m-%d')
        }
if __name__ == '__main__':
    predictor = ImprovedPredictor()
    symbol = 'MSFT'
    results = predictor.predict_and_visualize(symbol)
    print(f"Predictions for {symbol}:")
    print(json.dumps(results, indent=2))