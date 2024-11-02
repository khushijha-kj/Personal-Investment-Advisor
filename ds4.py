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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

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

    def train_linear_regression(self, X_train, y_train, steps):
        X_train_flat = X_train.reshape(X_train.shape[0], steps)  # Flatten the input to fit LinearRegression
        model = LinearRegression()
        model.fit(X_train_flat, y_train)
        return model

    def train_random_forest(self, X_train, y_train, steps):
        X_train_flat = X_train.reshape(X_train.shape[0], steps)  # Flatten the input to fit RandomForest
        model = RandomForestRegressor(n_estimators=self.config['RandomForest']['n_estimators'])
        model.fit(X_train_flat, y_train)
        return model

    def save_model(self, model, symbol, algo):
        model_path = os.path.join(self.models_dir, f"{symbol}_{algo}")
        if algo in ['LSTM', 'RNN', 'GRU']:
            model.save(model_path+".keras")  # Use TensorFlow save for LSTM/RNN/GRU
        else:
            joblib.dump(model, model_path + ".joblib")  # Use joblib for traditional models
        return model_path

    def load_model(self, symbol, algo):
        model_path = os.path.join(self.models_dir, f"{symbol}_{algo}")
        if algo in ['LSTM', 'RNN', 'GRU']:
            print(f"Loading {algo} model from {model_path}.keras")
            if os.path.exists(model_path+".keras"):
                return load_model(model_path+".keras")  # Use TensorFlow load for LSTM/RNN/GRU
            else:
                print(f"Model {model_path}.keras not found")
                return None
        else:
            joblib_model_path = model_path + ".joblib"
            if os.path.exists(joblib_model_path):
                return joblib.load(joblib_model_path)  # Use joblib for traditional models
        return None

    def predict(self, symbol, calculate_confidence=True, refresh_models=False, steps=None):
        if steps is None:
            steps = self.config['LSTM']['steps']
        data = self.get_stock_data(symbol)
        scaled_data = self.scaler.fit_transform(data)
        X, y = self.prepare_data(scaled_data, steps)

        predictions = {}
        confidences = {}
        models = {}

        # Load and train models, ensure models are retrained for different step sizes (features)
        # LSTM
        lstm_model = self.load_model(symbol, 'LSTM')
        if lstm_model is None or refresh_models:
            lstm_model = self.train_lstm(X, y)
            self.save_model(lstm_model, symbol, 'LSTM')
        lstm_pred = lstm_model.predict(X[-1].reshape(1, steps, 1))
        predictions['LSTM'] = self.scaler.inverse_transform(lstm_pred)[0][0]
        models['LSTM'] = lstm_model

        # RNN
        rnn_model = self.load_model(symbol, 'RNN')
        if rnn_model is None or refresh_models:
            rnn_model = self.train_rnn(X, y)
            self.save_model(rnn_model, symbol, 'RNN')
        rnn_pred = rnn_model.predict(X[-1].reshape(1, steps, 1))
        predictions['RNN'] = self.scaler.inverse_transform(rnn_pred)[0][0]
        models['RNN'] = rnn_model

        # GRU
        gru_model = self.load_model(symbol, 'GRU')
        if gru_model is None or refresh_models:
            gru_model = self.train_gru(X, y)
            self.save_model(gru_model, symbol, 'GRU')
        gru_pred = gru_model.predict(X[-1].reshape(1, steps, 1))
        predictions['GRU'] = self.scaler.inverse_transform(gru_pred)[0][0]
        models['GRU'] = gru_model

        # Linear Regression
        lr_model = self.load_model(symbol, 'LinearRegression')
        if lr_model is None or refresh_models or X.shape[1] != lr_model.n_features_in_:
            lr_model = self.train_linear_regression(X, y, steps)
            self.save_model(lr_model, symbol, 'LinearRegression')
        lr_pred = lr_model.predict(X[-1].reshape(1, steps))
        predictions['LinearRegression'] = self.scaler.inverse_transform(lr_pred.reshape(-1, 1))[0][0]
        models['LinearRegression'] = lr_model

        # Random Forest
        rf_model = self.load_model(symbol, 'RandomForest')
        if rf_model is None or refresh_models or X.shape[1] != rf_model.n_features_in_:
            rf_model = self.train_random_forest(X, y, steps)
            self.save_model(rf_model, symbol, 'RandomForest')
        rf_pred = rf_model.predict(X[-1].reshape(1, steps))
        predictions['RandomForest'] = self.scaler.inverse_transform(rf_pred.reshape(-1, 1))[0][0]
        models['RandomForest'] = rf_model

        if calculate_confidence:
            # Calculate confidences
            confidences = self.calculate_confidences(predictions, X, y, models, steps)

            return predictions, confidences
        else:
            return predictions, None

    def calculate_confidences(self, predictions, X_train, y_train, models, steps):
        """
        Calculate model confidence based on performance metrics and model-specific considerations.
        For Keras models (LSTM, RNN, GRU), use manual k-fold validation.
        For scikit-learn models, use cross_val_score.
        """
        confidences = {}
        
        for algo, pred in predictions.items():
            model = models[algo]

            if algo in ['LSTM', 'RNN', 'GRU']:
                # Manual k-fold validation for Keras models
                kfold = KFold(n_splits=5)
                cv_scores = []
                
                for train_idx, val_idx in kfold.split(X_train):
                    X_cv_train, X_cv_val = X_train[train_idx], X_train[val_idx]
                    y_cv_train, y_cv_val = y_train[train_idx], y_train[val_idx]
                    
                    model.fit(X_cv_train.reshape(-1, steps, 1), y_cv_train, epochs=5, batch_size=32, verbose=0)
                    y_cv_pred = model.predict(X_cv_val.reshape(-1, steps, 1))
                    mse = mean_squared_error(y_cv_val, y_cv_pred)
                    cv_scores.append(mse)
                
                cross_val_mean_error = np.mean(cv_scores)
                cross_val_std_error = np.std(cv_scores)
                penalty = cross_val_std_error / max(abs(cross_val_mean_error), 1e-6)
                performance_score = 100 - (cross_val_mean_error * 100)
                confidence = max(0, performance_score - penalty * 100)

            elif algo == 'LinearRegression':
                # Linear Regression: traditional metrics
                y_train_pred = model.predict(X_train.reshape(-1, steps))
                rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
                confidence = 100 - rmse

            elif algo == 'RandomForest':
                # RandomForest: Out-of-Bag (OOB) score or cross-validation
                if hasattr(model, 'oob_score_'):
                    confidence = model.oob_score_ * 100  # OOB score if available
                else:
                    cross_val_scores = cross_val_score(model, X_train.reshape(-1, steps), y_train, cv=5, scoring='neg_mean_squared_error')
                    confidence = 100 + np.mean(cross_val_scores)

            confidences[algo] = confidence

        # Ensemble confidence as the average of all models
        ensemble_confidence = np.mean(list(confidences.values()))
        confidences['Ensemble'] = ensemble_confidence

        return confidences

    def calculate_prediction_confidence_score(self, symbol, steps=None):
        """
        Calculate a prediction confidence score based on the performance of the models on the entire dataset.
        The score is a value between 1 and 100, where higher values indicate higher confidence in the predictions.
        """
        if steps is None:
            steps = self.config['LSTM']['steps']
        data = self.get_stock_data(symbol)
        scaled_data = self.scaler.fit_transform(data)
        X, y = self.prepare_data(scaled_data, steps)

        models = {}

        # Load or train models
        # LSTM
        lstm_model = self.load_model(symbol, 'LSTM')
        if lstm_model is None:
            lstm_model = self.train_lstm(X, y)
            self.save_model(lstm_model, symbol, 'LSTM')
        models['LSTM'] = lstm_model

        # RNN
        rnn_model = self.load_model(symbol, 'RNN')
        if rnn_model is None:
            rnn_model = self.train_rnn(X, y)
            self.save_model(rnn_model, symbol, 'RNN')
        models['RNN'] = rnn_model

        # GRU
        gru_model = self.load_model(symbol, 'GRU')
        if gru_model is None:
            gru_model = self.train_gru(X, y)
            self.save_model(gru_model, symbol, 'GRU')
        models['GRU'] = gru_model

        # Linear Regression
        lr_model = self.load_model(symbol, 'LinearRegression')
        if lr_model is None or X.shape[1] != lr_model.n_features_in_:
            lr_model = self.train_linear_regression(X, y, steps)
            self.save_model(lr_model, symbol, 'LinearRegression')
        models['LinearRegression'] = lr_model

        # Random Forest
        rf_model = self.load_model(symbol, 'RandomForest')
        if rf_model is None or X.shape[1] != rf_model.n_features_in_:
            rf_model = self.train_random_forest(X, y, steps)
            self.save_model(rf_model, symbol, 'RandomForest')
        models['RandomForest'] = rf_model

        # Calculate confidence scores for each model
        confidences = self.calculate_confidences({}, X, y, models, steps)

        # Calculate the overall prediction confidence score
        prediction_confidence_score = np.mean(list(confidences.values()))

        return prediction_confidence_score

    def short_term_prediction(self, symbol, refresh_models=True, calculate_confidence=True):
        return self.predict(symbol, calculate_confidence=calculate_confidence, refresh_models=refresh_models)

    def long_term_prediction(self, symbol, refresh_models=True, calculate_confidence=True):
        # For long-term prediction, we can use a larger time window or different models
        long_steps = self.config['LSTM']['steps'] * 2
        return self.predict(symbol, calculate_confidence=calculate_confidence, refresh_models=refresh_models, steps=long_steps)

# Example usage
if __name__ == "__main__":
    predictor = Predictor()
    symbol = "AAPL"
    short_predictions, short_confidences = predictor.short_term_prediction(symbol, calculate_confidence=True, refresh_models=False)
    long_predictions, long_confidences = predictor.long_term_prediction(symbol, calculate_confidence=True, refresh_models=False)
    prediction_confidence_score = predictor.calculate_prediction_confidence_score(symbol)
    
    print(f"Short-term Predictions for {symbol}: {short_predictions}")
    print(f"Short-term Confidences for {symbol}: {short_confidences}")
    print(f"Long-term Predictions for {symbol}: {long_predictions}")
    print(f"Long-term Confidences for {symbol}: {long_confidences}")
    print(f"Prediction Confidence Score for {symbol}: {prediction_confidence_score}")