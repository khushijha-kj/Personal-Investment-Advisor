import os
import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, SimpleRNN, GRU, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from datetime import datetime
import tensorflow as tf
import tensorflow_probability as tfp

# Add to the top of your file:
tfd = tfp.distributions
# Helper function to check if the model file exists and is less than 5 days old
def is_model_fresh(model_path):
    model_path='models/'+model_path
    if os.path.exists(model_path):
        last_modified_time = os.path.getmtime(model_path)
        file_age_days = (datetime.now() - datetime.fromtimestamp(last_modified_time)).days
        return file_age_days < 5
    return False

# Function to create a model (LSTM, RNN, GRU)
def create_model(model_type, input_shape):
    model = Sequential()
    if model_type == 'lstm':
        model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape, dropout=0.2, recurrent_dropout=0.2))
    elif model_type == 'rnn':
        model.add(SimpleRNN(units=50, return_sequences=True, input_shape=input_shape, dropout=0.2, recurrent_dropout=0.2))
    elif model_type == 'gru':
        model.add(GRU(units=50, return_sequences=True, input_shape=input_shape, dropout=0.2, recurrent_dropout=0.2))
    
    model.add(Dropout(0.2))
    model.add(Dense(units=25))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model
# Train and save a new model
def train_and_save_model(model, X_train, y_train, model_path):
    model.fit(X_train, y_train, epochs=600
    , batch_size=16)
    model.save('models/'+model_path)

# Function to generate predictions
def predict_future(model, data, days):
    predictions = []
    input_seq = data[-60:].reshape(1, 60, 1)  # Reshape to (1, 60, 1) for model input
    for _ in range(days):
        pred = model.predict(input_seq)  # Model returns a 2D array (1, 1)
        predictions.append(pred[0, 0])  # Extract the predicted value

        # Reshape the predicted value and append it to the input sequence, maintaining 3D shape
        input_seq = np.append(input_seq[:, 1:, :], [[pred[0, 0]]], axis=1)
        input_seq = input_seq.reshape(1, 60, 1)  # Ensure it's reshaped to (1, 60, 1)

    return np.array(predictions)

# Main function
# Function to compute prediction with uncertainty
def predict_with_confidence(model, data, days, n_iter=100):
    predictions = []
    input_seq = data[-60:].reshape(1, 60, 1)  # Reshape to (1, 60, 1) for model input

    for day in range(days):
        preds = []
        for _ in range(n_iter):
            pred = model(input_seq, training=True)  # Set training=True to enable dropout
            preds.append(pred.numpy()[0, 0])
        
        pred_mean = np.mean(preds)
        pred_std = np.std(preds)
        predictions.append((pred_mean, pred_std))

        # Reshape the predicted value and append it to the input sequence, maintaining 3D shape
        pred_mean_reshaped = np.array([[pred_mean]]).reshape(1, 1, 1)
        input_seq = np.append(input_seq[:, 1:, :], pred_mean_reshaped, axis=1)
        print(f"Day {day + 1}/{days} complete")
    
    return np.array(predictions)

def get_stock_predictions(symbol, with_confidence=True):
    # File paths for the models
    lstm_model_path = f'{symbol}_lstm.h5'
    rnn_model_path = f'{symbol}_rnn.h5'
    gru_model_path = f'{symbol}_gru.h5'
    
    # Fetch historical data
    stock = yf.Ticker(symbol)
    hist = stock.history(period="1y")
    
    # Handle missing or zero data
    if hist['Close'].isnull().any():
        print(f"Warning: Missing data detected for {symbol}. Filling missing values.")
        hist['Close'].fillna(method='ffill', inplace=True)
        hist['Close'].fillna(method='bfill', inplace=True)
    
    if hist['Close'].nunique() == 1:
        raise ValueError(f"Constant values detected in stock data for {symbol}. Model cannot train on constant data.")
    
    # Prepare data
    data = hist['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    if np.isnan(scaled_data).any():
        raise ValueError(f"NaN detected in scaled data for {symbol}. Check for missing or constant data.")
    
    # Split data for training
    train_size = int(len(scaled_data) * 0.8)
    train_data = scaled_data[:train_size]
    X_train, y_train = [], []
    for i in range(60, len(train_data)):
        X_train.append(train_data[i-60:i, 0])
        y_train.append(train_data[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    
    if np.isnan(X_train).any() or np.isnan(y_train).any():
        raise ValueError("NaN detected in training data.")
    
    # Load or create models
    if is_model_fresh(lstm_model_path):
        lstm_model = load_model(lstm_model_path, custom_objects={'tf': tf})
    else:
        lstm_model = create_model('lstm', (X_train.shape[1], 1))
        train_and_save_model(lstm_model, X_train, y_train, lstm_model_path)
    
    if is_model_fresh(rnn_model_path):
        rnn_model = load_model(rnn_model_path, custom_objects={'tf': tf})
    else:
        rnn_model = create_model('rnn', (X_train.shape[1], 1))
        train_and_save_model(rnn_model, X_train, y_train, rnn_model_path)
    
    if is_model_fresh(gru_model_path):
        gru_model = load_model(gru_model_path, custom_objects={'tf': tf})
    else:
        gru_model = create_model('gru', (X_train.shape[1], 1))
        train_and_save_model(gru_model, X_train, y_train, gru_model_path)
    
    # Generate predictions
    if with_confidence:
        predictions = {
            'short_term': {
                'lstm': predict_with_confidence(lstm_model, scaled_data, 7),
                'rnn': predict_with_confidence(rnn_model, scaled_data, 7),
                'gru': predict_with_confidence(gru_model, scaled_data, 7),
            },
            'long_term': {
                'lstm': predict_with_confidence(lstm_model, scaled_data, 30),
                'rnn': predict_with_confidence(rnn_model, scaled_data, 30),
                'gru': predict_with_confidence(gru_model, scaled_data, 30)
            }
        }
    else:
        predictions = {
            'short_term': {
                'lstm': predict_future(lstm_model, scaled_data, 7),
                'rnn': predict_future(rnn_model, scaled_data, 7),
                'gru': predict_future(gru_model, scaled_data, 7),
            },
            'long_term': {
                'lstm': predict_future(lstm_model, scaled_data, 30),
                'rnn': predict_future(rnn_model, scaled_data, 30),
                'gru': predict_future(gru_model, scaled_data, 30)
            }
        }
    
    # Inverse transform predictions to get actual stock prices
    for term in predictions:
        for model in predictions[term]:
            pred_values = predictions[term][model]
            if with_confidence:
                means = pred_values[:, 0]
                stds = pred_values[:, 1]
                actual_means = scaler.inverse_transform(means.reshape(-1, 1)).flatten()
                actual_stds = stds  # Standard deviation might need scaling if required
                predictions[term][model] = {'mean': actual_means, 'std': actual_stds}
            else:
                actual_means = scaler.inverse_transform(pred_values.reshape(-1, 1)).flatten()
                predictions[term][model] = {'mean': actual_means}
    
    print(predictions)
    
    return predictions

if __name__ == '__main__':
    predictions = get_stock_predictions('NVDA', with_confidence=False)
    print(predictions)