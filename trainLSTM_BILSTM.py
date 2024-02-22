#train lstm and bi-lstm model and plot a graph

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Bidirectional
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import tensorflow as tf

# Load data from CSV file
data_df = pd.read_csv('dataavg.csv')

# Function to create time series data for LSTM
def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X[i:(i + time_steps)]
        Xs.append(v)
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

# Function to create time series data for Bi-LSTM
def create_dataset_bilstm(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X[i:(i + time_steps)]
        Xs.append(v)
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

# Train LSTM model
def train_model(X_train, y_train):
    model = Sequential([
        LSTM(units=64, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dense(units=1)
    ])
    model.compile(optimizer=tf.compat.v1.train.AdamOptimizer(), loss='mean_squared_error')
    history = model.fit(X_train, y_train, epochs=70, batch_size=16, validation_split=0.1, verbose=0)
    return model, history

# Train Bi-LSTM model
def train_model_bilstm(X_train, y_train):
    model = Sequential([
        Bidirectional(LSTM(units=64), input_shape=(X_train.shape[1], X_train.shape[2])),
        Dense(units=1)
    ])
    model.compile(optimizer=tf.compat.v1.train.AdamOptimizer(), loss='mean_squared_error')
    history = model.fit(X_train, y_train, epochs=70, batch_size=16, validation_split=0.1, verbose=0)
    return model, history

# Calculate Mean Absolute Percentage Error (MAPE)
def calculate_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Scale data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data_df['Avg_Power'].values.reshape(-1, 1))

# Create time series data for LSTM
X_train, y_train = create_dataset(scaled_data, scaled_data, time_steps=10)

# Train LSTM model
model, history = train_model(X_train, y_train)

# Create time series data for Bi-LSTM
X_train_bilstm, y_train_bilstm = create_dataset_bilstm(scaled_data, scaled_data, time_steps=10)

# Train Bi-LSTM model
model_bilstm, history_bilstm = train_model_bilstm(X_train_bilstm, y_train_bilstm)

# Predict using LSTM model
y_pred_lstm = model.predict(X_train)
y_pred_inv_lstm = scaler.inverse_transform(y_pred_lstm).reshape(-1)  # Reshape to 1D array

# Predict using Bi-LSTM model
y_pred_bilstm = model_bilstm.predict(X_train_bilstm)
y_pred_inv_bilstm = scaler.inverse_transform(y_pred_bilstm).reshape(-1)  # Reshape to 1D array


