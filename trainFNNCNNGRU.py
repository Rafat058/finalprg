#train FNN,CNN and GRU model and plot a graph

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Bidirectional, SimpleRNN, GRU, Flatten, Conv1D, MaxPooling1D
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import tensorflow as tf

# Load data from CSV file
data_df = pd.read_csv('dataavg.csv')

# Function to create time series data
def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X[i:(i + time_steps)]
        Xs.append(v)
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

# Train model
def train_model(model, X_train, y_train):
    model.compile(optimizer=tf.compat.v1.train.AdamOptimizer(), loss='mean_squared_error')
    history = model.fit(X_train, y_train, epochs=70, batch_size=16, validation_split=0.1, verbose=0)
    return model, history

# Calculate Mean Absolute Percentage Error (MAPE)
def calculate_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Scale data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data_df['Avg_Power'].values.reshape(-1, 1))

# Create time series data
X_train, y_train = create_dataset(scaled_data, scaled_data, time_steps=10)

# FNN model
model_fnn = Sequential([
    Flatten(input_shape=(X_train.shape[1], X_train.shape[2])),
    Dense(units=64, activation='relu'),
    Dense(units=1)
])

model_fnn, history_fnn = train_model(model_fnn, X_train, y_train)

# CNN model
model_cnn = Sequential([
    Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(units=1)
])

model_cnn, history_cnn = train_model(model_cnn, X_train, y_train)

# GRU model
model_gru = Sequential([
    GRU(units=64, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dense(units=1)
])

model_gru, history_gru = train_model(model_gru, X_train, y_train)

# Predict using FNN model
y_pred_fnn = model_fnn.predict(X_train)
y_pred_inv_fnn = scaler.inverse_transform(y_pred_fnn).reshape(-1)  # Reshape to 1D array

# Predict using CNN model
y_pred_cnn = model_cnn.predict(X_train)
y_pred_inv_cnn = scaler.inverse_transform(y_pred_cnn).reshape(-1)  # Reshape to 1D array

# Predict using GRU model
y_pred_gru = model_gru.predict(X_train)
y_pred_inv_gru = scaler.inverse_transform(y_pred_gru).reshape(-1)  # Reshape to 1D array

# Calculate MAPE for FNN, CNN, and GRU models
mape_fnn = calculate_mape(data_df['Avg_Power'].iloc[10:].values, y_pred_inv_fnn)
mape_cnn = calculate_mape(data_df['Avg_Power'].iloc[10:].values, y_pred_inv_cnn)
mape_gru = calculate_mape(data_df['Avg_Power'].iloc[10:].values, y_pred_inv_gru)

# Plot real vs predicted power for all models
plt.figure(figsize=(30, 7))
plt.plot(data_df['Date_Hour'].iloc[10:], data_df['Avg_Power'].iloc[10:], label='Real Power', color='blue')
plt.plot(data_df['Date_Hour'].iloc[10:], y_pred_inv_fnn, label=f'Predicted Power (FNN) - MAPE: {mape_fnn:.2f}%', color='orange')
plt.plot(data_df['Date_Hour'].iloc[10:], y_pred_inv_cnn, label=f'Predicted Power (CNN) - MAPE: {mape_cnn:.2f}%', color='purple')
plt.plot(data_df['Date_Hour'].iloc[10:], y_pred_inv_gru, label=f'Predicted Power (GRU) - MAPE: {mape_gru:.2f}%', color='brown')
plt.xlabel('Date_Hour')
plt.ylabel('Power')
plt.title('Real vs Predicted Power vs Time')
plt.legend()

# Rotate x-axis tick labels for better readability
plt.xticks(rotation=45)

plt.show()
