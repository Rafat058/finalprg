import csv
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Bidirectional, Flatten, Conv1D, MaxPooling1D, GRU
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

# Scale data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data_df['Avg_Power'].values.reshape(-1, 1))

# Create time series data
X_train, y_train = create_dataset(scaled_data, scaled_data, time_steps=10)

# LSTM model
model_lstm = Sequential([
    LSTM(units=64, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dense(units=1)
])

model_lstm, history_lstm = train_model(model_lstm, X_train, y_train)

# Bi-LSTM model
model_bilstm = Sequential([
    Bidirectional(LSTM(units=64), input_shape=(X_train.shape[1], X_train.shape[2])),
    Dense(units=1)
])

model_bilstm, history_bilstm = train_model(model_bilstm, X_train, y_train)

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

# Select the last 10 data points from the dataset
last_data_points = scaled_data[-10:]

# Reshape the data to match the input shape of the models
last_data_points = last_data_points.reshape((1, 10, 1))

# Predict the power output for the next hour using each model
next_hour_pred_lstm = model_lstm.predict(last_data_points)[0][0]
next_hour_pred_bilstm = model_bilstm.predict(last_data_points)[0][0]
next_hour_pred_fnn = model_fnn.predict(last_data_points)[0][0]
next_hour_pred_cnn = model_cnn.predict(last_data_points)[0][0]
next_hour_pred_gru = model_gru.predict(last_data_points)[0][0]

# Invert the scaling to get the actual power output values
next_hour_pred_lstm = scaler.inverse_transform(np.array([[next_hour_pred_lstm]]))[0][0]
next_hour_pred_bilstm = scaler.inverse_transform(np.array([[next_hour_pred_bilstm]]))[0][0]
next_hour_pred_fnn = scaler.inverse_transform(np.array([[next_hour_pred_fnn]]))[0][0]
next_hour_pred_cnn = scaler.inverse_transform(np.array([[next_hour_pred_cnn]]))[0][0]
next_hour_pred_gru = scaler.inverse_transform(np.array([[next_hour_pred_gru]]))[0][0]

# Save predicted values to a CSV file
with open('predict.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['LSTM', 'Bi-LSTM', 'FNN', 'CNN', 'GRU'])
    writer.writerow([next_hour_pred_lstm, next_hour_pred_bilstm, next_hour_pred_fnn, next_hour_pred_cnn, next_hour_pred_gru])

# Print the predicted values
print("Predicted values saved to predict.csv")
