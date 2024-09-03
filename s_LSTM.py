# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 11:39:09 2023

@author: ieron
"""

#s_LSTM
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

# Download data from yfinance for the last 10 years
end_date = datetime.now().strftime('%Y-%m-%d')
start_date = (datetime.now() - pd.DateOffset(years=5)).strftime('%Y-%m-%d')
ticker = '^NDX'
data = yf.download(ticker, start=start_date, end=end_date)

# Calculate additional features
data['Medium'] = (data['High'] + data['Low']) / 2
data['Range'] = data['High'] - data['Low']

# Drop unnecessary features
data = data[['Open', 'High', 'Low', 'Medium', 'Range', 'Close']]

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Create the training data
window_size = 5

train_data = []
train_labels = []

for i in range(window_size, len(scaled_data)):
    train_data.append(scaled_data[i - window_size: i])
    train_labels.append(scaled_data[i][-1])

train_data = np.array(train_data)
train_labels = np.array(train_labels)

# Split the data into training, validation, and testing sets
x_train_val, x_test, y_train_val, y_test = train_test_split(train_data, train_labels, test_size=0.2, shuffle=False)
x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, test_size=0.25, shuffle=False)

# Reshape the input features to match the LSTM input shape
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], x_train.shape[2]))
x_val = np.reshape(x_val, (x_val.shape[0], x_val.shape[1], x_val.shape[2]))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], x_test.shape[2]))


# Define the LSTM model
def create_model(units=50, activation='relu'):
    model = Sequential()
    model.add(LSTM(units=units, input_shape=(x_train.shape[1], x_train.shape[2]), activation=activation))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Define parameter grid for manual search (including window size)
param_grid = {
    'units': [50, 100, 150],
    'epochs': [30, 50, 100],
    'batch_size': [32, 64, 128],
    'activation': ['relu', 'tanh', 'sigmoid'],
    'window_size': [5, 21, 63]  # Modify the window sizes
}

best_mse = float('inf')
best_model = None
best_params = {}

# Calculate min_close and max_close
min_close = data['Close'].min()
max_close = data['Close'].max()

# Iterate over parameter grid
for units in param_grid['units']:
    for epochs in param_grid['epochs']:
        for batch_size in param_grid['batch_size']:
            for activation in param_grid['activation']:
                for window_size in param_grid['window_size']:  # Add window size loop
                    # Create and train the model
                    model = create_model(units=units, activation=activation)
                    history = model.fit(x_train[:, -window_size:, :], y_train, validation_data=(x_val[:, -window_size:, :], y_val),
                                        epochs=epochs, batch_size=batch_size, verbose=0, callbacks=[EarlyStopping(patience=5)])

                    # Evaluate the model
                    val_loss = history.history['val_loss'][-1]  # Use 'val_loss' instead of 'loss'
                    mse = val_loss * (max_close - min_close) ** 2
                    mae = mean_absolute_error(y_val, model.predict(x_val))
                    r2 = r2_score(y_val, model.predict(x_val))
                    
                    # Update the best model and parameters if the MSE improves
                    if mse < best_mse:
                        best_mse = mse
                        best_model = model
                        best_params = {'units': units, 'epochs': epochs, 'batch_size': batch_size,
                                       'activation': activation, 'window_size': window_size}

# Print the best hyperparameters
print("Best Hyperparameters:")
for param, value in best_params.items():
    print(f"{param}: {value}")
    
print("Metrics:")
print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"R2: {r2}")

'''
total number of combinations: 
3 (units) * 3 (epochs) * 3 (batch_size) * 3 (activation) * 3 (window_size) = 243
    
Best Hyperparameters:
units: 50
epochs: 50
batch_size: 32
activation: sigmoid
window_size: 21

Metrics:
MAE: 0.01963755846297224
MSE: 71849.27345265627
R2: 0.9463631056173504
'''

###############################################################################
'''
# Initialize an empty list to store the predictions for each window size
all_predictions = []

# Iterate over different window sizes
for window_size in param_grid['window_size']:
    # Select the last window_size number of data points for forecasting
    test_data = x_test[:, -window_size:, :]

    # Generate predictions using the best model
    predictions = best_model.predict(test_data)

    # Rescale the predictions and actual values back to the original scale
    predictions = predictions * (max_close - min_close) + min_close
    actual_values = y_test[-window_size:] * (max_close - min_close) + min_close

    # Append the predictions to the list
    all_predictions.append(predictions)

    # Plot actual vs forecasted values
    plt.figure(figsize=(12, 6))
    plt.plot(range(window_size), actual_values, label='Actual')
    plt.plot(range(window_size), predictions[-window_size:], label='Forecasted')
    plt.xlabel('Time')
    plt.ylabel('Close Price')
    plt.title(f"Actual vs Forecasted Close Values (Window Size {window_size})")
    plt.legend()
    plt.show()

    # Train the model with the best parameters and plot learning curves
    model = create_model(best_params['units'], best_params['activation'])
    history = model.fit(x_train, y_train, epochs=best_params['epochs'], batch_size=best_params['batch_size'], validation_data=(x_val, y_val), verbose=0)

    # Plot learning curves
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, best_params['epochs'] + 1), history.history['loss'], label='Training Loss')
    plt.plot(range(1, best_params['epochs'] + 1), history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Learning Curves (Window Size {window_size})')
    plt.legend()
    plt.show()
'''
###############################################################################
# Initialize an empty list to store the predictions for each window size
all_predictions = []

# Create a dictionary to store the forecasted values for each window size
forecasted_values = {}

# Define the forecasting periods for each window size
forecasting_periods = {
    5: 5,
    21: 21,
    63: 63
}

# Iterate over different window sizes
for window_size in param_grid['window_size']:
    # Select the last window_size number of data points for forecasting
    test_data = x_test[:, -window_size:, :]

    # Generate predictions using the best model
    predictions = best_model.predict(test_data)

    # Reshape the predictions and actual values
    predictions = predictions.flatten()
    actual_values = y_test[-predictions.shape[0]:]

    # Trim the predictions and actual values to the desired forecasting period
    forecasting_period = forecasting_periods[window_size]
    predictions = predictions[:forecasting_period]
    actual_values = actual_values[:forecasting_period]

    # Calculate metrics
    mae = mean_absolute_error(actual_values, predictions)
    mse = mean_squared_error(actual_values, predictions)
    r2 = r2_score(actual_values, predictions)
    
    # Store the forecasted values for the current window size
    forecasted_values[window_size] = (actual_values, predictions)

    # Plot actual vs forecasted values
    plt.figure(figsize=(12, 6))
    plt.plot(range(forecasting_period), actual_values, label='Actual')
    plt.plot(range(forecasting_period), predictions, label='Forecasted')
    plt.xlabel('Time')
    plt.ylabel('Close Price')
    plt.title(f"Actual vs Forecasted Close Values (Window Size {window_size})")
    plt.legend()
    plt.show()

    # Print the metrics for the current window size
    print(f"Metrics for Window Size {window_size}:")
    print(f"MAE: {mae}")
    print(f"MSE: {mse}")
    print(f"R2: {r2}")

    # Train the model with the best parameters and plot learning curves
    model = create_model(best_params['units'], best_params['activation'])
    history = model.fit(x_train, y_train, epochs=best_params['epochs'], batch_size=best_params['batch_size'], validation_data=(x_val, y_val), verbose=0)

    # Plot learning curves
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, best_params['epochs'] + 1), history.history['loss'], label='Training Loss')
    plt.plot(range(1, best_params['epochs'] + 1), history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Learning Curves (Window Size {window_size})')
    plt.legend()
    plt.show()

'''
Metrics for Window Size 5:
MAE: 0.03064986421289555
MSE: 0.0016451114653519504
R2: -7.083479664526424

Metrics for Window Size 21:
MAE: 0.02345809234624381
MSE: 0.0008290173611028207
R2: -0.23589570194137566

Metrics for Window Size 63:
MAE: 0.021348584256537826
MSE: 0.0006593215512316105
R2: 0.8372927419041091
'''



