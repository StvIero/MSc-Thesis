# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 21:07:12 2023

@author: ieron
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Download data from yfinance until June X, 2023
end_date = '2023-06-23'
start_date = (datetime.strptime(end_date, '%Y-%m-%d') - pd.DateOffset(years=5)).strftime('%Y-%m-%d')
ticker = '^NDX'
data = yf.download(ticker, start=start_date, end=end_date)

data.plot.line(y='Close', use_index=True)

from pmdarima.arima.utils import ndiffs
ndiffs(data.Close, test='adf')

diff = data.Close.diff().dropna()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 4))
ax1.plot(diff)
ax1.set_title('Difference once')
ax2.set_ylim(0, 1)
plot_pacf(diff, ax=ax2)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 4))
ax1.plot(diff)
ax1.set_title('Difference once')
ax2.set_ylim(0, 0.5)
plot_acf(diff, ax=ax2)

row = int(len(data) * 0.8)
train = list(data[0:row]['Close'])
test = list(data[row:]['Close'])
print(len(train))
print(len(test))

# Initialize empty lists to store predictions and true values
model_predictions = []
true_test_values = []

history = train  # Use the initial training set for fitting the model

# Perform rolling forecast
for t in range(len(test)):
    model = ARIMA(history, order=(1, 1, 1))  # Fit the ARIMA model
    model_fit = model.fit()

    # Forecast one step ahead
    yhat = model_fit.forecast(steps=1)[0]

    # Store the prediction and true value
    model_predictions.append(yhat)
    true_test_values.append(test[t])

    # Update the history with the true value
    history.append(test[t])

MAE_error = mean_absolute_error(true_test_values, model_predictions)
MSE_error = mean_squared_error(true_test_values, model_predictions)
r2 = r2_score(true_test_values, model_predictions)

print('Mean Absolute Error (MAE):', MAE_error)
print('Mean Squared Error (MSE):', MSE_error)
print('R-squared (R^2):', r2)

# Plot diagnostics
model_fit.plot_diagnostics(figsize=(12, 6))
plt.show()

# Plot the forecast for the entire series
plt.figure(figsize=(12, 6))
data_range = data.index[row:]
plt.plot(data_range, model_predictions, color='orange', label='^NDX Predicted Price')
plt.plot(data_range, true_test_values, color='blue', label='^NDX Actual Price')
plt.title('ARIMA: ^NDX Prices Prediction (Rolling Forecast)')
plt.xlabel('Date')
plt.ylabel('Closing Prices')
plt.legend()
plt.show()

MAE_error = mean_absolute_error(true_test_values, model_predictions)
MSE_error = mean_squared_error(true_test_values, model_predictions)
r2 = r2_score(true_test_values, model_predictions)

print('Mean Absolute Error (MAE):', MAE_error)
print('Mean Squared Error (MSE):', MSE_error)
print('R-squared (R^2):', r2)

# Plot diagnostics
model_fit.plot_diagnostics(figsize=(12, 6))
plt.show()

# Plot the forecast for the entire series
plt.figure(figsize=(12, 6))
data_range = data.index[row:]
plt.plot(data_range, model_predictions, color='orange', label='^NDX Predicted Price')
plt.plot(data_range, true_test_values, color='blue', label='^NDX Actual Price')
plt.title('ARIMA: ^NDX Prices Prediction (Rolling Forecast)')
plt.xlabel('Date')
plt.ylabel('Closing Prices')
plt.legend()
plt.show()
















