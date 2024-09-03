# -*- coding: utf-8 -*-
"""
Created on Tue May 23 14:07:09 2023

@author: ieron
"""

#ARIMA
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from itertools import product
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Download MSFT data
ticker = '^NDX'
data = yf.download(ticker, period='max')

# Extract 'Adj Close' values and convert to DataFrame
data = data['Close'].to_frame(name='Close')
print(data)

# Check for stationarity before differencing
def check_stationarity(data):
    result = adfuller(data)
    print('Augmented Dickey-Fuller Test:')
    print(f'Test Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    print(f'Critical Values:')
    for key, value in result[4].items():
        print(f'  {key}: {value}')
    if result[1] <= 0.05:
        print('The series is likely stationary.')
    else:
        print('The series is likely non-stationary.')

# Perform stationarity check on 'Close'
check_stationarity(data['Close'])

# Plot 'Adj Close' values before differencing
plt.figure(figsize=(12, 6))
plt.plot(data['Close'])
plt.title(f'{ticker} Adjusted Close Prices (Before Differencing)')
plt.xlabel('Date')
plt.ylabel('Price')
plt.grid(True)
plt.show()

# Difference the 'Close' values to achieve stationarity
diff = data['Close'].diff().dropna()

# Check for stationarity after differencing
check_stationarity(diff)

# Plot differenced 'Close' values
plt.figure(figsize=(12, 6))
plt.plot(diff)
plt.title(f'Differenced {ticker} Adjusted Close Prices (After Differencing)')
plt.xlabel('Date')
plt.ylabel('Price Difference')
plt.grid(True)
plt.show()

# Perform grid search for optimal ARIMA parameters
p_range = range(0, 5)  # AR parameter range
d_range = range(0, 2)  # Differencing parameter range
q_range = range(0, 5)  # MA parameter range

# Initialize variables for optimal parameters and AIC score
best_aic = float('inf')
optimal_params = None

# Grid search for optimal parameters
for p, d, q in product(p_range, d_range, q_range):
    try:
        model = ARIMA(data['Close'], order=(p, d, q))
        results = model.fit()
        aic = results.aic
        if aic < best_aic:
            best_aic = aic
            optimal_params = (p, d, q)
        print(f'ARIMA({p},{d},{q}) - AIC: {aic}')
    except:
        continue

print(f'\nOptimal ARIMA parameters: {optimal_params}')

'''
Number of possible combinations: 5 * 2 * 5 = 50
 

'''

# Fit ARIMA model with optimal parameters
model = ARIMA(data['Close'], order=optimal_params)
results = model.fit()

forecast_start_date = '2022-01-03'
forecast_end_date = data.index[-1]
forecast = results.predict(start=forecast_start_date, end=forecast_end_date)

# Extract actual 'Close' values from 2022 onward until the last observation
actual = data.loc[forecast_start_date:forecast_end_date, 'Close']

##############################################################################
from statsmodels.stats.diagnostic import acorr_ljungbox

residuals = actual - forecast

lb_test = acorr_ljungbox(residuals, lags=20)

if any(p <= 0.05 for p in lb_test['lb_pvalue']):
    print('The residuals are likely autocorrelated, indicating the presence of patterns.')
else:
    print('The residuals are likely not autocorrelated, indicating white noise behavior.')

'''
 If at least one p-value is below 0.05, 
 it suggests the presence of autocorrelation in the residuals, 
 indicating patterns beyond white noise behavior. 
 If all p-values are above 0.05, it suggests the residuals 
 exhibit white noise characteristics.
'''
print(lb_test)
'''
The residuals are likely not autocorrelated, indicating white noise behavior.
'''
# Plot the residuals
plt.figure(figsize=(12, 6))
plt.plot(residuals)
plt.title('Residuals Plot')
plt.xlabel('Index')
plt.ylabel('Residuals')
plt.grid(True)
plt.show()

'''
If the Ljung-Box test indicates that the residuals are likely not 
autocorrelated and exhibit white noise behavior, it suggests that 
our model has captured the underlying patterns in the data and there 
are no significant residual correlations remaining. 
This is generally considered a desirable characteristic, 
indicating that the model is performing well in terms of capturing 
the information present in the data.
'''
###############################################################################

# Evaluate performance of forecasted values
mse = mean_squared_error(actual, forecast)
rmse = mse ** 0.5
mae = mean_absolute_error(actual, forecast)
r2 = r2_score(actual, forecast)

print(f'\nRoot Mean Squared Error (RMSE): {rmse}')
print(f'Mean Absolute Error (MAE): {mae}')
print(f'R-squared: {r2}')
#print(results.summary())

'''

'''

# Plot actual vs forecasted 'Close' values from 2022 onward until the last observation
plt.figure(figsize=(12, 6))
plt.plot(actual, label='Actual')
plt.plot(forecast, label='Forecast')
plt.title(f"ARIMA: {ticker}'s Actual vs Forecasted Close Values (2020)")
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()






















