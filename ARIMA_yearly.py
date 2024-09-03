# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 08:26:38 2023

@author: ieron
"""

import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Download AAPL data
ticker = 'AAPL'
data = yf.download(ticker, period='max')

# Extract 'Adj Close' values and convert to DataFrame
data = data['Close'].to_frame(name='Close')

# Select data from 2000 onwards
data = data[data.index.year >= 2010]

# Create a list to store results for each year
results_list = []

# Split data into yearly bins and conduct analysis for each year
for year, group in data.groupby(data.index.year):
    print(f'Year: {year}')
    print(group)

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
    check_stationarity(group['Close'])

    # Plot 'Close' values before differencing
    plt.figure(figsize=(12, 6))
    plt.plot(group['Close'])
    plt.title(f'{ticker} Adjusted Close Prices (Before Differencing) - Year {year}')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.grid(True)
    plt.show()

    # Difference the 'Close' values to achieve stationarity
    diff = group['Close'].diff().dropna()

    # Check for stationarity after differencing
    check_stationarity(diff)

    # Plot differenced 'Close' values
    plt.figure(figsize=(12, 6))
    plt.plot(diff)
    plt.title(f'Differenced {ticker} Adjusted Close Prices (After Differencing) - Year {year}')
    plt.xlabel('Date')
    plt.ylabel('Price Difference')
    plt.grid(True)
    plt.show()

    # Fit ARIMA model with specific parameters p = 3, d = 0, q = 3
    model = ARIMA(group['Close'], order=(3, 0, 3))
    results = model.fit()

    forecast_start_date = group.index[0]
    forecast_end_date = group.index[-1]
    forecast = results.predict(start=forecast_start_date, end=forecast_end_date)

    # Extract actual 'Close' values from the forecast start date to the last observation
    actual = group.loc[forecast_start_date:forecast_end_date, 'Close']

    # Calculate residuals
    residuals = actual - forecast

    # Perform Ljung-Box test for autocorrelation in residuals
    from statsmodels.stats.diagnostic import acorr_ljungbox

    lb_test = acorr_ljungbox(residuals, lags=20)

    if any(p <= 0.05 for p in lb_test['lb_pvalue']):
        print('The residuals are likely autocorrelated, indicating the presence of patterns.')
    else:
        print('The residuals are likely not autocorrelated, indicating white noise behavior.')

    print(lb_test)
    '''
    The residuals are likely not autocorrelated, indicating white noise behavior.
    '''

    # Plot the residuals
    plt.figure(figsize=(12, 6))
    plt.plot(residuals)
    plt.title(f'Residuals Plot - Year {year}')
    plt.xlabel('Index')
    plt.ylabel('Residuals')
    plt.grid(True)
    plt.show()

    # Evaluate performance of forecasted values
    mse = mean_squared_error(actual, forecast)
    rmse = mse ** 0.5
    mae = mean_absolute_error(actual, forecast)
    r2 = r2_score(actual, forecast)

    print(f'\nRoot Mean Squared Error (RMSE) - Year {year}: {rmse}')
    print(f'Mean Absolute Error (MAE) - Year {year}: {mae}')
    print(f'R-squared - Year {year}: {r2}')
    # print(results.summary())

    # Store the results for the year in a DataFrame or dictionary
    results_year = pd.DataFrame({
        'Year': year,
        'RMSE': rmse,
        'MAE': mae,
        'R-squared': r2
    }, index=[0])

    # Append the results to the results_list
    results_list.append(results_year)

# Create a DataFrame from the results_list
results_df = pd.concat(results_list, ignore_index=True)

# Save results to separate files for each year
for year, result in zip(results_df['Year'], results_list):
    result.to_csv(f'results_{ticker}_{year}.csv', index=False)

