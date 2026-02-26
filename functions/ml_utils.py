import pandas as pd
import itertools
import warnings
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import matplotlib.pyplot as plt

def group_orders_by_time(orders_clean, aggregation='month'):
    """
    Groups orders by time period (day or month).

    Args:
        orders_clean (pd.DataFrame): DataFrame containing the orders.
        aggregation (str): The aggregation level ('day' or 'month').

    Returns:
        pd.Series: A time series of aggregated orders.
    """
    if aggregation == 'month':
        grouped_orders = orders_clean.groupby(orders_clean['order_month']).size()
        grouped_orders.index = grouped_orders.index.to_timestamp() 
    elif aggregation == 'day':
        grouped_orders = orders_clean.groupby(orders_clean['order_date']).size()
    else:
        raise ValueError("Aggregation must be 'day' or 'month'.")

    grouped_orders = grouped_orders.sort_index()

    return grouped_orders

def adfuller_test(time_series):
    """
    Performs Augmented Dickey-Fuller (ADF) test on the given time series.

    Args:
        time_series (pd.Series): The time series data.

    Returns:
        tuple: ADF statistic and p-value.
    """
    result = adfuller(time_series)
    print(f"ADF Statistic: {result[0]}")
    print(f"p-value: {result[1]}")
    return result[0], result[1]

def difference_data(time_series):
    """
    Differencing the data to achieve stationarity.

    Args:
        time_series (pd.Series): The time series data.

    Returns:
        pd.Series: The differenced time series.
    """
    time_series_diff = time_series - time_series.shift(1)
    time_series_diff.dropna(inplace=True)
    return time_series_diff

def grid_search_arima(time_series):
    """
    Performs grid search to find the best ARIMA model parameters.

    Args:
        time_series (pd.Series): The differenced time series data.

    Returns:
        tuple: Best ARIMA model, AIC value, and model order.
    """
    p_values = [0, 1, 2, 3]
    d_values = [1]
    q_values = [0, 1, 2, 3]

    best_aic = float('inf')
    best_order = None
    best_model = None

    for p, d, q in itertools.product(p_values, d_values, q_values):
        try:
            model = ARIMA(time_series, order=(p, d, q))
            model_fit = model.fit()
            aic = model_fit.aic
            if aic < best_aic:
                best_aic = aic
                best_order = (p, d, q)
                best_model = model_fit
        except:
            continue

    return best_model, best_order, best_aic

def train_arima_model(time_series, order, train_test_break):
    """
    Fits an ARIMA model to the given time series.

    Args:
        time_series (pd.Series): The differenced time series data.
        order (tuple): The ARIMA model order (p, d, q).

    Returns:
        ARIMAResultsWrapper: Fitted ARIMA model.
    """

    train_data = time_series.loc[:train_test_break]
    test_data = time_series.loc[train_test_break:]

    model = ARIMA(train_data, order=order)
    model_fit = model.fit()

    forecast_values = model_fit.forecast(steps=len(test_data))

    rmse = root_mean_squared_error(test_data, forecast_values)
    mae = mean_absolute_error(test_data, forecast_values)

    return model_fit

def forecast_next_periods(model_fit, forecast_steps, last_original_value):
    """
    Forecasts the next periods and adjusts for reverse differencing.

    Args:
        model_fit (ARIMAResultsWrapper): The fitted ARIMA model.
        forecast_steps (int): The number of periods to forecast.
        last_original_value (float): The last value of the original series before differencing.

    Returns:
        tuple: Forecasted values and their confidence intervals in the original scale.
    """
    forecast_values = model_fit.forecast(steps=forecast_steps)

    # Get the confidence intervals
    conf_int = model_fit.get_forecast(steps=forecast_steps).conf_int(alpha=0.05)

    # Reverse differencing
    forecast_values_original_scale = forecast_values + last_original_value
    conf_int_original_scale = conf_int + last_original_value

    return forecast_values_original_scale, conf_int_original_scale
