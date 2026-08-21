import numpy as np
import pandas as pd

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # Avoid division by zero
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def root_mean_squared_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def evaluate_forecast(y_true, y_pred):
    """Returns standard forecasting metrics."""
    return {
        'MAPE': mean_absolute_percentage_error(y_true, y_pred),
        'RMSE': root_mean_squared_error(y_true, y_pred)
    }

def get_train_test_split(df, date_col='date', test_days=30):
    """Creates a time-respecting train/test split."""
    max_date = pd.to_datetime(df[date_col]).max()
    split_date = max_date - pd.Timedelta(days=test_days)
    
    train = df[pd.to_datetime(df[date_col]) <= split_date].copy()
    test = df[pd.to_datetime(df[date_col]) > split_date].copy()
    
    return train, test
