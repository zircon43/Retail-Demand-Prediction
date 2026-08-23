import pandas as pd
import numpy as np
import lightgbm as lgb
import os

def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object and str(col_type) != 'category' and 'datetime' not in str(col_type):
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    return df

def train_predict_lightgbm_per_store(df, store_id, test_days=28):
    """
    Trains a LightGBM model for a specific store utilizing Tweedie regression, 
    matching the winning M5 strategy.
    """
    print(f"Training LightGBM for store: {store_id}")
    store_df = df[df['store_id'] == store_id].copy()
    store_df['date'] = pd.to_datetime(store_df['date'])
    
    # Define features and target
    target = 'sales'
    features = [
        'sell_price', 'wday', 'month', 'year',
        'sales_lag_1', 'sales_lag_7', 'sales_lag_14', 'sales_lag_28',
        'rolling_avg_7', 'rolling_avg_28', 'rolling_std_7', 'rolling_std_28', 
        'month_to_date_sales'
    ]
    
    # Categorical encoding for item, and events
    store_df['item_id_cat'] = store_df['item_id'].astype('category').cat.codes
    store_df['event_name_1_cat'] = store_df['event_name_1'].astype('category').cat.codes
    features.extend(['item_id_cat', 'event_name_1_cat'])
    
    # Split using a time-respecting window
    max_date = store_df['date'].max()
    split_date = max_date - pd.Timedelta(days=test_days)
    
    train = store_df[store_df['date'] <= split_date].copy()
    test = store_df[store_df['date'] > split_date].copy()
    
    X_train, y_train = train[features], train[target]
    X_test, y_test = test[features], test[target]
    
    # Validation set for early stopping (e.g., last 28 days of train)
    val_split_date = split_date - pd.Timedelta(days=test_days)
    X_val = train[train['date'] > val_split_date][features]
    y_val = train[train['date'] > val_split_date][target]
    
    train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=['item_id_cat', 'event_name_1_cat'])
    val_data = lgb.Dataset(X_val, label=y_val, categorical_feature=['item_id_cat', 'event_name_1_cat'], reference=train_data)
    
    # Tweedie objective for zero-inflated demand (M5 Winning strategy)
    params = {
        'objective': 'tweedie',
        'tweedie_variance_power': 1.1,
        'metric': 'rmse',
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'max_depth': 8,
        'num_leaves': 64,
        'verbose': -1,
        'seed': 42
    }
    
    model = lgb.train(
        params, 
        train_data, 
        num_boost_round=1000, 
        valid_sets=[train_data, val_data],
        callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(100)]
    )
    
    # Predict
    test['predicted_sales'] = model.predict(X_test)
    test['predicted_sales'] = test['predicted_sales'].clip(lower=0)
    test = test.rename(columns={'sales': 'actual_sales'})
    
    return test[['date', 'item_id', 'store_id', 'actual_sales', 'predicted_sales']]

def run_ml_models(data_path='data/modeling_data.parquet', test_days=28, specific_stores=None):
    print("Loading data for LightGBM (M5 Winning Strategy)...")
    df = pd.read_parquet(data_path)
    df = reduce_mem_usage(df)
    
    stores = df['store_id'].unique()
    if specific_stores:
        stores = [s for s in stores if s in specific_stores]
        
    all_results = []
    for store in stores:
        res = train_predict_lightgbm_per_store(df, store, test_days)
        all_results.append(res)
        
    final_results = pd.concat(all_results, ignore_index=True)
    
    final_results.to_parquet('data/lightgbm_forecast_results.parquet')
    print("Saved LightGBM forecasts to data/lightgbm_forecast_results.parquet")
    return final_results

if __name__ == "__main__":
    # If running locally for a test, just run on 1 store. In Kaggle, specific_stores=None runs all.
    run_ml_models(specific_stores=['CA_1'])
