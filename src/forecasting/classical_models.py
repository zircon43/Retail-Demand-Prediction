import pandas as pd
from prophet import Prophet
import os
import multiprocessing

def train_predict_prophet_single(args):
    """
    Worker function to train a Prophet model for a specific SKU and Store.
    """
    df_subset, item_id, store_id, test_days = args
    # Prophet requires 'ds' (date) and 'y' (target) columns
    prophet_df = df_subset[['date', 'sales']].rename(columns={'date': 'ds', 'sales': 'y'})
    prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])
    
    # Split
    max_date = prophet_df['ds'].max()
    split_date = max_date - pd.Timedelta(days=test_days)
    
    train = prophet_df[prophet_df['ds'] <= split_date]
    test = prophet_df[prophet_df['ds'] > split_date]
    
    # Train
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    model.fit(train)
    
    # Predict
    forecast = model.predict(test[['ds']])
    
    # Format results
    results = test.copy()
    results['yhat'] = forecast['yhat'].values
    results['yhat'] = results['yhat'].clip(lower=0) # No negative sales
    results = results.rename(columns={'ds': 'date', 'y': 'actual_sales', 'yhat': 'predicted_sales'})
    results['item_id'] = item_id
    results['store_id'] = store_id
    
    return results

def run_classical_baseline(data_path='data/modeling_data.parquet', test_days=28, sample_frac=1.0):
    print("Loading data for Prophet baseline...")
    df = pd.read_parquet(data_path)
    
    if sample_frac < 1.0:
        # Sample items to speed up execution
        items_to_keep = pd.Series(df['item_id'].unique()).sample(frac=sample_frac, random_state=42).tolist()
        df = df[df['item_id'].isin(items_to_keep)]
        
    items = df['item_id'].unique()
    stores = df['store_id'].unique()
    
    tasks = []
    for store in stores:
        for item in items:
            df_subset = df[(df['item_id'] == item) & (df['store_id'] == store)].copy()
            if not df_subset.empty:
                tasks.append((df_subset, item, store, test_days))
            
    print(f"Training Prophet for {len(tasks)} item-store combinations...")
    
    # Using multiprocessing to speed up Prophet training
    num_cores = multiprocessing.cpu_count()
    with multiprocessing.Pool(processes=num_cores) as pool:
        all_results = pool.map(train_predict_prophet_single, tasks)
            
    final_df = pd.concat(all_results, ignore_index=True)
    final_df.to_parquet('data/prophet_forecast_results.parquet')
    print("Saved Prophet forecasts to data/prophet_forecast_results.parquet")
    return final_df

if __name__ == "__main__":
    # Use a small sample to test locally. In Kaggle set sample_frac=1.0
    run_classical_baseline(sample_frac=0.05)
