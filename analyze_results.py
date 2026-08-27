import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import os

def calculate_metrics(df, model_name):
    if df.empty:
        return {'Model': model_name, 'RMSE': np.nan, 'MAE': np.nan}
    df_clean = df.dropna(subset=['actual_sales', 'predicted_sales'])
    if df_clean.empty:
        return {'Model': model_name, 'RMSE': np.nan, 'MAE': np.nan}
    
    rmse = np.sqrt(mean_squared_error(df_clean['actual_sales'], df_clean['predicted_sales']))
    mae = mean_absolute_error(df_clean['actual_sales'], df_clean['predicted_sales'])
    return {'Model': model_name, 'RMSE': rmse, 'MAE': mae}

def main():
    print("Loading Results...")
    
    try:
        lgb_results = pd.read_parquet('lightgbm_results.parquet')
    except:
        lgb_results = pd.DataFrame()
        
    try:
        prophet_results = pd.read_parquet('prophet_results.parquet')
    except:
        prophet_results = pd.DataFrame()

    try:
        lstm_results = pd.read_parquet('lstm_results.parquet')
    except:
        lstm_results = pd.DataFrame()

    metrics = []
    metrics.append(calculate_metrics(lgb_results, "LightGBM (Winning M5 Architecture)"))
    metrics.append(calculate_metrics(prophet_results, "Prophet (Classical Baseline)"))
    metrics.append(calculate_metrics(lstm_results, "LSTM (Deep Learning Baseline)"))
    
    comparison = pd.DataFrame(metrics)
    print("\n--- MODEL COMPARISON ---")
    print(comparison.to_string(index=False))
    print("------------------------\n")
    
    os.makedirs('data', exist_ok=True)
    comparison.to_csv('data/final_model_comparison.csv', index=False)
    
    if not lgb_results.empty and not prophet_results.empty and not lstm_results.empty:
        # Find a common item and store
        common_items = set(lgb_results['item_id']).intersection(set(prophet_results['item_id'])).intersection(set(lstm_results['item_id']))
        if common_items:
            sample_sku = list(common_items)[0]
            # Pick the first store this SKU appears in
            sample_store = lgb_results[lgb_results['item_id'] == sample_sku]['store_id'].iloc[0]
            
            print(f"Generating plot for SKU: {sample_sku} at Store: {sample_store}")
            
            sku_lgb = lgb_results[(lgb_results['item_id'] == sample_sku) & (lgb_results['store_id'] == sample_store)].sort_values('date')
            sku_prophet = prophet_results[(prophet_results['item_id'] == sample_sku) & (prophet_results['store_id'] == sample_store)].sort_values('date')
            sku_lstm = lstm_results[(lstm_results['item_id'] == sample_sku) & (lstm_results['store_id'] == sample_store)].sort_values('date')
            
            plt.figure(figsize=(12, 6))
            plt.plot(sku_lgb['date'], sku_lgb['actual_sales'], label='Actual Sales', color='black', linewidth=2, marker='o')
            plt.plot(sku_lgb['date'], sku_lgb['predicted_sales'], label='LightGBM', color='blue', linestyle='--')
            plt.plot(sku_prophet['date'], sku_prophet['predicted_sales'], label='Prophet', color='orange', linestyle='-.')
            plt.plot(sku_lstm['date'], sku_lstm['predicted_sales'], label='LSTM', color='green', linestyle=':')
            
            plt.title(f'Forecast Comparison for {sample_sku} at {sample_store}')
            plt.xlabel('Date')
            plt.ylabel('Sales')
            plt.legend()
            plt.tight_layout()
            plt.savefig('data/final_forecast_comparison.png')
            print("Plot saved to data/final_forecast_comparison.png")

if __name__ == "__main__":
    main()
