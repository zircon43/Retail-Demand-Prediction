import nbformat as nbf
import os

def generate_dashboard():
    nb = nbf.v4.new_notebook()

    markdown_1 = """# Time Series Inventory Forecasting Results
This notebook visualizes the results of the ML forecasting models (LightGBM Tweedie per M5 winning strategy vs LSTM vs Prophet baseline) and actual sales."""

    code_1 = """import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error

# Load data (if running locally, files might be small sample, in Kaggle they will be full)
print("Loading results...")
try:
    lgb_results = pd.read_parquet('../data/lightgbm_forecast_results.parquet')
except:
    lgb_results = pd.DataFrame()
    
try:
    prophet_results = pd.read_parquet('../data/prophet_forecast_results.parquet')
except:
    prophet_results = pd.DataFrame()

try:
    lstm_results = pd.read_parquet('../data/lstm_forecast_results.parquet')
except:
    lstm_results = pd.DataFrame()

print(f"Loaded LightGBM: {len(lgb_results)} rows")
print(f"Loaded Prophet: {len(prophet_results)} rows")
print(f"Loaded LSTM: {len(lstm_results)} rows")"""

    markdown_2 = """## Model Evaluation metrics (RMSE)
Calculate the Root Mean Squared Error for the models."""

    code_2 = """def calculate_rmse(df, model_name):
    if df.empty:
        return f"{model_name}: N/A"
    df_clean = df.dropna(subset=['actual_sales', 'predicted_sales'])
    if df_clean.empty:
        return f"{model_name}: N/A"
    rmse = np.sqrt(mean_squared_error(df_clean['actual_sales'], df_clean['predicted_sales']))
    return f"{model_name} RMSE: {rmse:.2f}"

print("--- OVERALL RMSE ---")
print(calculate_rmse(lgb_results, "LightGBM (M5 Top Strategy)"))
print(calculate_rmse(prophet_results, "Prophet"))
print(calculate_rmse(lstm_results, "LSTM"))
print("--------------------")"""

    markdown_3 = """## Forecast vs Actual for Multiple SKUs
Visualizing the comparison for sample SKUs."""

    code_3 = """# Get a few sample SKUs present in all dataframes
if not lgb_results.empty:
    skus = lgb_results['item_id'].unique()[:4]
    
    plt.figure(figsize=(15, 10))
    for i, sku in enumerate(skus):
        plt.subplot(2, 2, i+1)
        sku_lgb = lgb_results[lgb_results['item_id'] == sku].sort_values('date')
        plt.plot(sku_lgb['date'], sku_lgb['actual_sales'], label='Actual Sales', marker='o', alpha=0.6, color='black')
        plt.plot(sku_lgb['date'], sku_lgb['predicted_sales'], label='LightGBM Forecast', linestyle='-', linewidth=2, color='blue')
        
        if not prophet_results.empty:
            sku_prophet = prophet_results[prophet_results['item_id'] == sku].sort_values('date')
            if not sku_prophet.empty:
                plt.plot(sku_prophet['date'], sku_prophet['predicted_sales'], label='Prophet Forecast', linestyle='--', linewidth=2, color='orange')
                
        if not lstm_results.empty:
            sku_lstm = lstm_results[lstm_results['item_id'] == sku].sort_values('date')
            if not sku_lstm.empty:
                plt.plot(sku_lstm['date'], sku_lstm['predicted_sales'], label='LSTM Forecast', linestyle='-.', linewidth=2, color='green')
        
        plt.title(f"Forecast Comparison: {sku}")
        plt.legend()
        plt.xticks(rotation=45)

    plt.tight_layout()
    os.makedirs('../data', exist_ok=True)
    plt.savefig('../data/forecast_vs_actual_multi.png')
    plt.show()
else:
    print("No LightGBM results to visualize.")"""

    nb['cells'] = [
        nbf.v4.new_markdown_cell(markdown_1),
        nbf.v4.new_code_cell(code_1),
        nbf.v4.new_markdown_cell(markdown_2),
        nbf.v4.new_code_cell(code_2),
        nbf.v4.new_markdown_cell(markdown_3),
        nbf.v4.new_code_cell(code_3)
    ]

    os.makedirs('notebooks', exist_ok=True)
    with open('notebooks/results_dashboard.ipynb', 'w') as f:
        nbf.write(nb, f)
    print("Notebook generated at notebooks/results_dashboard.ipynb")

if __name__ == "__main__":
    generate_dashboard()
