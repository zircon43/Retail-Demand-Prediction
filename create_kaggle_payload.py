import nbformat as nbf
import os

def create_payload():
    nb = nbf.v4.new_notebook()

    # Introduction
    cells = []
    cells.append(nbf.v4.new_markdown_cell("""# M5 Forecasting: Top Submission Strategy & Baselines
This notebook is designed to run directly in the Kaggle environment. It will:
1. Load the M5 dataset using DuckDB.
2. Perform feature engineering (lags, rolling stats).
3. Train the winning LightGBM strategy (Tweedie loss, per store).
4. Train Prophet & LSTM baselines.
5. Save the forecasts to output for you to download and analyze locally.

**Requirements**: Add the "M5 Forecasting - Accuracy" dataset to your Kaggle notebook."""))

    # Setup
    cells.append(nbf.v4.new_code_cell("""!pip install duckdb
import pandas as pd
import numpy as np
import duckdb
import lightgbm as lgb
from prophet import Prophet
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import multiprocessing
import gc
import os
import warnings
warnings.filterwarnings('ignore')"""))

    # Data Pipeline
    cells.append(nbf.v4.new_markdown_cell("## 1. Data Pipeline & Feature Engineering"))
    cells.append(nbf.v4.new_code_cell("""# Base path in Kaggle
BASE_PATH = '/kaggle/input/competitions/m5-forecasting-accuracy'

# Fallbacks if path differs slightly
if not os.path.exists(BASE_PATH):
    BASE_PATH = '../input/m5-forecasting-accuracy'
if not os.path.exists(BASE_PATH):
    BASE_PATH = '../data'

def build_data_pipeline():
    print("Building Data Pipeline with DuckDB...")
    # Use disk-backed database instead of memory to prevent OOM
    con = duckdb.connect('m5_data.db')
    
    # Create tables
    con.execute(f"CREATE TABLE calendar AS SELECT * FROM read_csv_auto('{BASE_PATH}/calendar.csv');")
    con.execute(f"CREATE TABLE sell_prices AS SELECT * FROM read_csv_auto('{BASE_PATH}/sell_prices.csv');")
    con.execute(f"CREATE TABLE sales_raw AS SELECT * FROM read_csv_auto('{BASE_PATH}/sales_train_evaluation.csv');")
    
    print("Unpivoting sales data...")
    con.execute(\"\"\"
    CREATE TABLE sales_unpivoted AS
    UNPIVOT sales_raw
    ON COLUMNS(* EXCLUDE (id, item_id, dept_id, cat_id, store_id, state_id))
    INTO
        NAME day_id
        VALUE sales;
    \"\"\")
    
    print("Engineering features...")
    con.execute(\"\"\"
    CREATE TABLE modeling_data AS
    WITH sales_with_dates AS (
        SELECT 
            s.item_id, s.store_id, s.sales,
            c.date::DATE AS date, c.wm_yr_wk, c.wday, c.month, c.year, c.event_name_1
        FROM sales_unpivoted s
        JOIN calendar c ON s.day_id = c.d
    ),
    sales_with_prices AS (
        SELECT s.*, p.sell_price
        FROM sales_with_dates s
        LEFT JOIN sell_prices p 
            ON s.item_id = p.item_id AND s.store_id = p.store_id AND s.wm_yr_wk = p.wm_yr_wk
    ),
    features AS (
        SELECT *,
            LAG(sales, 1) OVER w AS sales_lag_1,
            LAG(sales, 7) OVER w AS sales_lag_7,
            LAG(sales, 14) OVER w AS sales_lag_14,
            LAG(sales, 28) OVER w AS sales_lag_28,
            AVG(sales) OVER w_7 AS rolling_avg_7,
            AVG(sales) OVER w_28 AS rolling_avg_28,
            STDDEV_SAMP(sales) OVER w_7 AS rolling_std_7,
            STDDEV_SAMP(sales) OVER w_28 AS rolling_std_28,
            SUM(sales) OVER (PARTITION BY item_id, store_id, month, year ORDER BY date) AS month_to_date_sales
        FROM sales_with_prices
        WINDOW 
            w AS (PARTITION BY item_id, store_id ORDER BY date),
            w_7 AS (PARTITION BY item_id, store_id ORDER BY date ROWS BETWEEN 7 PRECEDING AND 1 PRECEDING),
            w_28 AS (PARTITION BY item_id, store_id ORDER BY date ROWS BETWEEN 28 PRECEDING AND 1 PRECEDING)
    )
    SELECT * FROM features WHERE sales_lag_28 IS NOT NULL;
    \"\"\")
    
    print("Exporting data to Parquet to save RAM...")
    con.execute("COPY (SELECT * FROM modeling_data) TO 'modeling_data.parquet' (FORMAT PARQUET)")
    
    # Get distinct stores to loop over
    stores_df = con.execute("SELECT DISTINCT store_id FROM modeling_data").fetchdf()
    stores = stores_df['store_id'].tolist()
    
    con.close()
    return stores

stores = build_data_pipeline()
gc.collect()
print(f"Data pipeline finished. Unique stores: {stores}")"""))

    # LightGBM
    cells.append(nbf.v4.new_markdown_cell("## 2. Top Submission Strategy: LightGBM Tweedie per Store"))
    cells.append(nbf.v4.new_code_cell("""def train_lightgbm_per_store(store_id, test_days=28):
    print(f"\\n--- Training LightGBM for store: {store_id} ---")
    # Load ONLY this store's data from Parquet into RAM
    store_df = pd.read_parquet('modeling_data.parquet', filters=[('store_id', '=', store_id)])
    
    # Optimize memory
    for col in store_df.select_dtypes(include=['float64']).columns:
        store_df[col] = pd.to_numeric(store_df[col], downcast='float')
        
    store_df['date'] = pd.to_datetime(store_df['date'])
    
    target = 'sales'
    features = [
        'sell_price', 'wday', 'month', 'year',
        'sales_lag_1', 'sales_lag_7', 'sales_lag_14', 'sales_lag_28',
        'rolling_avg_7', 'rolling_avg_28', 'rolling_std_7', 'rolling_std_28', 
        'month_to_date_sales'
    ]
    
    store_df['item_id_cat'] = store_df['item_id'].astype('category').cat.codes
    store_df['event_name_1'] = store_df['event_name_1'].fillna('None')
    store_df['event_name_1_cat'] = store_df['event_name_1'].astype('category').cat.codes
    features.extend(['item_id_cat', 'event_name_1_cat'])
    
    max_date = store_df['date'].max()
    split_date = max_date - pd.Timedelta(days=test_days)
    val_split_date = split_date - pd.Timedelta(days=test_days)
    
    train = store_df[store_df['date'] <= split_date].copy()
    test = store_df[store_df['date'] > split_date].copy()
    
    X_train, y_train = train[features], train[target]
    X_test, y_test = test[features], test[target]
    
    X_val = train[train['date'] > val_split_date][features]
    y_val = train[train['date'] > val_split_date][target]
    
    train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=['item_id_cat', 'event_name_1_cat'])
    val_data = lgb.Dataset(X_val, label=y_val, categorical_feature=['item_id_cat', 'event_name_1_cat'], reference=train_data)
    
    params = {
        'objective': 'tweedie', 'tweedie_variance_power': 1.1, 'metric': 'rmse',
        'learning_rate': 0.05, 'subsample': 0.8, 'colsample_bytree': 0.8,
        'max_depth': 8, 'num_leaves': 64, 'verbose': -1, 'seed': 42
    }
    
    model = lgb.train(params, train_data, num_boost_round=1000, valid_sets=[train_data, val_data],
                      callbacks=[lgb.early_stopping(stopping_rounds=50)])
    
    test['predicted_sales'] = model.predict(X_test)
    test['predicted_sales'] = test['predicted_sales'].clip(lower=0)
    
    return test[['date', 'item_id', 'store_id', 'sales', 'predicted_sales']].rename(columns={'sales':'actual_sales'})

# Run LightGBM for all stores
lgb_results = []
for store in stores:
    res = train_lightgbm_per_store(store)
    lgb_results.append(res)
    gc.collect()

lgb_final = pd.concat(lgb_results)
lgb_final.to_parquet('lightgbm_results.parquet')
print("Saved LightGBM results.")"""))

    # Prophet Baseline
    cells.append(nbf.v4.new_markdown_cell("## 3. Baseline: Prophet"))
    cells.append(nbf.v4.new_code_cell("""def train_prophet_single(args):
    df_subset, item_id, store_id, test_days = args
    prophet_df = df_subset[['date', 'sales']].rename(columns={'date': 'ds', 'sales': 'y'})
    prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])
    
    max_date = prophet_df['ds'].max()
    split_date = max_date - pd.Timedelta(days=test_days)
    
    train = prophet_df[prophet_df['ds'] <= split_date]
    test = prophet_df[prophet_df['ds'] > split_date]
    
    if len(train) < 50:
        return pd.DataFrame()
        
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    model.fit(train)
    forecast = model.predict(test[['ds']])
    
    results = test.copy()
    results['yhat'] = forecast['yhat'].values.clip(min=0)
    results = results.rename(columns={'ds': 'date', 'y': 'actual_sales', 'yhat': 'predicted_sales'})
    results['item_id'] = item_id
    results['store_id'] = store_id
    return results

# To keep notebook runtimes reasonable, Prophet will run on a subset of items
test_days = 28
tasks = []
for store in stores:
    store_df = pd.read_parquet('modeling_data.parquet', filters=[('store_id', '=', store)])
    items_sample = store_df['item_id'].unique()[:100]
    for item in items_sample:
        df_subset = store_df[store_df['item_id'] == item]
        tasks.append((df_subset, item, store, test_days))
    del store_df
    gc.collect()

print(f"Training Prophet for {len(tasks)} series...")
with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
    prophet_res = pool.map(train_prophet_single, tasks)

prophet_final = pd.concat([r for r in prophet_res if not r.empty])
prophet_final.to_parquet('prophet_results.parquet')
print("Saved Prophet results.")"""))

    # LSTM Baseline
    cells.append(nbf.v4.new_markdown_cell("## 4. Baseline: LSTM (PyTorch)"))
    cells.append(nbf.v4.new_code_cell("""class TimeSeriesDataset(Dataset):
    def __init__(self, X, y, sequence_length):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.X) - self.sequence_length

    def __getitem__(self, idx):
        return (self.X[idx:idx+self.sequence_length], self.y[idx+self.sequence_length])

class LSTMForecaster(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTMForecaster, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :]) 
        return out

# Also running LSTM on a subset for demonstration speed
lstm_results = []
seq_length = 14
test_days = 28
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for store in stores:
    print(f"\\n--- Training LSTM for store {store} ---")
    store_df = pd.read_parquet('modeling_data.parquet', filters=[('store_id', '=', store)])
    items_sample = store_df['item_id'].unique()[:100]
    store_df = store_df[store_df['item_id'].isin(items_sample)]
    
    # Simplistic scaling for single-feature sequential prediction
    scaler = StandardScaler()
    store_df['sales_scaled'] = scaler.fit_transform(store_df[['sales']])
    store_df['date'] = pd.to_datetime(store_df['date'])
    
    max_date = store_df['date'].max()
    split_date = max_date - pd.Timedelta(days=test_days)
    
    train = store_df[store_df['date'] <= split_date].copy()
    test = store_df[store_df['date'] > split_date].copy()
    
    for item in items_sample:
        item_train = train[train['item_id'] == item].sort_values('date')
        item_test = test[test['item_id'] == item].sort_values('date')
        
        if len(item_train) < 50:
            continue
            
        X_train = item_train[['sales_scaled']].values
        y_train = item_train['sales_scaled'].values
        
        dataset = TimeSeriesDataset(X_train, y_train, seq_length)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
        
        model = LSTMForecaster(input_dim=1, hidden_dim=32, num_layers=2, output_dim=1).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        model.train()
        for epoch in range(10):
            for batch_X, batch_y in dataloader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
                loss.backward()
                optimizer.step()
                
        model.eval()
        inputs = item_train['sales_scaled'].values[-seq_length:].tolist()
        preds = []
        with torch.no_grad():
            for _ in range(len(item_test)):
                seq = torch.tensor(inputs[-seq_length:], dtype=torch.float32).view(1, seq_length, 1).to(device)
                pred = model(seq).item()
                preds.append(pred)
                inputs.append(pred)
                
        pred_unscaled = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
        pred_unscaled = np.clip(pred_unscaled, 0, None)
        
        res = item_test.copy().rename(columns={'sales': 'actual_sales'})
        res['predicted_sales'] = pred_unscaled
        lstm_results.append(res[['date', 'item_id', 'store_id', 'actual_sales', 'predicted_sales']])

lstm_final = pd.concat(lstm_results)
lstm_final.to_parquet('lstm_results.parquet')
print("Saved LSTM results.")"""))

    # Completion
    cells.append(nbf.v4.new_markdown_cell("## 5. Download Instructions\nAll models have run and saved their results to `lightgbm_results.parquet`, `prophet_results.parquet`, and `lstm_results.parquet`. You can download these files from the Kaggle Output panel on the right sidebar and use them locally for downstream analysis and visualization!"))

    nb['cells'] = cells
    
    os.makedirs('notebooks', exist_ok=True)
    with open('notebooks/kaggle_payload.ipynb', 'w') as f:
        nbf.write(nb, f)
    print("Successfully created notebooks/kaggle_payload.ipynb")

if __name__ == '__main__':
    create_payload()
