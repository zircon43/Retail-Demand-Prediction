import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import os

class TimeSeriesDataset(Dataset):
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
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :]) 
        return out

def run_lstm_models(data_path='data/modeling_data.parquet', test_days=28, store_id='CA_1'):
    print(f"Loading data for LSTM (Store: {store_id})...")
    df = pd.read_parquet(data_path)
    df = df[df['store_id'] == store_id].copy()
    df['date'] = pd.to_datetime(df['date'])
    
    # We will aggregate by date and item for sequence processing
    features = ['sell_price', 'sales_lag_1', 'sales_lag_7']
    df = df.dropna(subset=features)
    
    # Pivot to get wide format (Items as columns) - simplified multi-variate approach
    # For a full implementation, panel data processing is preferred. 
    # Here we demonstrate a basic sequential approach for a subset of items to prove the concept.
    items = df['item_id'].unique()[:100] # Take 100 items to train quickly
    df = df[df['item_id'].isin(items)]
    
    # Scale features
    scaler = StandardScaler()
    df['sales_scaled'] = scaler.fit_transform(df[['sales']])
    
    max_date = df['date'].max()
    split_date = max_date - pd.Timedelta(days=test_days)
    
    train = df[df['date'] <= split_date].copy()
    test = df[df['date'] > split_date].copy()
    
    results = []
    
    print(f"Training LSTM for {len(items)} items...")
    for item in items:
        item_train = train[train['item_id'] == item].sort_values('date')
        item_test = test[test['item_id'] == item].sort_values('date')
        
        if len(item_train) < 50:
            continue
            
        seq_length = 14
        X_train = item_train[['sales_scaled']].values
        y_train = item_train['sales_scaled'].values
        
        dataset = TimeSeriesDataset(X_train, y_train, seq_length)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        model = LSTMForecaster(input_dim=1, hidden_dim=32, num_layers=2, output_dim=1)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Train
        model.train()
        epochs = 10
        for epoch in range(epochs):
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
                loss.backward()
                optimizer.step()
                
        # Predict 
        model.eval()
        inputs = item_train['sales_scaled'].values[-seq_length:].tolist()
        predictions_scaled = []
        
        with torch.no_grad():
            for _ in range(len(item_test)):
                seq = torch.tensor(inputs[-seq_length:], dtype=torch.float32).view(1, seq_length, 1)
                pred = model(seq).item()
                predictions_scaled.append(pred)
                inputs.append(pred)
                
        pred_unscaled = scaler.inverse_transform(np.array(predictions_scaled).reshape(-1, 1)).flatten()
        pred_unscaled = np.clip(pred_unscaled, 0, None) # No negative sales
        
        res = item_test.copy()
        res = res.rename(columns={'sales': 'actual_sales'})
        res['predicted_sales'] = pred_unscaled
        results.append(res[['date', 'item_id', 'store_id', 'actual_sales', 'predicted_sales']])
        
    final_results = pd.concat(results, ignore_index=True)
    final_results.to_parquet('data/lstm_forecast_results.parquet')
    print("Saved LSTM forecasts to data/lstm_forecast_results.parquet")
    return final_results

if __name__ == "__main__":
    run_lstm_models()
