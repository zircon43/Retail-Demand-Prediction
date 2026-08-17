import pandas as pd
import numpy as np

def calculate_inventory_policy(forecast_df, inventory_df, service_level_z=1.645):
    """
    Calculates safety stock and Reorder Point (ROP) based on forecast.
    Simulates the inventory over the backtest period to get business metrics.
    """
    print("Calculating optimal inventory policy based on forecasts...")
    
    # Merge forecast with inventory data
    forecast_df['date'] = pd.to_datetime(forecast_df['date'])
    inventory_df['date'] = pd.to_datetime(inventory_df['date'])
    
    merged = pd.merge(
        forecast_df, 
        inventory_df[['date', 'item_id', 'store_id', 'lead_time_days', 'daily_holding_cost']], 
        on=['date', 'item_id', 'store_id'], 
        how='left'
    )
    
    # Calculate RMSE per SKU as a proxy for demand uncertainty (std dev of forecast error)
    sku_rmse = merged.groupby(['store_id', 'item_id']).apply(
        lambda x: np.sqrt(np.mean((x['actual_sales'] - x['predicted_sales'])**2))
    ).reset_index(name='rmse')
    
    merged = pd.merge(merged, sku_rmse, on=['store_id', 'item_id'], how='left')
    
    # Safety Stock = Z * RMSE * sqrt(Lead_Time)
    merged['safety_stock'] = service_level_z * merged['rmse'] * np.sqrt(merged['lead_time_days'])
    
    # ROP = Forecasted Demand over Lead Time + Safety Stock
    # For simplicity, we use (daily predicted sales * lead time) as the lead time demand
    merged['reorder_point'] = (merged['predicted_sales'] * merged['lead_time_days']) + merged['safety_stock']
    
    # --- Simulate Business Metrics ---
    # We will simulate "Stockouts" if actual sales > current stock.
    # We'll use a naive policy vs our ML policy.
    
    # Naive policy: Reorder point is just average past sales * lead time (no safety stock)
    # Here we simulate the cost difference.
    # To do a full simulation properly, we'd need a loop. For portfolio proof-of-work, 
    # we calculate the theoretical stockout probability and holding costs.
    
    # Holding Cost = Safety Stock * Daily Holding Cost
    merged['holding_cost'] = merged['safety_stock'] * merged['daily_holding_cost']
    
    # Simulate actual demand over the lead time (what the ROP needs to cover)
    # We approximate it by taking the actual daily sales and scaling it by lead time, adding random noise
    np.random.seed(42)
    merged['actual_lead_time_demand'] = merged['actual_sales'] * merged['lead_time_days'] + np.random.normal(0, merged['rmse'], len(merged))
    
    # Stockout risk (if actual lead time demand > ROP)
    merged['stockout_units_ml'] = np.maximum(0, merged['actual_lead_time_demand'] - merged['reorder_point'])
    
    # Naive ROP (uses naive moving average forecast * lead time, no safety stock)
    # We simulate naive forecast as actual_sales lagged by 1 day
    naive_forecast_daily = merged['actual_sales'].shift(1).fillna(merged['actual_sales'].mean())
    merged['naive_reorder_point'] = naive_forecast_daily * merged['lead_time_days']
    merged['stockout_units_naive'] = np.maximum(0, merged['actual_lead_time_demand'] - merged['naive_reorder_point'])
    
    total_stockout_ml = merged['stockout_units_ml'].sum()
    total_stockout_naive = merged['stockout_units_naive'].sum()
    total_holding_cost = merged['holding_cost'].sum()
    
    reduction = (total_stockout_naive - total_stockout_ml) / total_stockout_naive * 100 if total_stockout_naive > 0 else 0
    
    print("\n" + "="*50)
    print("INVENTORY OPTIMIZATION RESULTS (SIMULATED)")
    print("="*50)
    print(f"Total Stockout Units (Naive ROP): {total_stockout_naive:,.0f}")
    print(f"Total Stockout Units (ML ROP with Safety Stock): {total_stockout_ml:,.0f}")
    print(f"Business Impact: Reduced projected stockout rate by {reduction:.1f}%")
    print(f"Incremental Carrying Cost for Safety Stock: ${total_holding_cost:,.2f}")
    print("="*50 + "\n")
    
    merged.to_parquet('data/inventory_optimization_results.parquet')
    return merged

if __name__ == "__main__":
    xgb_results = pd.read_parquet('data/xgboost_forecast_results.parquet')
    inv_data = pd.read_csv('data/warehouse_inventory.csv')
    calculate_inventory_policy(xgb_results, inv_data)
