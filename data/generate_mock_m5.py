import pandas as pd
import numpy as np
import datetime
import os

# Create data directory
os.makedirs('data', exist_ok=True)

# Parameters
NUM_SKUS = 10
NUM_DAYS = 500
START_DATE = datetime.date(2014, 1, 1)

print("Generating synthetic M5 dataset...")
np.random.seed(42)

# 1. Generate calendar.csv
dates = [START_DATE + datetime.timedelta(days=i) for i in range(NUM_DAYS)]
calendar_df = pd.DataFrame({
    'date': dates,
    'wm_yr_wk': [11401 + (i // 7) for i in range(NUM_DAYS)], # Mock week ID
    'weekday': [d.strftime('%A') for d in dates],
    'wday': [(d.weekday() + 2) % 7 + 1 for d in dates], # Saturday = 1
    'month': [d.month for d in dates],
    'year': [d.year for d in dates],
    'd': [f"d_{i+1}" for i in range(NUM_DAYS)],
    'event_name_1': [np.nan] * NUM_DAYS # Simplified
})
calendar_df.to_csv('data/calendar.csv', index=False)
print("- calendar.csv generated.")

# 2. Generate sales_train_validation.csv
store_id = 'CA_1'
state_id = 'CA'
dept_id = 'HOBBIES_1'
cat_id = 'HOBBIES'

sales_data = []
for i in range(NUM_SKUS):
    item_id = f"HOBBIES_1_{str(i+1).zfill(3)}"
    id_str = f"{item_id}_{store_id}_validation"
    
    # Generate random walk with seasonality and noise
    base_sales = np.random.randint(10, 50)
    sales = [max(0, int(base_sales + 10 * np.sin(2 * np.pi * j / 365.25) + np.random.normal(0, 5))) for j in range(NUM_DAYS)]
    
    row = {
        'id': id_str,
        'item_id': item_id,
        'dept_id': dept_id,
        'cat_id': cat_id,
        'store_id': store_id,
        'state_id': state_id
    }
    for j in range(NUM_DAYS):
        row[f"d_{j+1}"] = sales[j]
    sales_data.append(row)

sales_df = pd.DataFrame(sales_data)
sales_df.to_csv('data/sales_train_validation.csv', index=False)
print("- sales_train_validation.csv generated.")

# 3. Generate sell_prices.csv
prices_data = []
for i in range(NUM_SKUS):
    item_id = f"HOBBIES_1_{str(i+1).zfill(3)}"
    base_price = np.round(np.random.uniform(2.99, 19.99), 2)
    # Price constant for each week
    weeks = calendar_df['wm_yr_wk'].unique()
    for w in weeks:
        # Occasional price changes
        if np.random.rand() < 0.05:
            base_price = np.round(base_price * np.random.uniform(0.9, 1.1), 2)
        prices_data.append({
            'store_id': store_id,
            'item_id': item_id,
            'wm_yr_wk': w,
            'sell_price': base_price
        })

prices_df = pd.DataFrame(prices_data)
prices_df.to_csv('data/sell_prices.csv', index=False)
print("- sell_prices.csv generated.")

# 4. Generate warehouse_inventory.csv (Our custom addition)
# We track daily inventory levels, lead times, and holding costs
inventory_data = []
for i in range(NUM_SKUS):
    item_id = f"HOBBIES_1_{str(i+1).zfill(3)}"
    current_stock = np.random.randint(100, 300)
    lead_time = np.random.randint(3, 14) # Days to restock
    holding_cost = np.round(np.random.uniform(0.01, 0.05), 4) # Daily cost per unit
    
    # Retrieve daily sales to simulate stock depletion
    daily_sales = sales_df.loc[sales_df['item_id'] == item_id].iloc[0, 6:].values
    
    reorder_point = 150 # Simplified ROP for mock generation
    order_qty = 200
    pending_orders = {}
    
    for j, date in enumerate(dates):
        # Receive orders
        if j in pending_orders:
            current_stock += pending_orders.pop(j)
            
        sales_today = daily_sales[j]
        current_stock = max(0, current_stock - sales_today)
        
        # Simple replenishment logic to keep stock > 0 for history
        if current_stock < reorder_point:
            delivery_day = j + lead_time
            pending_orders[delivery_day] = pending_orders.get(delivery_day, 0) + order_qty
            
        inventory_data.append({
            'date': date,
            'store_id': store_id,
            'item_id': item_id,
            'stock_on_hand': current_stock,
            'lead_time_days': lead_time,
            'daily_holding_cost': holding_cost
        })
        
inventory_df = pd.DataFrame(inventory_data)
inventory_df.to_csv('data/warehouse_inventory.csv', index=False)
print("- warehouse_inventory.csv generated.")

print("Mock data generation complete!")
