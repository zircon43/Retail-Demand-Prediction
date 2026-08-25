import nbformat as nbf
import os

def generate_eda():
    nb = nbf.v4.new_notebook()

    markdown_1 = """# M5 Forecasting Competition - Exploratory Data Analysis (EDA)
This notebook provides a comprehensive EDA of the M5 dataset, uncovering patterns in intermittent demand, seasonality, and the effect of external variables like SNAP days and events."""

    code_1 = """import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

plt.style.use('ggplot')

# Load the data
print("Loading data...")
sales = pd.read_csv('../data/sales_train_evaluation.csv')
calendar = pd.read_csv('../data/calendar.csv')
prices = pd.read_csv('../data/sell_prices.csv')
print("Data loaded successfully.")"""

    markdown_2 = """## 1. Top Level Sales Analysis"""

    code_2 = """# Aggregate total sales across all items and stores per day
d_cols = [c for c in sales.columns if c.startswith('d_')]
total_sales = sales[d_cols].sum()

# Merge with calendar to get dates
total_sales = pd.DataFrame({'d': total_sales.index, 'sales': total_sales.values})
total_sales = total_sales.merge(calendar[['d', 'date']], on='d', how='left')
total_sales['date'] = pd.to_datetime(total_sales['date'])

plt.figure(figsize=(15, 6))
plt.plot(total_sales['date'], total_sales['sales'], linewidth=1)
plt.title('Total Daily Sales Across All Stores & Items', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Total Units Sold', fontsize=12)
plt.tight_layout()
plt.show()"""

    markdown_3 = """## 2. Seasonality and Day of Week Effects
The M5 dataset exhibits strong weekly seasonality. Let's visualize this."""

    code_3 = """# Map weekday names
total_sales = total_sales.merge(calendar[['date', 'wday', 'weekday']], on='date')

plt.figure(figsize=(10, 6))
sns.boxplot(x='weekday', y='sales', data=total_sales, order=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
plt.title('Sales Distribution by Day of Week', fontsize=16)
plt.ylabel('Total Units Sold')
plt.show()"""

    markdown_4 = """## 3. Intermittent Demand (Zero-Inflated)
Many items in retail are not sold every day. This creates "zero-inflated" data, which is why the top submission used Tweedie regression."""

    code_4 = """# Sample a few items to check the proportion of zero sales
sample_sales = sales.sample(1000, random_state=42)[d_cols]
zero_prop = (sample_sales == 0).sum(axis=1) / len(d_cols)

plt.figure(figsize=(10, 6))
sns.histplot(zero_prop, bins=50, kde=True)
plt.title('Proportion of Days with Zero Sales per Item (Sample)', fontsize=16)
plt.xlabel('Proportion of Zero Sales')
plt.ylabel('Number of Items')
plt.axvline(zero_prop.mean(), color='red', linestyle='dashed', label=f'Mean: {zero_prop.mean():.2f}')
plt.legend()
plt.show()"""

    nb['cells'] = [
        nbf.v4.new_markdown_cell(markdown_1),
        nbf.v4.new_code_cell(code_1),
        nbf.v4.new_markdown_cell(markdown_2),
        nbf.v4.new_code_cell(code_2),
        nbf.v4.new_markdown_cell(markdown_3),
        nbf.v4.new_code_cell(code_3),
        nbf.v4.new_markdown_cell(markdown_4),
        nbf.v4.new_code_cell(code_4)
    ]

    os.makedirs('notebooks', exist_ok=True)
    with open('notebooks/eda_m5.ipynb', 'w') as f:
        nbf.write(nb, f)
    print("Generated notebooks/eda_m5.ipynb")

if __name__ == "__main__":
    generate_eda()
