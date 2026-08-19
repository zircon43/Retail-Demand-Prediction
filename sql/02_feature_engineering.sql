-- sql/02_feature_engineering.sql
DROP TABLE IF EXISTS modeling_data;

CREATE TABLE modeling_data AS
WITH sales_with_dates AS (
    SELECT 
        s.item_id,
        s.store_id,
        s.sales,
        c.date::DATE AS date,
        c.wm_yr_wk,
        c.wday,
        c.month,
        c.year,
        c.event_name_1
    FROM sales_unpivoted s
    JOIN calendar c ON s.day_id = c.d
),
sales_with_prices AS (
    SELECT
        s.*,
        p.sell_price
    FROM sales_with_dates s
    LEFT JOIN sell_prices p 
        ON s.item_id = p.item_id 
        AND s.store_id = p.store_id 
        AND s.wm_yr_wk = p.wm_yr_wk
),
features AS (
    SELECT 
        *,
        -- Window functions for lag features
        LAG(sales, 1) OVER (PARTITION BY item_id, store_id ORDER BY date) AS sales_lag_1,
        LAG(sales, 7) OVER (PARTITION BY item_id, store_id ORDER BY date) AS sales_lag_7,
        LAG(sales, 14) OVER (PARTITION BY item_id, store_id ORDER BY date) AS sales_lag_14,
        LAG(sales, 28) OVER (PARTITION BY item_id, store_id ORDER BY date) AS sales_lag_28,
        
        -- Moving averages (rolling features)
        AVG(sales) OVER (PARTITION BY item_id, store_id ORDER BY date ROWS BETWEEN 7 PRECEDING AND 1 PRECEDING) AS rolling_avg_7,
        AVG(sales) OVER (PARTITION BY item_id, store_id ORDER BY date ROWS BETWEEN 28 PRECEDING AND 1 PRECEDING) AS rolling_avg_28,
        
        -- Moving standard deviations
        STDDEV_SAMP(sales) OVER (PARTITION BY item_id, store_id ORDER BY date ROWS BETWEEN 7 PRECEDING AND 1 PRECEDING) AS rolling_std_7,
        STDDEV_SAMP(sales) OVER (PARTITION BY item_id, store_id ORDER BY date ROWS BETWEEN 28 PRECEDING AND 1 PRECEDING) AS rolling_std_28,
        
        -- Cumulative sum of sales over the month
        SUM(sales) OVER (PARTITION BY item_id, store_id, month, year ORDER BY date) AS month_to_date_sales
    FROM sales_with_prices
)
SELECT * FROM features
-- Filter out rows where lag features are null to ensure clean data for ML
WHERE sales_lag_28 IS NOT NULL;
