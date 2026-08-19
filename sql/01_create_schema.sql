-- sql/01_create_schema.sql
-- Drop tables if they exist
DROP TABLE IF EXISTS calendar;
DROP TABLE IF EXISTS sell_prices;
DROP TABLE IF EXISTS sales_raw;
DROP TABLE IF EXISTS sales_unpivoted;

-- Load raw data from real M5 dataset
CREATE TABLE calendar AS SELECT * FROM read_csv_auto('data/calendar.csv');
CREATE TABLE sell_prices AS SELECT * FROM read_csv_auto('data/sell_prices.csv');
CREATE TABLE sales_raw AS SELECT * FROM read_csv_auto('data/sales_train_evaluation.csv');

-- Unpivot the sales data from wide to long format
-- This transforms columns d_1 to d_1913 into rows
CREATE TABLE sales_unpivoted AS
UNPIVOT sales_raw
ON COLUMNS(* EXCLUDE (id, item_id, dept_id, cat_id, store_id, state_id))
INTO
    NAME day_id
    VALUE sales;
