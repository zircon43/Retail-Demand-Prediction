import duckdb
import os

def run_sql_pipeline():
    print("Starting SQL Pipeline...")
    
    # Connect to a persistent DuckDB database file
    con = duckdb.connect('data/inventory.db')
    
    # Run Schema Creation
    print("Running 01_create_schema.sql...")
    with open('sql/01_create_schema.sql', 'r') as f:
        schema_sql = f.read()
    con.execute(schema_sql)
    
    # Run Feature Engineering
    print("Running 02_feature_engineering.sql...")
    with open('sql/02_feature_engineering.sql', 'r') as f:
        feature_sql = f.read()
    con.execute(feature_sql)
    
    # Verify results
    row_count = con.execute("SELECT COUNT(*) FROM modeling_data").fetchone()[0]
    print(f"Pipeline complete! Created modeling_data table with {row_count} rows.")
    
    # Export to parquet for easy pandas loading in modeling scripts
    con.execute("COPY modeling_data TO 'data/modeling_data.parquet' (FORMAT PARQUET)")
    print("Exported modeling_data to data/modeling_data.parquet")

    con.close()

if __name__ == "__main__":
    run_sql_pipeline()
