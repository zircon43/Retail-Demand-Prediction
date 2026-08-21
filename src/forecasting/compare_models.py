import pandas as pd
from evaluation import evaluate_forecast

def compare_models():
    print("Comparing Forecasting Models (Prophet vs XGBoost)...\n")
    
    prophet_df = pd.read_parquet('data/prophet_forecast_results.parquet')
    xgb_df = pd.read_parquet('data/xgboost_forecast_results.parquet')
    
    # Calculate Prophet Metrics
    prophet_metrics = evaluate_forecast(prophet_df['actual_sales'], prophet_df['predicted_sales'])
    
    # Calculate XGBoost Metrics
    xgb_metrics = evaluate_forecast(xgb_df['actual_sales'], xgb_df['predicted_sales'])
    
    comparison = pd.DataFrame([
        {'Model': 'Prophet (Classical)', 'MAPE (%)': prophet_metrics['MAPE'], 'RMSE': prophet_metrics['RMSE']},
        {'Model': 'XGBoost (ML)', 'MAPE (%)': xgb_metrics['MAPE'], 'RMSE': xgb_metrics['RMSE']}
    ])
    
    print(comparison.to_string(index=False))
    comparison.to_csv('data/model_comparison.csv', index=False)
    
if __name__ == "__main__":
    compare_models()
