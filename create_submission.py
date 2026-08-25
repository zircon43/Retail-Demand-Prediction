import pandas as pd
import numpy as np

def create_submission():
    print("Loading LightGBM predictions...")
    preds = pd.read_parquet('lightgbm_results.parquet')
    
    # We only care about the forecasted values
    preds = preds[['date', 'item_id', 'store_id', 'predicted_sales']].copy()
    
    # Sort by date so we can map them to F1, F2, ..., F28
    preds = preds.sort_values('date')
    
    # Create the 'id' column for evaluation
    preds['id'] = preds['item_id'] + "_" + preds['store_id'] + "_evaluation"
    
    # Rank dates to get the day number (1 to 28)
    preds['day_num'] = preds.groupby('id').cumcount() + 1
    preds['day_col'] = 'F' + preds['day_num'].astype(str)
    
    print("Pivoting data to wide format...")
    # Pivot from long to wide
    wide_preds = preds.pivot(index='id', columns='day_col', values='predicted_sales').reset_index()
    
    # Load sample submission to ensure we match the exact format and row order
    print("Loading sample_submission.csv template...")
    sample_sub = pd.read_csv('data/sample_submission.csv')
    
    print("Merging predictions with template...")
    # We need both _validation and _evaluation rows for a valid submission.
    # The actual predictions we made are for the evaluation set (the future 28 days).
    # For the validation set (the past 28 days), we can either use our predictions or just leave as 0 
    # since the competition is over and evaluation is what matters for the private leaderboard score.
    # We will duplicate the wide_preds for validation just to ensure the file is accepted without NaNs.
    
    val_preds = wide_preds.copy()
    val_preds['id'] = val_preds['id'].str.replace('_evaluation', '_validation')
    
    all_preds = pd.concat([val_preds, wide_preds], ignore_index=True)
    
    # Merge with sample submission to keep exact row order and fill missing with 0
    final_sub = pd.merge(sample_sub[['id']], all_preds, on='id', how='left').fillna(0)
    
    print("Saving submission.csv...")
    final_sub.to_csv('submission.csv', index=False)
    print(f"Successfully saved submission.csv with {len(final_sub)} rows!")

if __name__ == '__main__':
    create_submission()
