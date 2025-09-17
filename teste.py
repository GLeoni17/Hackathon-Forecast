import pandas as pd

if __name__=='__main__':
    weekly_full = pd.read_parquet("/home/luanguerra/Teste/data/weekly_full.parquet")
    target_weeks = pd.date_range("2023-01-02", periods=5, freq='7D')

    pairs = weekly_full[['pdv_id','produto_id']].drop_duplicates().reset_index(drop=True)
    
    future_df = (
    pairs.assign(key=1)
         .merge(pd.DataFrame({'week_start': target_weeks, 'key':1}), on="key")
         .drop(columns="key")   
    )
    
    merged = future_df.merge(
    weekly_full,
    on=["week_start","pdv_id","produto_id"],
    how="left",   
    suffixes=("", "_hist")
    )

    merged = merged.sort_values(['week_start','pdv_id','produto_id']).reset_index(drop=True)
    merged['quantity'] = merged['quantity'].fillna(0)
    
    merged['series_id'] = merged['pdv_id'] + '_' + merged['produto_id'] 
    
    merged.to_parquet('/home/luanguerra/Teste/data/weekly_future.parquet', index=False, compression='snappy')