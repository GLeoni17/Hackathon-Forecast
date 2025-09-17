import pandas as pd

def make_lags(df, lags=[1,2,3,4]):
    for l in lags:
        df[f'lag_{l}'] = df.groupby('series_id')['quantity'].shift(l)
    return df

def make_rollings(df, windows=[4,8]):
    for w in windows:
        df[f'roll_mean_{w}'] = df.groupby('series_id')['quantity'].shift(1).rolling(w).mean().reset_index(0,drop=True)
    return df


""" def extract_features(weekly_full):
    
    weekly_full = weekly_full.sort_values(['pdv_id','produto_id','week_start'])
    weekly_full['series_id'] = weekly_full['pdv_id'].astype(str)+'_'+weekly_full['produto_id'].astype(str)
    
    weekly_full = make_lags(weekly_full)
    weekly_full = make_rollings(weekly_full)
    
    weekly_full.to_parquet('/home/luanguerra/Teste/data/weekly_full_features.parquet', index=False) """

def extract_features(weekly_full, weekly_test):
    
    weekly_full = weekly_full.sort_values(['pdv_id','produto_id','week_start'])
    weekly_test = weekly_test.sort_values(['pdv_id','produto_id','week_start'])
    
    print(weekly_full.shape)
    print(weekly_test.shape)
    
    weekly_full['series_id'] = weekly_full['pdv_id'].astype(str)+'_'+weekly_full['produto_id'].astype(str)
    
    full = pd.concat([weekly_full, weekly_test], ignore_index=True, sort=False)
    
    full = full.sort_values(['series_id','week_start']).reset_index(drop=True)
    
    full = make_lags(full)
    full = make_rollings(full)
    
    print(full.shape)
    
    cutoff = pd.Timestamp("2023-01-02")
    
    weekly_train = full[full['week_start'] < cutoff].copy()
    weekly_test   = full[(full['week_start'] >= cutoff) & (full['week_start'] < cutoff + pd.Timedelta(weeks=5))].copy()
    
    weekly_train.to_parquet('/home/luanguerra/Teste/data/weekly_train.parquet', index=False)
    weekly_test.to_parquet('/home/luanguerra/Teste/data/weekly_test.parquet', index=False)

if __name__=='__main__':
    
    weekly_full = pd.read_parquet("/home/luanguerra/Teste/data/weekly_full.parquet")
    weekly_future = pd.read_parquet("/home/luanguerra/Teste/data/weekly_future.parquet")
    
    extract_features(weekly_full, weekly_future)