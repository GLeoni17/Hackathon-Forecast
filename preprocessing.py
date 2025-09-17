import pandas as pd

def create_weekly():
    pdv = pd.read_parquet("/home/luanguerra/Teste/data/pdv.parquet")
    transacao = pd.read_parquet("/home/luanguerra/Teste/data/transacao.parquet")
    produto = pd.read_parquet("/home/luanguerra/Teste/data/produto.parquet")
    
    transacao['transaction_date'] = pd.to_datetime(transacao['transaction_date'])
    
    transacao = transacao.rename(columns={
    'internal_store_id':'pdv_id',
    'internal_product_id':'produto_id',
    'quantity':'quantity',
    'gross_value':'gross_value',
    'net_value':'net_value'
    })
    
    pdv = pdv.rename(columns={'pdv':'pdv_id'})
    produto = produto.rename(columns={'produto':'produto_id'})
    
    transacao = transacao.merge(pdv, left_on='pdv_id', right_on='pdv_id', how='left')
    transacao = transacao.merge(produto, left_on='produto_id', right_on='produto_id', how='left')
    
    for c in ['categoria_pdv','premise','zipcode']:
        if c in transacao.columns:
            transacao[c] = transacao[c].fillna('unknown')
    
    #transacao.to_parquet('transacaoV2.parquet', index=False)
    
    transacao['week_start'] = transacao['transaction_date'] - pd.to_timedelta(transacao['transaction_date'].dt.weekday, unit='d')
    
    weekly = (transacao.groupby(['week_start','pdv_id','produto_id'], as_index=False)
                .agg(quantity=('quantity','sum'),
                    gross_value=('gross_value','sum'),
                    net_value=('net_value','sum'),
                    transactions=('quantity','size')))
    
    weekly = pd.DataFrame(weekly)
    weekly.to_parquet('/home/luanguerra/Teste/data/weekly.parquet', index=False)
    
    all_weeks = pd.DataFrame({'week_start': pd.date_range("2022-01-03","2022-12-26",freq='7D')})
    pdv_prod = transacao[['pdv_id','produto_id']].drop_duplicates()
    full_index = pdv_prod.merge(all_weeks, how='cross')
    weekly_full = full_index.merge(weekly, on=['pdv_id','produto_id','week_start'], how='left').fillna({'quantity':0, 'gross_value':0, 'net_value':0, 'transactions':0})
    
    for c in ['pdv_id','produto_id','categoria_pdv','categoria','marca','fabricante']:
        if c in weekly_full.columns:
            weekly_full[c] = weekly_full[c].astype('category')
            
    weekly_full['quantity'] = pd.to_numeric(weekly_full['quantity'], downcast='integer')
    
    weekly_full = pd.DataFrame(weekly_full)
    weekly.to_parquet('/home/luanguerra/Teste/data/weekly_full.parquet', index=False, compression='snappy')
    
if __name__=='__main__':
    #create_weekly()
    df = pd.read_parquet("/home/luanguerra/Teste/data/weekly_train.parquet")
    print(df['week_start'])