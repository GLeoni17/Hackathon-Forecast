import pandas as pd
import os
from joblib import Parallel, delayed
from tqdm import tqdm

PATH_WEEKLY = './data_parquet/weekly.parquet'
PATH_WEEKLY_FULL  = './data_parquet/weekly_full.parquet'

PATH_PDV = './data_parquet/part-00000-tid-2779033056155408584-f6316110-4c9a-4061-ae48-69b77c7c8c36-4-1-c000.snappy.parquet' # /home/luanguerra/Teste/data/pdv.parquet
PATH_TRANSACAO = './data_parquet/part-00000-tid-5196563791502273604-c90d3a24-52f2-4955-b4ec-fb143aae74d8-4-1-c000.snappy.parquet' # /home/luanguerra/Teste/data/transacao.parquet
PATH_PRODUTO = './data_parquet/part-00000-tid-7173294866425216458-eae53fbf-d19e-4130-ba74-78f96b9675f1-4-1-c000.snappy.parquet' # /home/luanguerra/Teste/data/produto.parquet

BATCH_SIZE = 1000

def process_batch(batch, weekly_df):
    merged_batch = batch.merge(weekly_df, on=['pdv_id', 'produto_id', 'week_start'], how='left')
    filled_batch = merged_batch.fillna({'quantity': 0, 'gross_value': 0, 'net_value': 0, 'transactions': 0})
    return filled_batch

def create_weekly():
    # Arquivos já criados, já passou pela função
    if os.path.isfile(PATH_WEEKLY) and os.path.isfile(PATH_WEEKLY_FULL): return

    # Leitura do dataset
    print('Leitura do dataset...')
    pdv = pd.read_parquet(PATH_PDV)
    transacao = pd.read_parquet(PATH_TRANSACAO)
    produto = pd.read_parquet(PATH_PRODUTO)

    # Pré Processamento dos arquivos
    print('Pré Processamento dos arquivos...')
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

    # Merge das tabelas
    print('Merge das tabelas...')
    transacao = transacao.merge(pdv, left_on='pdv_id', right_on='pdv_id', how='left')
    transacao = transacao.merge(produto, left_on='produto_id', right_on='produto_id', how='left')

    # Limpeza de valores nulos
    print('Limpeza de valores nulos...')
    for c in ['categoria_pdv','premise','zipcode']:
        if c in transacao.columns:
            transacao[c] = transacao[c].fillna('unknown')

    # Separação por semana
    print('Separação por semana...')
    transacao['week_start'] = transacao['transaction_date'] - pd.to_timedelta(transacao['transaction_date'].dt.weekday, unit='d')
    weekly = (transacao.groupby(['week_start','pdv_id','produto_id'], as_index=False)
                .agg(quantity=('quantity','sum'),
                    gross_value=('gross_value','sum'),
                    net_value=('net_value','sum'),
                    transactions=('quantity','size')))
    weekly = pd.DataFrame(weekly)
    weekly.to_parquet(PATH_WEEKLY, index=False, compression='snappy')
    all_weeks = pd.DataFrame({'week_start': pd.date_range("2022-01-03","2022-12-26",freq='7D')})

    # Remoção de duplicatas e merge de todas as semanas
    print('Remoção de duplicatas e merge de todas as semanas...')
    pdv_prod = transacao[['pdv_id','produto_id']].drop_duplicates()
    full_index = pdv_prod.merge(all_weeks, how='cross')
    
    # Carregamento da weekly_full, separado em batches para carregamento na memória, paralelismo para tornar mais rapido
    print('Processando batches semanas...')
    batches_weekly_full = [full_index.iloc[i:i + BATCH_SIZE] for i in range(0, len(full_index), BATCH_SIZE)]
    weekly_full_fragment = Parallel(n_jobs=-1, backend="multiprocessing")(
        delayed(process_batch)(batch, weekly) for batch in tqdm(batches_weekly_full)
    )
    weekly_full = pd.concat(weekly_full_fragment, ignore_index=True)
    
    # Tratamento de colunas
    print('Tratamento final das colunas...')
    for c in ['pdv_id','produto_id','categoria_pdv','categoria','marca','fabricante']:
        if c in weekly_full.columns:
            weekly_full[c] = weekly_full[c].astype('category')      
    weekly_full['quantity'] = pd.to_numeric(weekly_full['quantity'], downcast='integer')
    
    # Armazenamento de todas as semanas em arquivo
    print('Armazenamento em arquivo de todas as semanas...')
    weekly_full = pd.DataFrame(weekly_full)
    weekly_full.to_parquet(PATH_WEEKLY_FULL, index=False, compression='snappy')
    
if __name__=='__main__':
    create_weekly()