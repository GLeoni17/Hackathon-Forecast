import pandas as pd
from pathlib import Path

absolute_path = Path.cwd()

if __name__=='__main__':
    print('Criando weekly_future.parquet com as semanas alvo (5 semanas de janeiro/2023)...') 
    #Carrega o df weekly
    weekly = pd.read_parquet(Path.joinpath(absolute_path,"data/weekly.parquet"))
    
    #Semanas alvo do dataset de teste (5 semanas de janeiro/2023)
    target_weeks = pd.date_range("2023-01-02", periods=5, freq='7D')

    #Pega todos os pares (pdv, sku) do df weekly 
    pairs = weekly[['pdv_id','produto_id']].drop_duplicates().reset_index(drop=True)
    
    #Para cada par, cria a tripla (week_start, pdv, sku) com as 5 semanas alvo
    future_df = (
    pairs.assign(key=1)
         .merge(pd.DataFrame({'week_start': target_weeks, 'key':1}), on="key")
         .drop(columns="key")   
    )
    
    #Merge com weekly para manter as mesmas colunas 
    merged = future_df.merge(
    weekly,
    on=["week_start","pdv_id","produto_id"],
    how="left",   
    suffixes=("", "_hist")
    )

    #Ordena os IDs e 'limpa' a coluna 'quantity', transformando NaNs em 0
    merged = merged.sort_values(['week_start','pdv_id','produto_id']).reset_index(drop=True)
    merged['quantity'] = merged['quantity'].fillna(0)
    
    #Series ID para facilitar o reconhecimento dos pares
    merged['series_id'] = merged['pdv_id'] + '_' + merged['produto_id'] 
    
    print('weekly_future.parquet criado!')
    
    print('Salvando weekly_future.parquet...')
    #Salva o df com as semanas alvo
    merged.to_parquet(Path.joinpath(absolute_path,"data/weekly_future.parquet"), index=False, compression='snappy')
    print('Weekly_future.parquet salvo com sucesso em: ', Path.joinpath(absolute_path,"data/weekly_future.parquet"))