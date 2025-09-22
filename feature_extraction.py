import pandas as pd
from pathlib import Path

absolute_path = Path.cwd()

def make_lags(df, lags=[1,2,3,4]):
    """Função que calcula as features Lags para uma dada quantidade de semanas anteriores.
       As features Lags são valores passados da mesma variável, deslocados no tempo. Dessa forma,
       o número da feature determina em quantas semanas atrás eu observo para pegar os valores.
    """
    #Extrai as features Lags
    for l in lags:
        df[f'lag_{l}'] = df.groupby('series_id')['quantity'].shift(l)
    return df

def make_rollings(df, windows=[4,8]):
    """Função que calcula as features Rollings para uma dada quantidade de semanas anteriores.
       As features Rollings são estatísticas calculadas sobre uma janela deslizante de valores 
       passados. Assim como os lags, o número dado define quantas semanas atrás serão consideradas
       para realizar o cálculo das estatísticas (Média, Desvio Padrão, Max, Min, etc.).
    """
    for w in windows:
        df[f'roll_mean_{w}'] = df.groupby('series_id')['quantity'].shift(1).rolling(w).mean().reset_index(0,drop=True)
    return df


def extract_features(weekly_full, weekly_test):
    """Função que extrai as features dos dados de treino e teste.
    """
    
    #Carrega dados de treino e teste
    weekly_full = weekly_full.sort_values(['pdv_id','produto_id','week_start'])
    weekly_test = weekly_test.sort_values(['pdv_id','produto_id','week_start'])
    
    print('Dimensão dos dados de treino antes da extração de features: ', weekly_full.shape)
    print('Dimensão dos dados de teste antes da extração de features :', weekly_test.shape)
    
    #Monta o series_id para o teste
    weekly_full['series_id'] = weekly_full['pdv_id'].astype(str)+'_'+weekly_full['produto_id'].astype(str)
    
    #Concatena os dados de treino e teste para poder calcular as features
    full = pd.concat([weekly_full, weekly_test], ignore_index=True, sort=False)
    
    #Ordena os valores pela semana e series_id
    full = full.sort_values(['series_id','week_start']).reset_index(drop=True)
    
    print('Extraindo as features dos dados...')
    #Calcula as features
    full = make_lags(full)
    full = make_rollings(full)
    
    print('Features extraídas!')
    
    #Define um data para separar os dados em treino e teste novamente
    cutoff = pd.Timestamp("2023-01-02")
    
    weekly_train = full[full['week_start'] < cutoff].copy()
    weekly_test   = full[(full['week_start'] >= cutoff) & (full['week_start'] < cutoff + pd.Timedelta(weeks=5))].copy()
    
    print('Dimensão dos dados de treino com as features extraídas: ', weekly_train.shape)
    print('Dimensão dos dados de teste com as features extraídas :', weekly_test.shape)
    
    print('\nSalvando os dados de treino e teste...')
        
    #Salvando os dados de treino e teste com as features
    weekly_train.to_parquet(Path.joinpath(absolute_path,"data/weekly_train.parquet"), index=False)
    weekly_test.to_parquet(Path.joinpath(absolute_path,"data/weekly_test.parquet"), index=False)
    
    print('Dados salvos!')

if __name__=='__main__':
    
    print('Carregando .parquet de treino e teste...')
    #Carrega os dados de treino e teste para realizar a extração de características
    weekly_full = pd.read_parquet(Path.joinpath(absolute_path,"data/weekly.parquet"))
    weekly_future = pd.read_parquet(Path.joinpath(absolute_path,"data/weekly_future.parquet"))
    
    print('.parquet carregados!\n')
    
    extract_features(weekly_full, weekly_future)