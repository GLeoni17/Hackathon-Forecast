import pandas as pd
from xgboost import XGBRegressor
import numpy as np
from pathlib import Path

absolute_path = Path.cwd()

if __name__=='__main__':
    print('Carregando dados de treino e teste...')
    #Carrega os dados de treino e teste
    weekly_train = pd.read_parquet(Path.joinpath(absolute_path,"data/weekly_train.parquet"))
    weekly_test = pd.read_parquet(Path.joinpath(absolute_path,"data/weekly_test.parquet"))
    
    #Selecionando as features extraídas
    features = [c for c in weekly_train.columns if c.startswith('lag_') or c.startswith('roll_') or c in ['month','weekofyear','has_history', 'is_month_start','is_month_end','total_historical_volume']]

    #Limpa as linhas com features NaN a fim de otimizar o treinamento e o teste
    train = weekly_train.dropna(subset=features + ['quantity'])
    test = weekly_test.dropna(subset=features + ['quantity'])
    
    #X_train são as features extraídas e y_train s quantidades a serem preditas
    X_train = train[features].values 
    y_train = train['quantity'].values
    
    #X_teste contém as features do df de teste
    X_test = test[features].values
    
    print('Dados de treino e teste carregados!\n')
    
    print('Carregando modelo XGBoost para regressão...')
    
    #XGBoost para regressão
    xgb_model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.02,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    early_stopping_rounds=50,
    tree_method='hist'
    )
    
    print('Treinando o modelo...')
    #Treinamento do modelo
    xgb_model.fit(X_train, y_train,
                eval_set=[(X_train, y_train)],
                verbose=50)

    
    print('Modelo treinado!\n')
    
    xgb_model.save_model(Path.joinpath(absolute_path,"model_weights/xgb.json"))
    print('Modelo salvo em: ',Path.joinpath(absolute_path,"model_weights/xgb.json"))
    
    print('Calculando predições...')
    
    #Usa o modelo treinado para calcular as predições dos dados de teste
    preds_test = xgb_model.predict(X_test)
    
    print('Predições realizadas!')
    
    
    print('Criando .parquet com as predições...')
    df_preds = test.copy()
    
    df_preds['quantity'] = preds_test
    
    #Processando o df para ter a estrutura [semana, pdv, produto, quantidade]
    df_preds['quantity'] = df_preds['quantity'].round()
    df_preds['week_start'] = pd.to_datetime(df_preds['week_start'])
    
    #Processando as semanas
    df_preds["week_start"] = ((df_preds["week_start"] - pd.Timestamp("2023-01-02")).dt.days // 7) + 1
    
    #Renomeando as colunas
    df_preds = df_preds.rename(columns={
    'week_start':'semana',
    'pdv_id':'pdv',
    'produto_id':'produto',
    'quantity':'quantidade'
    })
    
    #Passando as colunas para int
    df_preds["quantidade"] = df_preds["quantidade"].astype(int)
    df_preds["semana"] = df_preds["semana"].astype(int)
    df_preds["pdv"] = df_preds["pdv"].astype(int)
    df_preds["produto"] = df_preds["produto"].astype(int)
    
    #Salvando o .parquet com as predições
    df_preds.to_parquet(Path.joinpath(absolute_path,'predicts/predicoes.parquet'), index=False)
    
    print('.parquet com as predições alvo em: ', Path.joinpath(absolute_path,'predicts/predicoes.parquet'))
    