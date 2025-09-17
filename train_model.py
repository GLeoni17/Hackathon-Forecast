import pandas as pd
import lightgbm as lgb
from xgboost import XGBRegressor
import numpy as np
from joblib import dump
import json
#from predict import wmape

def wmape(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    denom = y_true.sum()
    if denom == 0:
        return np.nan
    return (np.abs(y_true - y_pred).sum() / denom) * 100.0  # em porcentagem

if __name__=='__main__':
    weekly_train = pd.read_parquet("/home/luanguerra/Teste/data/weekly_train.parquet")
    weekly_test = pd.read_parquet("/home/luanguerra/Teste/data/weekly_test.parquet")
    
    # escolher data de corte para validação (ajuste conforme disponibilidade)
    cutoff = pd.Timestamp("2022-11-27")  # treinamento até aqui
    train = weekly_train[weekly_train['week_start'] <= cutoff].copy()
    val   = weekly_train[(weekly_train['week_start'] > cutoff) & (weekly_train['week_start'] <= cutoff + pd.Timedelta(weeks=5))].copy()
    #test = weekly_test[weekly_test['week_start']].copy()
    # selecionar features
    features = [c for c in weekly_train.columns if c.startswith('lag_') or c.startswith('roll_') or c in ['month','weekofyear']]
    
    print(features)

    train = train.dropna(subset=features + ['quantity'])
    val = val.dropna(subset=features + ['quantity'])
    test = weekly_test.dropna(subset=features + ['quantity'])
    print(test.shape)
    
    X_train = train[features].fillna(0.0).values 
    y_train = train['quantity'].values
    X_val   = val[features].fillna(0.0).values
    y_val   = val['quantity'].values
    
    X_test = test[features].fillna(0.0).values


    
    xgb_model = XGBRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    early_stopping_rounds=50,
    tree_method='hist'
    )

    xgb_model.fit(X_train, y_train,
                eval_set=[(X_train, y_train),(X_val, y_val)],
                verbose=50)
    
    xgb_model.save_model('/home/luanguerra/Teste/model_weights/xgb.json')
    
    #dump(xgb_model, '/home/luanguerra/Teste/model_weights/xgb_sklearn.joblib')

    meta = {'features': features, 'best_iteration': int(xgb_model.get_booster().best_iteration)}
    with open('/home/luanguerra/Teste/model_weights/xgb_meta_wrapper.json','w',encoding='utf8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
        
    preds_test = xgb_model.predict(X_test)
    preds_test = np.clip(preds_test, 0, None)
    print(preds_test)
    #print("WMAPE (val):", wmape(y_val, preds_val))
    
    df_preds = test.copy()
    
    df_preds['quantity'] = preds_test
    
    df_preds[['week_start', 'pdv_id', 'produto_id', 'quantity']].to_csv(
    "/home/luanguerra/Teste/data/predicoes_jan_2023.csv", index=False
    )





    #LGB
    """dtrain = lgb.Dataset(train[features], label=train['quantity'])
    dval   = lgb.Dataset(val[features], label=val['quantity'])
    params = {
        'objective':'regression',
        'metric':'rmse',
        'learning_rate':0.05,
        'num_leaves':20,
        'min_data_in_leaf':20,
        'verbosity': -1
    }
    bst = lgb.train(params, dtrain, valid_sets=[dtrain,dval], num_boost_round=2000)
    
    bst.save_model('/home/luanguerra/Teste/model_weights/lgb_booster.txt')  """