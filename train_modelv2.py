import pandas as pd
import lightgbm as lgb
from xgboost import XGBRegressor
from sklearn.ensemble import (
    RandomForestRegressor, 
    GradientBoostingRegressor, 
    HistGradientBoostingRegressor,
    AdaBoostRegressor,
    ExtraTreesRegressor
)
from sklearn.linear_model import (
    Ridge, 
    Lasso, 
    ElasticNet,
    BayesianRidge
)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.dummy import DummyRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor
import numpy as np
from joblib import dump, load
import json
import time
import warnings
warnings.filterwarnings('ignore')

def wmape(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    denom = y_true.sum()
    if denom == 0:
        return np.nan
    return (np.abs(y_true - y_pred).sum() / denom) * 100.0

def train_models(X_train, y_train, X_val, y_val, features):
    """Treina múltiplos modelos e retorna resultados"""
    models = {}
    results = {}
    
    # 1. XGBoost
    print("Treinando XGBoost...")
    start_time = time.time()
    
    xgb_model = XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        early_stopping_rounds=50,
        tree_method='hist',
        n_jobs=-1
    )

    xgb_model.fit(X_train, y_train,
                 eval_set=[(X_train, y_train), (X_val, y_val)],
                 verbose=0)
    
    preds_xgb = xgb_model.predict(X_val)
    preds_xgb = np.clip(preds_xgb, 0, None)
    wmape_xgb = wmape(y_val, preds_xgb)
    
    models['xgb'] = xgb_model
    results['xgb'] = {'wmape': wmape_xgb, 'time': time.time() - start_time}
    print(f"XGBoost - WMAPE: {wmape_xgb:.2f}%, Tempo: {results['xgb']['time']:.1f}s")
    
    # 2. LightGBM (CORRIGIDO)
    print("Treinando LightGBM...")
    start_time = time.time()
    
    # Formato correto para LightGBM
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'learning_rate': 0.05,
        'max_depth': 8,
        'num_leaves': 31,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'verbosity': -1,
        'n_jobs': -1
    }
    
    lgb_model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[train_data, val_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=0),
            lgb.log_evaluation(period=100)
        ]
    )
    
    preds_lgb = lgb_model.predict(X_val)
    preds_lgb = np.clip(preds_lgb, 0, None)
    wmape_lgb = wmape(y_val, preds_lgb)
    
    models['lgb'] = lgb_model
    results['lgb'] = {'wmape': wmape_lgb, 'time': time.time() - start_time}
    print(f"LightGBM - WMAPE: {wmape_lgb:.2f}%, Tempo: {results['lgb']['time']:.1f}s")
    
    # 3. Random Forest
    print("Treinando Random Forest...")
    start_time = time.time()
    
    sample_size = min(50000, len(X_train))
    sample_idx = np.random.choice(len(X_train), sample_size, replace=False)
    
    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        n_jobs=-1,
        random_state=42
    )
    
    rf_model.fit(X_train[sample_idx], y_train[sample_idx])
    preds_rf = rf_model.predict(X_val)
    preds_rf = np.clip(preds_rf, 0, None)
    wmape_rf = wmape(y_val, preds_rf)
    
    models['rf'] = rf_model
    results['rf'] = {'wmape': wmape_rf, 'time': time.time() - start_time}
    print(f"Random Forest - WMAPE: {wmape_rf:.2f}%, Tempo: {results['rf']['time']:.1f}s")
    
    # 4. HistGradientBoosting (Mais rápido que GBDT normal)
    print("Treinando HistGradientBoosting...")
    start_time = time.time()
    
    hgb_model = HistGradientBoostingRegressor(
        max_iter=200,
        learning_rate=0.05,
        max_depth=8,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20
    )
    
    hgb_model.fit(X_train, y_train)
    preds_hgb = hgb_model.predict(X_val)
    preds_hgb = np.clip(preds_hgb, 0, None)
    wmape_hgb = wmape(y_val, preds_hgb)
    
    models['hgb'] = hgb_model
    results['hgb'] = {'wmape': wmape_hgb, 'time': time.time() - start_time}
    print(f"HistGradientBoosting - WMAPE: {wmape_hgb:.2f}%, Tempo: {results['hgb']['time']:.1f}s")
    
    # 5. Ridge Regression
    print("Treinando Ridge Regression...")
    start_time = time.time()
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    ridge_model = Ridge(alpha=1.0, random_state=42)
    ridge_model.fit(X_train_scaled, y_train)
    preds_ridge = ridge_model.predict(X_val_scaled)
    preds_ridge = np.clip(preds_ridge, 0, None)
    wmape_ridge = wmape(y_val, preds_ridge)
    
    models['ridge'] = ridge_model
    models['ridge_scaler'] = scaler
    results['ridge'] = {'wmape': wmape_ridge, 'time': time.time() - start_time}
    print(f"Ridge - WMAPE: {wmape_ridge:.2f}%, Tempo: {results['ridge']['time']:.1f}s")
    
    # 6. ElasticNet
    print("Treinando ElasticNet...")
    start_time = time.time()
    
    enet_model = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
    enet_model.fit(X_train_scaled, y_train)
    preds_enet = enet_model.predict(X_val_scaled)
    preds_enet = np.clip(preds_enet, 0, None)
    wmape_enet = wmape(y_val, preds_enet)
    
    models['enet'] = enet_model
    results['enet'] = {'wmape': wmape_enet, 'time': time.time() - start_time}
    print(f"ElasticNet - WMAPE: {wmape_enet:.2f}%, Tempo: {results['enet']['time']:.1f}s")
    
    # 7. MLP (Neural Network)
    print("Treinando MLP...")
    start_time = time.time()
    
    # Amostrar para ser mais rápido
    sample_size_mlp = min(50000, len(X_train_scaled))
    sample_idx_mlp = np.random.choice(len(X_train_scaled), sample_size_mlp, replace=False)
    
    mlp_model = MLPRegressor(
        hidden_layer_sizes=(100, 50),
        activation='relu',
        solver='adam',
        alpha=0.001,
        learning_rate_init=0.001,
        max_iter=200,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1
    )
    
    mlp_model.fit(X_train_scaled[sample_idx_mlp], y_train[sample_idx_mlp])
    preds_mlp = mlp_model.predict(X_val_scaled)
    preds_mlp = np.clip(preds_mlp, 0, None)
    wmape_mlp = wmape(y_val, preds_mlp)
    
    models['mlp'] = mlp_model
    results['mlp'] = {'wmape': wmape_mlp, 'time': time.time() - start_time}
    print(f"MLP - WMAPE: {wmape_mlp:.2f}%, Tempo: {results['mlp']['time']:.1f}s")

    # 8. CatBoost
    print("Treinando CatBoost...")
    start_time = time.time()

    from catboost import CatBoostRegressor

    cat_model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.05,
        depth=8,
        random_seed=42,
        verbose=0,
        early_stopping_rounds=50
    )

    cat_model.fit(X_train, y_train, eval_set=(X_val, y_val))
    preds_cat = cat_model.predict(X_val)
    preds_cat = np.clip(preds_cat, 0, None)
    wmape_cat = wmape(y_val, preds_cat)

    models['catboost'] = cat_model
    results['catboost'] = {'wmape': wmape_cat, 'time': time.time() - start_time}
    print(f"CatBoost - WMAPE: {wmape_cat:.2f}%, Tempo: {results['catboost']['time']:.1f}s")

    # 9. Support Vector Regression
    print("Treinando SVR...")
    start_time = time.time()

    from sklearn.svm import SVR

    # Amostrar para SVR ser mais rápido
    sample_size_svr = min(20000, len(X_train_scaled))
    sample_idx_svr = np.random.choice(len(X_train_scaled), sample_size_svr, replace=False)

    svr_model = SVR(
        kernel='rbf',
        C=1.0,
        epsilon=0.1
    )

    svr_model.fit(X_train_scaled[sample_idx_svr], y_train[sample_idx_svr])
    preds_svr = svr_model.predict(X_val_scaled)
    preds_svr = np.clip(preds_svr, 0, None)
    wmape_svr = wmape(y_val, preds_svr)

    models['svr'] = svr_model
    results['svr'] = {'wmape': wmape_svr, 'time': time.time() - start_time}
    print(f"SVR - WMAPE: {wmape_svr:.2f}%, Tempo: {results['svr']['time']:.1f}s")

    # 10. K-Nearest Neighbors
    print("Treinando KNN...")
    start_time = time.time()

    from sklearn.neighbors import KNeighborsRegressor

    # Amostrar para KNN ser mais rápido
    sample_size_knn = min(30000, len(X_train_scaled))
    sample_idx_knn = np.random.choice(len(X_train_scaled), sample_size_knn, replace=False)

    knn_model = KNeighborsRegressor(
        n_neighbors=5,
        weights='distance',
        n_jobs=-1
    )

    knn_model.fit(X_train_scaled[sample_idx_knn], y_train[sample_idx_knn])
    preds_knn = knn_model.predict(X_val_scaled)
    preds_knn = np.clip(preds_knn, 0, None)
    wmape_knn = wmape(y_val, preds_knn)

    models['knn'] = knn_model
    results['knn'] = {'wmape': wmape_knn, 'time': time.time() - start_time}
    print(f"KNN - WMAPE: {wmape_knn:.2f}%, Tempo: {results['knn']['time']:.1f}s")

    # 11. Decision Tree
    print("Treinando Decision Tree...")
    start_time = time.time()

    from sklearn.tree import DecisionTreeRegressor

    dt_model = DecisionTreeRegressor(
        max_depth=10,
        min_samples_split=5,
        random_state=42
    )

    dt_model.fit(X_train, y_train)
    preds_dt = dt_model.predict(X_val)
    preds_dt = np.clip(preds_dt, 0, None)
    wmape_dt = wmape(y_val, preds_dt)

    models['decision_tree'] = dt_model
    results['decision_tree'] = {'wmape': wmape_dt, 'time': time.time() - start_time}
    print(f"Decision Tree - WMAPE: {wmape_dt:.2f}%, Tempo: {results['decision_tree']['time']:.1f}s")

    # 12. AdaBoost
    print("Treinando AdaBoost...")
    start_time = time.time()

    from sklearn.ensemble import AdaBoostRegressor

    ada_model = AdaBoostRegressor(
        n_estimators=100,
        learning_rate=0.1,
        random_state=42
    )

    ada_model.fit(X_train, y_train)
    preds_ada = ada_model.predict(X_val)
    preds_ada = np.clip(preds_ada, 0, None)
    wmape_ada = wmape(y_val, preds_ada)

    models['adaboost'] = ada_model
    results['adaboost'] = {'wmape': wmape_ada, 'time': time.time() - start_time}
    print(f"AdaBoost - WMAPE: {wmape_ada:.2f}%, Tempo: {results['adaboost']['time']:.1f}s")

    # 13. Extra Trees
    print("Treinando Extra Trees...")
    start_time = time.time()

    from sklearn.ensemble import ExtraTreesRegressor

    et_model = ExtraTreesRegressor(
        n_estimators=100,
        max_depth=10,
        n_jobs=-1,
        random_state=42
    )

    et_model.fit(X_train, y_train)
    preds_et = et_model.predict(X_val)
    preds_et = np.clip(preds_et, 0, None)
    wmape_et = wmape(y_val, preds_et)

    models['extra_trees'] = et_model
    results['extra_trees'] = {'wmape': wmape_et, 'time': time.time() - start_time}
    print(f"Extra Trees - WMAPE: {wmape_et:.2f}%, Tempo: {results['extra_trees']['time']:.1f}s")

    # 14. Bayesian Ridge
    print("Treinando Bayesian Ridge...")
    start_time = time.time()

    from sklearn.linear_model import BayesianRidge

    br_model = BayesianRidge()

    br_model.fit(X_train_scaled, y_train)
    preds_br = br_model.predict(X_val_scaled)
    preds_br = np.clip(preds_br, 0, None)
    wmape_br = wmape(y_val, preds_br)

    models['bayesian_ridge'] = br_model
    results['bayesian_ridge'] = {'wmape': wmape_br, 'time': time.time() - start_time}
    print(f"Bayesian Ridge - WMAPE: {wmape_br:.2f}%, Tempo: {results['bayesian_ridge']['time']:.1f}s")

    # 15. Dummy Regressor (Baseline)
    print("Treinando Dummy Regressor...")
    start_time = time.time()

    from sklearn.dummy import DummyRegressor

    dummy_model = DummyRegressor(strategy='mean')
    dummy_model.fit(X_train, y_train)
    preds_dummy = dummy_model.predict(X_val)
    preds_dummy = np.clip(preds_dummy, 0, None)
    wmape_dummy = wmape(y_val, preds_dummy)

    models['dummy'] = dummy_model
    results['dummy'] = {'wmape': wmape_dummy, 'time': time.time() - start_time}
    print(f"Dummy - WMAPE: {wmape_dummy:.2f}%, Tempo: {results['dummy']['time']:.1f}s")
            
    return models, results

def save_models(models, results, features):
    """Salva os modelos e resultados"""
    
    # Salvar metadata
    meta = {
        'features': features,
        'results': results,
        'best_model': min(results.items(), key=lambda x: x[1]['wmape'])[0],
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open('./model_weights/all_models_meta.json', 'w', encoding='utf8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    
    # Salvar modelos
    for name, model in models.items():
        if name == 'xgb':
            model.save_model(f'./model_weights/{name}_model.json')
        elif name == 'lgb':
            model.save_model(f'./model_weights/{name}_model.txt')
        elif name.endswith('_scaler'):
            continue  # Skip scalers
        else:
            dump(model, f'./model_weights/{name}_model.joblib')
    
    print("Modelos salvos em './model_weights/'")

if __name__=='__main__':
    # Carregar dados
    weekly_train = pd.read_parquet("./data_parquet/weekly_train.parquet")
    weekly_test = pd.read_parquet("./data_parquet/weekly_test.parquet")
    
    cutoff = pd.Timestamp("2022-11-27")
    train = weekly_train[weekly_train['week_start'] <= cutoff].copy()
    val   = weekly_train[(weekly_train['week_start'] > cutoff) & 
                        (weekly_train['week_start'] <= cutoff + pd.Timedelta(weeks=5))].copy()
    
    features = [c for c in weekly_train.columns if c.startswith('lag_') or 
               c.startswith('roll_') or c in ['month','week_of_year','year']]
    
    print(f"Features: {len(features)}")
    print(f"Train: {len(train)}, Val: {len(val)}")
    
    # Limpar dados
    train = train.dropna(subset=features + ['quantity'])
    val = val.dropna(subset=features + ['quantity'])
    test = weekly_test.dropna(subset=features + ['quantity'])
    
    # Preparar dados
    X_train = train[features].fillna(0.0).values 
    y_train = train['quantity'].values
    X_val   = val[features].fillna(0.0).values
    y_val   = val['quantity'].values
    
    # Treinar modelos
    print("\n=== TREINANDO 7 MODELOS DIFERENTES ===")
    models, results = train_models(X_train, y_train, X_val, y_val, features)
    
    # Resultados
    print("\n=== RESULTADOS FINAIS ===")
    print(f"{'Modelo':20} {'WMAPE':>8} {'Tempo(s)':>10}")
    print("-" * 40)
    
    for model_name, result in sorted(results.items(), key=lambda x: x[1]['wmape']):
        print(f"{model_name:20} {result['wmape']:8.2f}% {result['time']:10.1f}s")
    
    best_model_name, best_result = min(results.items(), key=lambda x: x[1]['wmape'])
    print(f"\n🎯 MELHOR MODELO: {best_model_name} - WMAPE: {best_result['wmape']:.2f}%")
    
    # Salvar modelos
    save_models(models, results, features)
    
    # Fazer previsões no teste com o melhor modelo
    X_test = test[features].fillna(0.0).values
    best_model = models[best_model_name]
    
    if best_model_name in ['ridge', 'enet', 'mlp']:
        scaler = models['ridge_scaler']
        X_test_scaled = scaler.transform(X_test)
        test_preds = best_model.predict(X_test_scaled)
    else:
        test_preds = best_model.predict(X_test)
    
    test_preds = np.clip(test_preds, 0, None)
    
    # Salvar previsões
    test['quantity_pred'] = test_preds
    test[['week_start', 'pdv_id', 'produto_id', 'quantity_pred']].to_csv(
        "./data_parquet/predicoes_teste.csv", index=False
    )
    
    print("Previsões salvas!")