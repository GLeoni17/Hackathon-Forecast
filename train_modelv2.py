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
# Adicione esses imports
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.multioutput import MultiOutputRegressor
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from pycaret.time_series import *
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

    '''
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
    '''
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

    '''
    # 16. LSTM
    print("Treinando LSTM...")
    start_time = time.time()

    # Preparar dados para LSTM (3D: [samples, timesteps, features])
    def create_sequences(X, y, time_steps=10):
        Xs, ys = [], []
        for i in range(len(X) - time_steps):
            Xs.append(X[i:(i + time_steps)])
            ys.append(y[i + time_steps])
        return np.array(Xs), np.array(ys)

    # Criar sequências temporais
    time_steps = 8
    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train, time_steps)
    X_val_seq, y_val_seq = create_sequences(X_val_scaled, y_val, time_steps)

    # Modelo LSTM
    lstm_model = Sequential([
        LSTM(64, activation='relu', input_shape=(time_steps, X_train_scaled.shape[1]), return_sequences=True),
        Dropout(0.2),
        LSTM(32, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])

    lstm_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')



    # Treinar
    history = lstm_model.fit(
        X_train_seq, y_train_seq,
        epochs=50,
        batch_size=32,
        validation_data=(X_val_seq, y_val_seq),
        verbose=0,
        callbacks=[EarlyStopping(patience=10, restore_best_weights=True)]
    )

    # Prever
    preds_lstm = lstm_model.predict(X_val_seq, verbose=0).flatten()
    preds_lstm = np.clip(preds_lstm, 0, None)
    wmape_lstm = wmape(y_val[time_steps:], preds_lstm)

    models['lstm'] = lstm_model
    results['lstm'] = {'wmape': wmape_lstm, 'time': time.time() - start_time}
    print(f"LSTM - WMAPE: {wmape_lstm:.2f}%, Tempo: {results['lstm']['time']:.1f}s")
    '''
    # 17. CNN 1D
    print("Treinando CNN 1D...")
    start_time = time.time()

    cnn_model = Sequential([
        Conv1D(64, kernel_size=3, activation='relu', input_shape=(time_steps, X_train_scaled.shape[1])),
        MaxPooling1D(pool_size=2),
        Conv1D(32, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(1)
    ])

    cnn_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

    history = cnn_model.fit(
        X_train_seq, y_train_seq,
        epochs=50,
        batch_size=32,
        validation_data=(X_val_seq, y_val_seq),
        verbose=0,
        callbacks=[EarlyStopping(patience=10, restore_best_weights=True)]
    )

    preds_cnn = cnn_model.predict(X_val_seq, verbose=0).flatten()
    preds_cnn = np.clip(preds_cnn, 0, None)
    wmape_cnn = wmape(y_val[time_steps:], preds_cnn)

    models['cnn'] = cnn_model
    results['cnn'] = {'wmape': wmape_cnn, 'time': time.time() - start_time}
    print(f"CNN 1D - WMAPE: {wmape_cnn:.2f}%, Tempo: {results['cnn']['time']:.1f}s")

    # 18. MLP Profundo
    print("Treinando MLP Profundo...")
    start_time = time.time()

    deep_mlp_model = MLPRegressor(
        hidden_layer_sizes=(256, 128, 64, 32),
        activation='relu',
        solver='adam',
        alpha=0.001,
        learning_rate_init=0.0001,
        max_iter=300,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1
    )

    deep_mlp_model.fit(X_train_scaled, y_train)
    preds_deep_mlp = deep_mlp_model.predict(X_val_scaled)
    preds_deep_mlp = np.clip(preds_deep_mlp, 0, None)
    wmape_deep_mlp = wmape(y_val, preds_deep_mlp)

    models['deep_mlp'] = deep_mlp_model
    results['deep_mlp'] = {'wmape': wmape_deep_mlp, 'time': time.time() - start_time}
    print(f"Deep MLP - WMAPE: {wmape_deep_mlp:.2f}%, Tempo: {results['deep_mlp']['time']:.1f}s")

    # 19. ARIMA
    print("Treinando ARIMA...")
    start_time = time.time()

    # Para séries temporais univariadas (usando a média por semana)
    try:
        time_series = train.groupby('week_start')['quantity'].mean().values
        
        arima_model = ARIMA(time_series, order=(2,1,2))
        arima_fit = arima_model.fit()
        
        # Prever para o período de validação
        preds_arima = arima_fit.forecast(steps=len(val['week_start'].unique()))
        preds_arima = np.clip(preds_arima, 0, None)
        
        # Mapear para todas as linhas (simplificado)
        wmape_arima = wmape(val['quantity'], np.repeat(preds_arima.mean(), len(val)))
        
        models['arima'] = arima_fit
        results['arima'] = {'wmape': wmape_arima, 'time': time.time() - start_time}
        print(f"ARIMA - WMAPE: {wmape_arima:.2f}%, Tempo: {results['arima']['time']:.1f}s")
        
    except Exception as e:
        print(f"ARIMA falhou: {e}")
        results['arima'] = {'wmape': np.nan, 'time': time.time() - start_time}

    # 20. Exponential Smoothing
    print("Treinando Exponential Smoothing...")
    start_time = time.time()

    try:
        es_model = ExponentialSmoothing(
            time_series,
            seasonal_periods=52,
            trend='add',
            seasonal='add'
        )
        es_fit = es_model.fit()
        
        preds_es = es_fit.forecast(steps=len(val['week_start'].unique()))
        preds_es = np.clip(preds_es, 0, None)
        
        wmape_es = wmape(val['quantity'], np.repeat(preds_es.mean(), len(val)))
        
        models['exp_smoothing'] = es_fit
        results['exp_smoothing'] = {'wmape': wmape_es, 'time': time.time() - start_time}
        print(f"Exp Smoothing - WMAPE: {wmape_es:.2f}%, Tempo: {results['exp_smoothing']['time']:.1f}s")
        
    except Exception as e:
        print(f"Exponential Smoothing falhou: {e}")
        results['exp_smoothing'] = {'wmape': np.nan, 'time': time.time() - start_time}

    # 21. Stacking Ensemble
    print("Treinando Stacking Ensemble...")
    start_time = time.time()

    from sklearn.ensemble import StackingRegressor

    # Definir base learners
    base_learners = [
        ('ridge', Ridge()),
        ('rf', RandomForestRegressor(n_estimators=50, random_state=42)),
        ('xgb', XGBRegressor(n_estimators=100, random_state=42))
    ]

    stacking_model = StackingRegressor(
        estimators=base_learners,
        final_estimator=GradientBoostingRegressor(random_state=42),
        n_jobs=-1
    )

    stacking_model.fit(X_train_scaled, y_train)
    preds_stacking = stacking_model.predict(X_val_scaled)
    preds_stacking = np.clip(preds_stacking, 0, None)
    wmape_stacking = wmape(y_val, preds_stacking)

    models['stacking'] = stacking_model
    results['stacking'] = {'wmape': wmape_stacking, 'time': time.time() - start_time}
    print(f"Stacking - WMAPE: {wmape_stacking:.2f}%, Tempo: {results['stacking']['time']:.1f}s")

    # 22. Voting Ensemble
    print("Treinando Voting Ensemble...")
    start_time = time.time()

    from sklearn.ensemble import VotingRegressor

    voting_model = VotingRegressor([
        ('ridge', Ridge()),
        ('xgb', XGBRegressor(n_estimators=100, random_state=42)),
        ('lgb', lgb.LGBMRegressor(n_estimators=100, random_state=42))
    ])

    voting_model.fit(X_train_scaled, y_train)
    preds_voting = voting_model.predict(X_val_scaled)
    preds_voting = np.clip(preds_voting, 0, None)
    wmape_voting = wmape(y_val, preds_voting)

    models['voting'] = voting_model
    results['voting'] = {'wmape': wmape_voting, 'time': time.time() - start_time}
    print(f"Voting - WMAPE: {wmape_voting:.2f}%, Tempo: {results['voting']['time']:.1f}s")

    
            
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
    print("\n=== TREINANDO MODELOS DIFERENTES ===")
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