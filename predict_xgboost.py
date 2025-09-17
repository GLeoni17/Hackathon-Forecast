# Validação temporal: usar últimas 5 semanas como teste e prever recursivamente com LightGBM
import pandas as pd
import numpy as np
import os
import json
import lightgbm as lgb
import xgboost as xgb
from joblib import load

# ---------------- CONFIGURAR AQUI ----------------
WEEKLY_PARQUET = "weekly_full_features.parquet"   # seu arquivo com histórico semanal
MODEL_DIR  = "/home/luanguerra/Teste/model_weights"              # onde estão xgb_meta.json e o modelo
MODEL_BOOSTER = os.path.join(MODEL_DIR, "xgb.json")     # se você salvou com bst.save_model(...)
MODEL_SKLEARN = os.path.join(MODEL_DIR, "xgb_sklearn.joblib")
OUT_PRED_CSV   = "val_predictions_5w.csv"
LAGS           = [1,2,3,4,13,26,52]      # ajuste conforme o que você usou no treino
ROLL_WINDOWS   = [4,8,13]                # ajuste conforme treino
FEATURES = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_13', 'roll_mean_4', 'roll_mean_8']
# -------------------------------------------------

def wmape(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    denom = y_true.sum()
    if denom == 0:
        return np.nan
    return (np.abs(y_true - y_pred).sum() / denom) * 100.0  # em porcentagem

# Função vetorizada que cria lags e rollings para uma semana alvo
def build_features_for_target_week_vectorized(hist, pairs, target_week, lags=LAGS, roll_windows=ROLL_WINDOWS):
    """
    hist: DataFrame com ['pdv_id','produto_id','week_start','quantity'] contendo histórico até a semana anterior ao target_week
    pairs: DataFrame com ['pdv_id','produto_id'] (pares a prever)
    target_week: pd.Timestamp (semana que queremos prever)
    Retorna: DataFrame (pairs + week_start + lag_*/roll_* cols)
    """
    target_week = pd.to_datetime(target_week)
    Xw = pairs.copy()
    Xw['pdv_id'] = Xw['pdv_id'].astype(str)
    Xw['produto_id'] = Xw['produto_id'].astype(str)
    Xw['week_start'] = target_week

    # lags via merge com snapshot de cada semana target - l
    for l in lags:
        wk = (target_week - pd.Timedelta(weeks=l))
        snap = hist[hist['week_start'] == wk][['pdv_id','produto_id','quantity']].rename(columns={'quantity': f'lag_{l}'})
        Xw = Xw.merge(snap, on=['pdv_id','produto_id'], how='left')

    # rollings: agregar nas semanas target-1 ... target-w
    for w in roll_windows:
        weeks = [(target_week - pd.Timedelta(weeks=k)) for k in range(1, w+1)]
        window_df = hist[hist['week_start'].isin(weeks)]
        agg = (window_df.groupby(['pdv_id','produto_id'])['quantity']
               .agg([('roll_mean_'+str(w),'mean'), ('roll_std_'+str(w),'std')])
               .reset_index())
        Xw = Xw.merge(agg, on=['pdv_id','produto_id'], how='left')

    return Xw

# ------------------ Pipeline principal ------------------
# 1) carregar dados
weekly_full = pd.read_parquet('/home/luanguerra/Teste/data/weekly_full.parquet')
weekly_full['week_start'] = pd.to_datetime(weekly_full['week_start'])

# garantir tipos
weekly_full['pdv_id'] = weekly_full['pdv_id'].astype(str)
weekly_full['produto_id'] = weekly_full['produto_id'].astype(str)

# 2) determinar semanas de teste (últimas 5 semanas do dataset)
all_weeks = sorted(weekly_full['week_start'].unique())

test_weeks = all_weeks[-5:]    # lista ordenada das 5 últimas semanas
test_start = test_weeks[0]
test_end   = test_weeks[-1]

# 3) histórico observável para treino = tudo até a semana anterior ao test_start
train_end = (test_start - pd.Timedelta(weeks=1))
hist_for_training = weekly_full[weekly_full['week_start'] <= train_end].copy()
print("Histórico para treino até:", train_end.date(), "| semanas disponíveis (treino):", hist_for_training['week_start'].nunique())

# 4) pares pdv x produto a serem previstos (use os pares que existem no histórico total ou apenas no treino)
# usar pares observados no histórico total (ou ajuste para pares específicos)
pairs = weekly_full[['pdv_id','produto_id']].drop_duplicates().reset_index(drop=True)

# 5) carregar modelo e features
model = xgb.Booster()
model.load_model(MODEL_BOOSTER)
#model = lgb.Booster(model_file='/home/luanguerra/Teste/model_weights/lgb_booster.txt')

# 6) predição recursiva vetorizada para as 5 semanas de teste
hist = hist_for_training.copy()  # será atualizado com predições iterativas
preds_list = []

for i, tgt in enumerate(test_weeks):
    print(f"Previsão recursiva - step {i+1} / {len(test_weeks)} -> semana {tgt.date()}")
    # construir features para a semana alvo usando somente hist (que contém até a semana anterior)
    Xw = build_features_for_target_week_vectorized(hist, pairs, tgt, lags=LAGS, roll_windows=ROLL_WINDOWS)
    
    # garantir todas features que o modelo espera existam (criar com 0 se ausente)
    missing = [f for f in FEATURES if f not in Xw.columns]
    if missing:
        # preencher com 0 (ajuste se você tiver encoders salvos — aplique-os antes)
        for f in missing:
            Xw[f] = 0.0

    # preparar matriz de entrada respetando a ordem FEATURES
    X_pred = Xw[FEATURES].copy()
    # preencher NaNs: LightGBM aceita NaN, mas para consistência aqui substituímos por 0 (você pode optar por deixar NaN)
    X_pred = X_pred.fillna(0.0)

    
    dmat = xgb.DMatrix(X_pred.values, feature_names=FEATURES)
    best_ntree = int(model.attributes().get('best_iteration')) if model.attributes().get('best_iteration') else None

    if best_ntree:
        yhat = model.predict(dmat)
    else:
        yhat = model.predict(dmat)
    
    print(yhat)

                
    # pós-processar: clip negativo a zero e arredondar
    yhat = np.clip(yhat, 0, None)
    yhat = np.round(yhat).astype(int)

    Xw['forecast_qty'] = yhat
    preds_step = Xw[['week_start','pdv_id','produto_id','forecast_qty']].copy()
    preds_list.append(preds_step)

    # inserir predições no histórico como observadas para o próximo passo (recursão)
    new_hist = preds_step.rename(columns={'forecast_qty':'quantity'})
    hist = pd.concat([hist, new_hist[['pdv_id','produto_id','week_start','quantity']]], ignore_index=True)
    # manter ordenação opcional
    hist = hist.sort_values(['pdv_id','produto_id','week_start']).reset_index(drop=True)

# juntar previsões das 5 semanas
preds_all = pd.concat(preds_list, ignore_index=True)
preds_all = preds_all.sort_values(['pdv_id','produto_id','week_start']).reset_index(drop=True)

# 7) obter ground-truth para as mesmas semanas e calcular WMAPE
truth = weekly_full[(weekly_full['week_start'] >= test_start) & (weekly_full['week_start'] <= test_end)][['week_start','pdv_id','produto_id','quantity']]

# juntar (left join preds -> truth). Se faltar truth considerar 0 (ajuste conforme sua competição)
merged = preds_all.merge(truth, on=['week_start','pdv_id','produto_id'], how='left')
merged['quantity'] = merged['quantity'].fillna(0).astype(float)
merged['forecast_qty'] = merged['forecast_qty'].astype(float)

# WMAPE global
score = wmape(merged['quantity'], merged['forecast_qty'])
print(f"WMAPE (últimas 5 semanas) = {score:.4f}%")

# também reportar por semana e média
wmape_per_week = merged.groupby('week_start').apply(lambda g: wmape(g['quantity'], g['forecast_qty']))
print("WMAPE por semana:")
print(wmape_per_week)
