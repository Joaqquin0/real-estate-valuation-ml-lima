# =============================================================================
# 03_ENTRENAMIENTO_ALQUILER.py
# =============================================================================
# Entrenamiento del Modelo XGBoost — Predicción de Precio de Alquiler
# Valoración Inmobiliaria en Lima Metropolitana
#
# CRISP-DM: Fase 3 — Modelado
#
# Dataset de entrada:
#   - train_alquiler.csv (BCRP 2016-2023, partición temporal de entrenamiento)
#   - test_alquiler.csv  (BCRP 2024-2025, partición temporal de evaluación pura)
#   - features_metadata_alquiler.json
#
# Output:
#   - models/xgboost_alquiler_v1.pkl
#   - Métricas en test: MAE, RMSE, MAPE, R²
#   - Gráficos de diagnóstico en reports/figures/alquiler/
# =============================================================================

import pandas as pd
import numpy as np
import json
import os
import time
import joblib
import warnings

from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
)
from xgboost import XGBRegressor

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["figure.dpi"] = 120

# %% 1. Configuración de Rutas
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports", "figures", "alquiler")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

TARGET = "Alquiler_Soles_Const"
TARGET_LOG = f"{TARGET}_log"
RANDOM_SEED = 42

print("=" * 60)
print("ENTRENAMIENTO MODELO XGBOOST — PREDICCIÓN DE ALQUILER")
print("=" * 60)

# %% 2. Carga de Datos
train_path = os.path.join(DATA_DIR, "train_alquiler.csv")
test_path = os.path.join(DATA_DIR, "test_alquiler.csv")
metadata_path = os.path.join(DATA_DIR, "features_metadata_alquiler.json")

df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)

with open(metadata_path, "r", encoding="utf-8") as f:
    metadata = json.load(f)

feature_cols = metadata["features"]

print(f"Train: {df_train.shape[0]:,} registros x {df_train.shape[1]} columnas")
print(f"Test:  {df_test.shape[0]:,} registros x {df_test.shape[1]} columnas")
print(f"Número de features: {len(feature_cols)}")
print(f"Target: {TARGET_LOG} (escala logarítmica)")

# %% 3. Separar Matrices X e y
X_train = df_train[feature_cols]
y_train_log = df_train[TARGET_LOG]
y_train_real = df_train[TARGET]

X_test = df_test[feature_cols]
y_test_log = df_test[TARGET_LOG]
y_test_real = df_test[TARGET]

# Ponderación temporal E1 (más peso a los años más recientes del train)
DECAY = 0.85
ANIO_REF = 2023
sample_weights = df_train["Anio"].apply(lambda y: DECAY ** (ANIO_REF - y) if y <= ANIO_REF else 1.0).values
print(f"Ponderación temporal aplicada: peso min={sample_weights.min():.3f}, medio={sample_weights.mean():.3f}, max={sample_weights.max():.3f}")

# %% 4. Funciones de Evaluación
def evaluar_modelo(y_real, y_pred, nombre="Modelo"):
    mae = mean_absolute_error(y_real, y_pred)
    rmse = np.sqrt(mean_squared_error(y_real, y_pred))
    mape = mean_absolute_percentage_error(y_real, y_pred) * 100
    r2 = r2_score(y_real, y_pred)

    print(f"\n{'=' * 50}")
    print(f" {nombre}")
    print(f"{'=' * 50}")
    print(f"  MAE:   S/. {mae:>12,.2f}")
    print(f"  RMSE:  S/. {rmse:>12,.2f}")
    print(f"  MAPE:  {mape:>12.2f}%")
    print(f"  R²:    {r2:>12.4f}")

    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2}

def predecir_escala_real(modelo, X):
    pred_log = modelo.predict(X)
    return np.expm1(pred_log)

# %% 5. Validación Cruzada Temporal (TimeSeriesSplit)
print("\n--- Validación Cruzada Temporal (5 Folds) ---")
tscv = TimeSeriesSplit(n_splits=5)

cv_model = XGBRegressor(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=5.0,
    min_child_weight=3,
    random_state=RANDOM_SEED,
    n_jobs=-1,
)

cv_scores = cross_val_score(cv_model, X_train, y_train_log, cv=tscv, scoring="neg_mean_absolute_percentage_error")
print(f"MAPE CV medio en escala log: {-cv_scores.mean() * 100:.2f}% (std: {cv_scores.std() * 100:.2f}%)")

# %% 6. Entrenamiento del Modelo Final de Alquiler
print("\n--- Entrenando Modelo Final de Alquiler con XGBoost ---")
t0 = time.time()

model_final = XGBRegressor(
    n_estimators=600,
    max_depth=7,
    learning_rate=0.04,
    subsample=0.85,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=4.0,
    min_child_weight=3,
    random_state=RANDOM_SEED,
    n_jobs=-1,
)

model_final.fit(
    X_train,
    y_train_log,
    sample_weight=sample_weights,
    eval_set=[(X_train, y_train_log), (X_test, y_test_log)],
    verbose=100,
)
print(f"Entrenamiento completado en {time.time() - t0:.1f} segundos")

# %% 7. Evaluación Out-of-Time en Test Set (2024–2025)
y_pred_train = predecir_escala_real(model_final, X_train)
y_pred_test = predecir_escala_real(model_final, X_test)

metricas_train = evaluar_modelo(y_train_real, y_pred_train, "Métricas Train (2016-2023)")
metricas_test = evaluar_modelo(y_test_real, y_pred_test, "Métricas Test Out-of-Time (2024-2025)")

# %% 8. Gráficos de Diagnóstico
# 8.1 Real vs Predicho
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(y_test_real, y_pred_test, alpha=0.15, color="teal", s=15)
max_val = max(y_test_real.max(), y_pred_test.max())
ax.plot([0, max_val], [0, max_val], "r--", lw=1.5, label="Predicción Perfecta")
ax.set_xlabel("Alquiler Real (S/. constantes)")
ax.set_ylabel("Alquiler Predicho (S/. constantes)")
ax.set_title(f"Test Out-of-Time 2024-2025 — R²={metricas_test['R2']:.3f} | MAPE={metricas_test['MAPE']:.2f}%")
ax.legend()
plt.tight_layout()
fig_path_rvp = os.path.join(REPORTS_DIR, "real_vs_predicho_alquiler.png")
plt.savefig(fig_path_rvp)
plt.close()
print(f"\n[OK] Gráfico Real vs Predicho guardado: {fig_path_rvp}")

# 8.2 Importancia de Variables (Top 15)
importances = pd.Series(model_final.feature_importances_, index=feature_cols).sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(10, 6))
importances.head(15).plot(kind="barh", ax=ax, color="steelblue")
ax.invert_yaxis()
ax.set_title("Top 15 Features Más Importantes — XGBoost Alquiler")
ax.set_xlabel("Importancia Relativa (Gain)")
plt.tight_layout()
fig_path_imp = os.path.join(REPORTS_DIR, "feature_importance_alquiler.png")
plt.savefig(fig_path_imp)
plt.close()
print(f"[OK] Gráfico Feature Importance guardado: {fig_path_imp}")

# %% 9. Guardar Artefactos del Modelo
model_save_path = os.path.join(MODELS_DIR, "xgboost_alquiler_v1.pkl")
joblib.dump(model_final, model_save_path)
print(f"\n[OK] Modelo serializado guardado en: {model_save_path}")

# Guardar métricas en JSON
metricas_alquiler = {
    "modelo": "XGBoost Alquiler v1",
    "fecha_evaluacion": time.strftime("%Y-%m-%d %H:%M:%S"),
    "train_years": metadata["train_years"],
    "test_years": metadata["test_years"],
    "train_rows": len(X_train),
    "test_rows": len(X_test),
    "metricas_train": metricas_train,
    "metricas_test": metricas_test,
    "top_5_features": importances.head(5).to_dict(),
}

metrics_json_path = os.path.join(REPORTS_DIR, "metricas_alquiler.json")
with open(metrics_json_path, "w", encoding="utf-8") as f:
    json.dump(metricas_alquiler, f, indent=2, ensure_ascii=False)
print(f"[OK] Resumen de métricas guardado en: {metrics_json_path}")

print("\n" + "=" * 60)
print("[OK] MODELO DE ALQUILER ENTRENADO Y EVALUADO CON ÉXITO")
print("=" * 60)
