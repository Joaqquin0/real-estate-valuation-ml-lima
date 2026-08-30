"""
Re-entrenamiento con XGBoost 1.7.6 + SHAP completo.
Usa los hiperparametros optimos ya encontrados por Optuna.
No necesita re-correr el tuning.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import joblib
import json
import os
import shap
import xgboost
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error

print(f"XGBoost version: {xgboost.__version__}")
print(f"SHAP version: {shap.__version__}")

# ============================================================
# RUTAS
# ============================================================
BASE_DIR = r"d:\NuevaCarpetaLool\python_modelo_tesis"
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports", "figures")
os.makedirs(REPORTS_DIR, exist_ok=True)

TARGET = "Precio_Soles_Const"
TARGET_LOG = f"{TARGET}_log"
RANDOM_SEED = 42
MAPE_BENCHMARK = 17.89

# ============================================================
# CARGAR DATOS
# ============================================================
with open(os.path.join(DATA_DIR, "features_metadata.json"), "r") as f:
    metadata = json.load(f)
feature_cols = metadata["features"]

df_train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
df_test  = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

X_train = df_train[feature_cols]
y_train_log = df_train[TARGET_LOG]
y_train_real = df_train[TARGET]

X_test = df_test[feature_cols]
y_test_log = df_test[TARGET_LOG]
y_test_real = df_test[TARGET]

print(f"Train: {X_train.shape} | Test: {X_test.shape}")

# ============================================================
# E1: PONDERACION TEMPORAL (decaimiento exponencial por anio)
# Los datos recientes pesan mas: 2023 pesa 1.0, 2016 pesa 0.80^7 ≈ 0.21
# Esto calibra el modelo hacia el comportamiento del mercado reciente.
# ============================================================
DECAY = 0.85
ANIO_MAX_TRAIN = df_train["Anio"].max()  # 2023
sample_weights = df_train["Anio"].apply(lambda y: DECAY ** (ANIO_MAX_TRAIN - y)).values

print(f"\n--- E1: PONDERACION TEMPORAL (decay={DECAY}) ---")
print(f"  Anio 2023 (peso=1.0) ... Anio 2016 (peso={DECAY**(ANIO_MAX_TRAIN - df_train['Anio'].min()):.3f})")
print(f"  Pesos: min={sample_weights.min():.3f} | media={sample_weights.mean():.3f} | max={sample_weights.max():.3f}")

# ============================================================
# HIPERPARAMETROS OPTIMOS (60 trials Optuna, corrida original)
# Nota: Re-tuning con 20 trials no mejoro vs 60 trials originales.
# Se mantienen los params probados con el dataset sin interaccion
# ya que sup_x_distrito no mejoro las metricas en test.
# ============================================================
# ============================================================
# HIPERPARAMETROS OPTIMOS (60 trials Optuna)
# E1 aplicado via sample_weight en fit()
# Nota: reg:pseudohubererror no converge con log1p target en XGBoost 1.7.6
# Se mantiene reg:squarederror (default) sobre escala log, que es la forma
# correcta de aproximar minimizacion de error relativo/MAPE.
# ============================================================
best_params = {
    "n_estimators": 956,
    "max_depth": 4,
    "learning_rate": 0.06268598404619886,
    "subsample": 0.8517657357304793,
    "colsample_bytree": 0.8467798626692052,
    "reg_alpha": 4.0598061516484,
    "reg_lambda": 0.030663573061419286,
    "min_child_weight": 9,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
    "use_label_encoder": False,
    "eval_metric": "rmse",
}

# ============================================================
# ENTRENAR MODELO FINAL
# ============================================================
print("\n--- ENTRENANDO MODELO FINAL (XGBoost 1.7.6) ---")
import time
start = time.time()
modelo = XGBRegressor(**best_params)
modelo.fit(
    X_train, y_train_log,
    sample_weight=sample_weights,   # E1: Ponderacion temporal
    eval_set=[(X_test, y_test_log)],
    verbose=100,
)
print(f"Tiempo: {time.time()-start:.1f}s")

# ============================================================
# EVALUACION
# ============================================================
def evaluar(y_real, y_pred, nombre):
    m = {
        "MAE":  mean_absolute_error(y_real, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_real, y_pred)),
        "MAPE": mean_absolute_percentage_error(y_real, y_pred) * 100,
        "R2":   r2_score(y_real, y_pred),
    }
    print(f"\n{'='*50}")
    print(f"{nombre}")
    print(f"{'='*50}")
    print(f"  MAE:   S/. {m['MAE']:>12,.2f}")
    print(f"  RMSE:  S/. {m['RMSE']:>12,.2f}")
    print(f"  MAPE:  {m['MAPE']:>12.2f}%")
    print(f"  R2:    {m['R2']:>12.4f}")
    estado = "[OK] Supera benchmark" if m["MAPE"] < MAPE_BENCHMARK else "[!!] No supera benchmark"
    print(f"  {estado}")
    return m

y_pred_log = modelo.predict(X_test)
y_pred_real = np.expm1(y_pred_log)
metricas = evaluar(y_test_real, y_pred_real, "MODELO FINAL - TEST")

y_pred_train_log = modelo.predict(X_train)
y_pred_train_real = np.expm1(y_pred_train_log)
evaluar(y_train_real, y_pred_train_real, "MODELO FINAL - TRAIN")

# ============================================================
# GUARDAR MODELO
# ============================================================
model_path = os.path.join(MODELS_DIR, "xgboost_venta_v2.pkl")
joblib.dump(modelo, model_path)
print(f"\nModelo guardado: {model_path}")

metricas_path = os.path.join(MODELS_DIR, "xgboost_venta_v2_metricas.json")
with open(metricas_path, "w") as f:
    json.dump({"test": metricas, "xgboost_version": xgboost.__version__}, f, indent=2)

# ============================================================
# GRAFICOS DE DIAGNOSTICO
# ============================================================
print("\n--- GRAFICOS DE DIAGNOSTICO ---")

# Real vs Predicho
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(y_test_real, y_pred_real, alpha=0.2, s=5, color="#4A90D9")
lim = [min(y_test_real.min(), y_pred_real.min()), max(y_test_real.max(), y_pred_real.max())]
ax.plot(lim, lim, "r--", linewidth=1.5)
ax.set_xlabel("Valor Real (S/.)")
ax.set_ylabel("Valor Predicho (S/.)")
ax.set_title(f"Real vs Predicho - MAPE: {metricas['MAPE']:.2f}%")
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, "real_vs_predicho.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  real_vs_predicho.png guardado")

# Residuos
residuos = y_test_real.values - y_pred_real
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].scatter(y_pred_real, residuos, alpha=0.2, s=5, color="#4A90D9")
axes[0].axhline(0, color="red", linestyle="--")
axes[0].set_xlabel("Predicho (S/.)")
axes[0].set_ylabel("Residuo (S/.)")
axes[0].set_title("Residuos vs Predichos")
axes[1].hist(residuos, bins=80, edgecolor="black", alpha=0.7, color="#4A90D9")
axes[1].axvline(0, color="red", linestyle="--")
axes[1].set_title("Distribucion de Residuos")
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, "residuos.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  residuos.png guardado")

# Feature Importance nativa
importance_df = pd.DataFrame({
    "feature": feature_cols,
    "importance": modelo.feature_importances_,
}).sort_values("importance", ascending=False)
fig, ax = plt.subplots(figsize=(10, 8))
importance_df.head(20).plot(kind="barh", x="feature", y="importance", ax=ax, color="#4A90D9", legend=False)
ax.set_title("Top 20 Features - Importancia XGBoost")
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, "feature_importance_xgboost.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  feature_importance_xgboost.png guardado")

# MAPE por distrito
distrito_ref = pd.read_csv(os.path.join(DATA_DIR, "distrito_referencia.csv"))
dist_test = distrito_ref[distrito_ref["split"] == "test"]["Distrito"].reset_index(drop=True)
df_eval = pd.DataFrame({
    "real": y_test_real.values,
    "pred": y_pred_real,
    "distrito": dist_test.values[:len(y_test_real)],
})
df_eval["ape"] = np.abs((df_eval["real"] - df_eval["pred"]) / df_eval["real"]) * 100
mape_dist = df_eval.groupby("distrito")["ape"].mean().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(12, 7))
mape_dist.plot(kind="barh", ax=ax, color="#4A90D9")
ax.axvline(MAPE_BENCHMARK, color="red", linestyle="--", linewidth=1.5, label=f"Benchmark {MAPE_BENCHMARK}%")
ax.axvline(10, color="green", linestyle="--", linewidth=1.5, label="IAAO 10%")
ax.set_title("MAPE por Distrito")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, "mape_por_distrito.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  mape_por_distrito.png guardado")

# ============================================================
# SHAP
# ============================================================
print("\n--- SHAP EXPLICABILIDAD ---")
N_SHAP = min(3000, len(X_test))
X_shap = X_test.sample(n=N_SHAP, random_state=RANDOM_SEED).reset_index(drop=True)
print(f"Calculando SHAP values para {N_SHAP:,} observaciones...")

explainer = shap.TreeExplainer(modelo)
shap_values = explainer(X_shap)
print(f"SHAP values calculados: {shap_values.shape}")

# Beeswarm
print("[1/5] Beeswarm...")
fig, ax = plt.subplots(figsize=(12, 9))
shap.plots.beeswarm(shap_values, max_display=20, show=False)
plt.title("Importancia Global de Variables - SHAP Beeswarm")
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, "shap_beeswarm.png"), dpi=150, bbox_inches="tight")
plt.close()

# Bar
print("[2/5] Bar...")
fig, ax = plt.subplots(figsize=(10, 7))
shap.plots.bar(shap_values, max_display=15, show=False)
plt.title("Top 15 Features - Importancia SHAP")
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, "shap_bar.png"), dpi=150, bbox_inches="tight")
plt.close()

# Waterfall
print("[3/5] Waterfall...")
fig, ax = plt.subplots(figsize=(12, 8))
shap.plots.waterfall(shap_values[0], max_display=15, show=False)
plt.title("Explicacion Individual - Ejemplo")
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, "shap_waterfall.png"), dpi=150, bbox_inches="tight")
plt.close()

# Dependence Superficie
print("[4/5] Dependence Superficie...")
if "Superficie" in X_shap.columns:
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.plots.scatter(shap_values[:, "Superficie"], show=False)
    plt.title("Dependencia SHAP: Superficie")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "shap_dependence_superficie.png"), dpi=150, bbox_inches="tight")
    plt.close()

# Dependence Distrito
print("[5/5] Dependence Distrito...")
if "distrito_encoded" in X_shap.columns:
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.plots.scatter(shap_values[:, "distrito_encoded"], show=False)
    plt.title("Dependencia SHAP: Distrito")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "shap_dependence_distrito.png"), dpi=150, bbox_inches="tight")
    plt.close()

# Resumen SHAP
mean_abs = pd.DataFrame({
    "feature": feature_cols,
    "shap_importance": np.abs(shap_values.values).mean(axis=0),
}).sort_values("shap_importance", ascending=False)
print("\nIMPORTANCIA SHAP (mean |value|):")
print(mean_abs.head(15).to_string(index=False))

# ============================================================
# RESUMEN FINAL
# ============================================================
print("\n" + "=" * 60)
print("PIPELINE COMPLETADO")
print("=" * 60)
print(f"\nMETRICAS FINALES (Test):")
print(f"  MAE:  S/. {metricas['MAE']:,.2f}")
print(f"  RMSE: S/. {metricas['RMSE']:,.2f}")
print(f"  MAPE: {metricas['MAPE']:.2f}%  (benchmark: {MAPE_BENCHMARK}%)")
print(f"  R2:   {metricas['R2']:.4f}")
print(f"\nResultado: {'[OK] MODELO APROBADO - Supera benchmark' if metricas['MAPE'] < MAPE_BENCHMARK else '[!!] REQUIERE MEJORAS'}")

archivos_png = [f for f in os.listdir(REPORTS_DIR) if f.endswith(".png")]
print(f"\nGraficos generados ({len(archivos_png)}):")
for f in sorted(archivos_png):
    print(f"  reports/figures/{f}")
print(f"\nModelo: {model_path}")
