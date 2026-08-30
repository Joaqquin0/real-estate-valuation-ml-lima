# =============================================================================
# 03_ENTRENAMIENTO_XGBOOST.py
# =============================================================================
# Entrenamiento del Modelo XGBoost - Prediccion de Precio de Venta
# Valoracion Inmobiliaria en Lima Metropolitana
#
# CRISP-DM: Fase 3 - Modelado
# Historia Tecnica: TH-02 (Entrenar modelo de venta)
#
# Input:  train.csv, test.csv (generados por 02_preprocesamiento.py)
# Output: xgboost_venta.pkl, metricas, graficos SHAP
# =============================================================================

# %% [markdown]
# #  Entrenamiento del Modelo XGBoost
# ## Baseline -> Validacion Cruzada -> Tuning -> Evaluacion -> SHAP

# %% Instalacion (solo en Colab)
# !pip install -q xgboost shap optuna

# %% Importaciones
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
matplotlib.use('Agg')  # Backend sin GUI para ejecucion en terminal
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["figure.dpi"] = 120

# %% [markdown]
# ## 1. Configuracion y Carga de Datos

# %% Configuracion
# =============================================
# Rutas locales
# =============================================
BASE_DIR = r"d:\NuevaCarpetaLool\python_modelo_tesis"
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports", "figures")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

TARGET = "Precio_Soles_Const"
TARGET_LOG = f"{TARGET}_log"
RANDOM_SEED = 42
MAPE_BENCHMARK = 17.89  # Oporto et al. (2024)

# %% Cargar datos
df_train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
df_test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

# Cargar metadata
with open(os.path.join(DATA_DIR, "features_metadata.json"), "r") as f:
    metadata = json.load(f)

feature_cols = metadata["features"]

print(f"Train: {df_train.shape[0]:,} x {df_train.shape[1]}")
print(f"Test:  {df_test.shape[0]:,} x {df_test.shape[1]}")
print(f"Features: {len(feature_cols)}")
print(f"Target: {TARGET_LOG} (log-transformado)")

# %% Separar X e y
X_train = df_train[feature_cols]
y_train_log = df_train[TARGET_LOG]
y_train_real = df_train[TARGET]

X_test = df_test[feature_cols]
y_test_log = df_test[TARGET_LOG]
y_test_real = df_test[TARGET]

print(f"\nX_train: {X_train.shape}")
print(f"X_test:  {X_test.shape}")

# %% [markdown]
# ## 2. Funciones de Evaluacion

# %% Funciones auxiliares
def evaluar_modelo(y_real, y_pred, nombre="Modelo"):
    """Calcula y muestra las 4 metricas del TI."""
    metricas = {
        "MAE": mean_absolute_error(y_real, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_real, y_pred)),
        "MAPE": mean_absolute_percentage_error(y_real, y_pred) * 100,
        "R2": r2_score(y_real, y_pred),
    }

    print(f"\n{'=' * 50}")
    print(f" {nombre}")
    print(f"{'=' * 50}")
    print(f"  MAE:   S/. {metricas['MAE']:>12,.2f}")
    print(f"  RMSE:  S/. {metricas['RMSE']:>12,.2f}")
    print(f"  MAPE:  {metricas['MAPE']:>12.2f}%")
    print(f"  R²:    {metricas['R2']:>12.4f}")

    # Comparar con benchmark
    if metricas["MAPE"] < 10:
        print(f"   MAPE < 10% -> Estandar IAAO: EXCELENTE")
    elif metricas["MAPE"] < MAPE_BENCHMARK:
        print(f"   MAPE < {MAPE_BENCHMARK}% -> Supera Oporto et al. (2024)")
    else:
        print(f"   MAPE >= {MAPE_BENCHMARK}% -> No supera benchmark")

    return metricas


def predecir_escala_real(modelo, X, y_log):
    """Predice en log-scale y convierte a escala real (S/.)."""
    y_pred_log = modelo.predict(X)
    y_pred_real = np.expm1(y_pred_log)    # Inversa de log1p
    y_real = np.expm1(y_log)              # Inversa de log1p
    return y_real, y_pred_real

# %% [markdown]
# ## 3. Modelo Baseline (Parametros Default)
#
# Primer paso: entrenar con parametros razonables para tener un punto de referencia.

# %% Baseline
print(" PASO 1: MODELO BASELINE")
print("-" * 50)

params_baseline = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 5,
    "min_child_weight": 3,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
    "tree_method": "hist",    # Mas rapido para datasets grandes
}

start = time.time()
modelo_baseline = XGBRegressor(**params_baseline)
modelo_baseline.fit(
    X_train, y_train_log,
    eval_set=[(X_test, y_test_log)],
    verbose=100,
)
tiempo_baseline = time.time() - start
print(f"\nTiempo de entrenamiento: {tiempo_baseline:.1f}s")

# Evaluar en escala real
y_real_test, y_pred_baseline = predecir_escala_real(modelo_baseline, X_test, y_test_log)
metricas_baseline = evaluar_modelo(y_real_test, y_pred_baseline, "BASELINE (Test)")

# Tambien evaluar en train para verificar overfitting
y_real_train, y_pred_train = predecir_escala_real(modelo_baseline, X_train, y_train_log)
metricas_train = evaluar_modelo(y_real_train, y_pred_train, "BASELINE (Train)")

gap_mape = metricas_train["MAPE"] - metricas_baseline["MAPE"]
print(f"\n Gap Train-Test MAPE: {abs(gap_mape):.2f}% "
      f"({' Posible overfitting' if abs(gap_mape) > 5 else ' Aceptable'})")

# %% [markdown]
# ## 4. Validacion Cruzada Temporal
#
# Usamos `TimeSeriesSplit` con 5 folds para verificar la estabilidad
# del modelo a traves del tiempo (no solo un split fijo).

# %% Validacion cruzada temporal
print("\n PASO 2: VALIDACION CRUZADA TEMPORAL")
print("-" * 50)

tscv = TimeSeriesSplit(n_splits=5)
modelo_cv = XGBRegressor(**params_baseline)

# CV con scoring MAPE (viene negativo por convencion sklearn)
scores_neg_mape = cross_val_score(
    modelo_cv, X_train, y_train_log,
    cv=tscv,
    scoring="neg_mean_absolute_percentage_error",
    n_jobs=-1,
)

mape_scores = -scores_neg_mape * 100  # Convertir a porcentaje positivo

print("Resultados por fold:")
for i, s in enumerate(mape_scores):
    print(f"  Fold {i+1}: MAPE = {s:.2f}%")
print(f"\n  Media:   {mape_scores.mean():.2f}%")
print(f"  Std:     {mape_scores.std():.2f}%")
print(f"  Rango:   [{mape_scores.min():.2f}% - {mape_scores.max():.2f}%]")

# %% [markdown]
# ## 5. Tuning de Hiperparametros con Optuna
#
# Optuna busca automaticamente la mejor combinacion de hiperparametros
# usando validacion cruzada temporal como metrica de evaluacion.

# %% Tuning con Optuna
print("\n PASO 3: TUNING CON OPTUNA")
print("-" * 50)

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "random_state": RANDOM_SEED,
            "n_jobs": -1,
            "tree_method": "hist",
        }

        modelo = XGBRegressor(**params)
        tscv = TimeSeriesSplit(n_splits=5)

        scores = cross_val_score(
            modelo, X_train, y_train_log,
            cv=tscv,
            scoring="neg_mean_absolute_percentage_error",
            n_jobs=1,
        )
        return -scores.mean()  # Minimizar MAPE

    study = optuna.create_study(direction="minimize")
    study.optimize(
        objective,
        n_trials=60,       # Ajustar si tarda mucho (minimo 30)
        timeout=900,        # Maximo 15 minutos
        show_progress_bar=True,
    )

    best_params = study.best_trial.params
    best_params["random_state"] = RANDOM_SEED
    best_params["n_jobs"] = -1
    best_params["tree_method"] = "hist"

    print(f"\n Mejor MAPE encontrado (CV): {study.best_value * 100:.2f}%")
    print(f"Mejores hiperparametros:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")

    OPTUNA_OK = True

except ImportError:
    print(" Optuna no instalado. Usando parametros baseline.")
    print("  Para instalar: !pip install optuna")
    best_params = params_baseline
    OPTUNA_OK = False

# %% [markdown]
# ## 6. Entrenar Modelo Final con Mejores Parametros

# %% Modelo final
print("\n PASO 4: MODELO FINAL")
print("-" * 50)

start = time.time()
modelo_final = XGBRegressor(**best_params)
modelo_final.fit(
    X_train, y_train_log,
    eval_set=[(X_test, y_test_log)],
    verbose=100,
)
tiempo_final = time.time() - start
print(f"Tiempo de entrenamiento: {tiempo_final:.1f}s")

# Evaluar modelo final
y_real_test, y_pred_final = predecir_escala_real(modelo_final, X_test, y_test_log)
metricas_final = evaluar_modelo(y_real_test, y_pred_final, "MODELO FINAL (Test)")

y_real_train, y_pred_train_final = predecir_escala_real(modelo_final, X_train, y_train_log)
metricas_train_final = evaluar_modelo(y_real_train, y_pred_train_final, "MODELO FINAL (Train)")

# %% Comparar baseline vs final
print("\n COMPARACION BASELINE vs FINAL:")
print(f"{'Metrica':<10} {'Baseline':>12} {'Final':>12} {'Mejora':>12}")
print("-" * 48)
for metric in ["MAE", "RMSE", "MAPE", "R2"]:
    base = metricas_baseline[metric]
    final = metricas_final[metric]
    if metric == "R2":
        mejora = final - base
        print(f"{metric:<10} {base:>12.4f} {final:>12.4f} {mejora:>+12.4f}")
    else:
        mejora = base - final  # Para MAE, RMSE, MAPE menor es mejor
        print(f"{metric:<10} {base:>12.2f} {final:>12.2f} {mejora:>+12.2f}")

# %% [markdown]
# ## 7. Guardar Modelo

# %% Guardar
model_path = os.path.join(MODELS_DIR, "xgboost_venta.pkl")
joblib.dump(modelo_final, model_path)

# Guardar hiperparametros
params_path = os.path.join(MODELS_DIR, "xgboost_venta_params.json")
with open(params_path, "w") as f:
    json.dump(best_params, f, indent=2, default=str)

# Guardar metricas
metricas_export = {
    "baseline_test": metricas_baseline,
    "final_test": metricas_final,
    "final_train": metricas_train_final,
    "benchmark_oporto": MAPE_BENCHMARK,
    "aprobado": metricas_final["MAPE"] < MAPE_BENCHMARK,
}
metricas_path = os.path.join(MODELS_DIR, "xgboost_venta_metricas.json")
with open(metricas_path, "w") as f:
    json.dump(metricas_export, f, indent=2, default=str)

print(f"\n ARCHIVOS GUARDADOS:")
print(f"  {model_path}")
print(f"  {params_path}")
print(f"  {metricas_path}")

# %% [markdown]
# ## 8. Diagnosticos del Modelo

# %% 8.1 Real vs Predicho
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(y_real_test, y_pred_final, alpha=0.2, s=5, color="#4A90D9")
min_val = min(y_real_test.min(), y_pred_final.min())
max_val = max(y_real_test.max(), y_pred_final.max())
ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1.5, label="Prediccion perfecta")
ax.set_xlabel("Valor Real (S/.)")
ax.set_ylabel("Valor Predicho (S/.)")
ax.set_title(f"Real vs Predicho - MAPE: {metricas_final['MAPE']:.2f}%")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, "real_vs_predicho.png"), dpi=150, bbox_inches="tight")
plt.show()

# %% 8.2 Residuos
residuos = y_real_test - y_pred_final

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].scatter(y_pred_final, residuos, alpha=0.2, s=5, color="#4A90D9")
axes[0].axhline(y=0, color="red", linestyle="--", linewidth=1)
axes[0].set_xlabel("Valor Predicho (S/.)")
axes[0].set_ylabel("Residuo (S/.)")
axes[0].set_title("Residuos vs Predichos")

axes[1].hist(residuos, bins=80, edgecolor="black", alpha=0.7, color="#4A90D9")
axes[1].axvline(x=0, color="red", linestyle="--", linewidth=1)
axes[1].set_xlabel("Residuo (S/.)")
axes[1].set_ylabel("Frecuencia")
axes[1].set_title("Distribucion de Residuos")

plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, "residuos.png"), dpi=150, bbox_inches="tight")
plt.show()

# %% 8.3 Feature Importance (nativa XGBoost)
importance = modelo_final.feature_importances_
importance_df = pd.DataFrame({
    "feature": feature_cols,
    "importance": importance,
}).sort_values("importance", ascending=False)

fig, ax = plt.subplots(figsize=(10, 8))
importance_df.head(20).plot(
    kind="barh", x="feature", y="importance",
    ax=ax, color="#4A90D9", legend=False,
)
ax.set_xlabel("Importancia (gain)")
ax.set_title("Top 20 Features - Importancia XGBoost")
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, "feature_importance_xgboost.png"), dpi=150, bbox_inches="tight")
plt.show()

print("\nTop 10 Features:")
print(importance_df.head(10).to_string(index=False))

# %% 8.4 Error por distrito
distrito_ref = pd.read_csv(os.path.join(DATA_DIR, "distrito_referencia.csv"))
distrito_test = distrito_ref[distrito_ref["split"] == "test"]["Distrito"].values

df_eval = pd.DataFrame({
    "real": y_real_test,
    "pred": y_pred_final,
    "distrito": distrito_test[:len(y_real_test)],
})
df_eval["ape"] = np.abs((df_eval["real"] - df_eval["pred"]) / df_eval["real"]) * 100

mape_distrito = df_eval.groupby("distrito")["ape"].agg(["mean", "count"]).rename(
    columns={"mean": "MAPE", "count": "n_registros"}
).sort_values("MAPE", ascending=False)

fig, ax = plt.subplots(figsize=(12, 7))
mape_distrito["MAPE"].plot(kind="barh", ax=ax, color="#4A90D9")
ax.axvline(x=MAPE_BENCHMARK, color="red", linestyle="--", linewidth=1.5, label=f"Benchmark {MAPE_BENCHMARK}%")
ax.axvline(x=10, color="green", linestyle="--", linewidth=1.5, label="Estandar IAAO 10%")
ax.set_xlabel("MAPE (%)")
ax.set_title("MAPE por Distrito")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, "mape_por_distrito.png"), dpi=150, bbox_inches="tight")
plt.show()

print("\nMAPE por Distrito:")
print(mape_distrito.to_string())

# %% [markdown]
# ## 9. Explicabilidad SHAP
#
# SHAP (SHapley Additive exPlanations) abre la "caja negra" del modelo.
# Muestra exactamente que variables subieron o bajaron el precio predicho.

# %% SHAP
print("\n PASO 5: EXPLICABILIDAD SHAP")
print("-" * 50)

import shap

# Crear explainer
explainer = shap.TreeExplainer(modelo_final)

# Calcular SHAP values (usar muestra si el dataset es muy grande)
N_SHAP = min(5000, len(X_test))  # Maximo 5000 para no saturar memoria
X_shap = X_test.sample(n=N_SHAP, random_state=RANDOM_SEED)
shap_values = explainer(X_shap)

print(f"SHAP values calculados para {N_SHAP:,} observaciones")

# %% 9.1 Summary Plot (Beeswarm)
plt.figure(figsize=(12, 8))
shap.plots.beeswarm(shap_values, max_display=20, show=False)
plt.title("Importancia Global de Variables (SHAP)")
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, "shap_beeswarm.png"), dpi=150, bbox_inches="tight")
plt.show()

# %% 9.2 Bar Plot
plt.figure(figsize=(10, 6))
shap.plots.bar(shap_values, max_display=15, show=False)
plt.title("Top 15 Features - Importancia SHAP")
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, "shap_bar.png"), dpi=150, bbox_inches="tight")
plt.show()

# %% 9.3 Waterfall (ejemplo individual)
plt.figure(figsize=(10, 8))
shap.plots.waterfall(shap_values[0], max_display=15, show=False)
plt.title("Explicacion Individual - Ejemplo de Prediccion")
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, "shap_waterfall.png"), dpi=150, bbox_inches="tight")
plt.show()

# %% 9.4 Dependence: Superficie
if "Superficie" in X_shap.columns:
    plt.figure(figsize=(10, 6))
    shap.plots.scatter(shap_values[:, "Superficie"], show=False)
    plt.title("Dependencia SHAP: Superficie")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "shap_dependence_superficie.png"), dpi=150, bbox_inches="tight")
    plt.show()

# %% 9.5 Dependence: distrito_encoded
if "distrito_encoded" in X_shap.columns:
    plt.figure(figsize=(10, 6))
    shap.plots.scatter(shap_values[:, "distrito_encoded"], show=False)
    plt.title("Dependencia SHAP: Distrito (encoded)")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "shap_dependence_distrito.png"), dpi=150, bbox_inches="tight")
    plt.show()

# %% [markdown]
# ## 10. Resumen Final

# %% Resumen
print("\n" + "=" * 60)
print(" RESUMEN FINAL DEL ENTRENAMIENTO")
print("=" * 60)

print(f"\n MODELO FINAL: XGBoost Regressor")
print(f"   Features: {len(feature_cols)}")
print(f"   Train: {len(X_train):,} | Test: {len(X_test):,}")

print(f"\n METRICAS EN TEST:")
print(f"   MAE:  S/. {metricas_final['MAE']:,.2f}")
print(f"   RMSE: S/. {metricas_final['RMSE']:,.2f}")
print(f"   MAPE: {metricas_final['MAPE']:.2f}%")
print(f"   R²:   {metricas_final['R2']:.4f}")

print(f"\n VS BENCHMARK:")
print(f"   Oporto et al. (2024): {MAPE_BENCHMARK}%")
aprobado = metricas_final["MAPE"] < MAPE_BENCHMARK
print(f"   Resultado: {' SUPERA BENCHMARK' if aprobado else ' NO SUPERA - Requiere mejoras'}")

if aprobado and metricas_final["MAPE"] < 10:
    print(f"    Ademas cumple estandar IAAO (<10%)")

print(f"\n ARCHIVOS GENERADOS:")
print(f"   Modelo: {model_path}")
print(f"   Graficos: {REPORTS_DIR}/")
archivos_reportes = [f for f in os.listdir(REPORTS_DIR) if f.endswith(".png")]
for f in sorted(archivos_reportes):
    print(f"      {f}")

print(f"\n Pipeline completo. Modelo listo para integracion con API REST.")
