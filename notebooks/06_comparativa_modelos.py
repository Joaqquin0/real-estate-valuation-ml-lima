# =============================================================================
# 06_COMPARATIVA_MODELOS.py
# =============================================================================
# Comparativa Final: 3 Versiones del Modelo XGBoost de Valoracion Inmobiliaria
#
# CRISP-DM: Fase 4 - Evaluacion
#
# Modelos comparados:
#   M1: BCRP Baseline        -> xgboost_venta.pkl
#   M2: BCRP Final (E1)      -> xgboost_venta_v2.pkl
#   M3: BCRP Ampliado (E1)   -> xgboost_venta_ampliado.pkl
#
# Fuente de datos: BCRP (Banco Central de Reserva del Peru) 2016-2025
# Evaluacion: test.csv (2024-2025, out-of-time, nunca tocado)
# Output: reports/figures/comparativa_modelos.png
#         reports/comparativa_modelos.csv
# =============================================================================

# %% [markdown]
# # Comparativa de Modelos: Baseline vs BCRP Final vs BCRP Ampliado

# %% Importaciones
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import pandas as pd
import numpy as np
import json
import os
import joblib
import warnings

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore")
plt.rcParams["figure.dpi"] = 130
plt.rcParams["font.family"] = "DejaVu Sans"

# %% Configuracion de rutas
BASE_DIR    = r"d:\NuevaCarpetaLool\python_modelo_tesis"
DATA_DIR    = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR  = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports", "figures")
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "reports"), exist_ok=True)

TARGET      = "Precio_Soles_Const"
TARGET_LOG  = f"{TARGET}_log"
RANDOM_SEED = 42
MAPE_BENCH  = 17.89  # Oporto et al. (2024)
MAPE_IAAO   = 10.0   # Estandar internacional IAAO

print("=" * 65)
print("06 - COMPARATIVA FINAL DE MODELOS")
print("=" * 65)

# %% Cargar datos de test
df_test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

with open(os.path.join(DATA_DIR, "features_metadata.json"), "r") as f:
    metadata = json.load(f)
feature_cols = metadata["features"]

X_test     = df_test[feature_cols]
y_test_log = df_test[TARGET_LOG]

print(f"\n[OK] Test set: {len(df_test):,} filas x {df_test.shape[1]} columnas")
print(f"[OK] Features: {len(feature_cols)}")

# %% Definicion de modelos a comparar
# M1 tiene pkl incompatible (XGBoost versión antigua + target con transformación distinta).
# Sus métricas se leen desde el JSON guardado en su momento de entrenamiento.
# M2 y M3 se evalúan en vivo contra test.csv.

MODELOS_LIVE = [
    {
        "nombre":      "M2 - BCRP Final (E1 + 33 feat)",
        "pkl":         "xgboost_venta_v2.pkl",
        "color":       "#56C271",
        "descripcion": "33 features | Ponderacion exponencial E1 (delta=0.85)",
    },
]

# M1 métricas leídas desde JSON (entrenado con versión anterior de XGBoost)
M1_METRICAS_PATH = os.path.join(MODELS_DIR, "xgboost_venta_metricas.json")
with open(M1_METRICAS_PATH) as f:
    m1_json = json.load(f)
m1_metricas_test = m1_json["final_test"]

# Resultado M1 (sin y_real/y_pred — no se puede re-evaluar)
resultado_m1 = {
    "Modelo":      "M1 - BCRP Baseline",
    "Descripcion": "28 features | Sin ponderacion temporal (métricas del JSON guardado)",
    "MAE":         m1_metricas_test["MAE"],
    "RMSE":        m1_metricas_test["RMSE"],
    "MAPE":        m1_metricas_test["MAPE"],
    "R2":          m1_metricas_test["R2"],
    "color":       "#5B8DD9",
    "y_real":      None,
    "y_pred":      None,
}

# %% Evaluar M2 y M3 en vivo
import xgboost as xgb

def cargar_modelo_live(pkl_path):
    """Carga modelo sklearn de XGBoost reciente."""
    try:
        modelo = joblib.load(pkl_path)
        _ = modelo.get_xgb_params()  # verifica compatibilidad
        return modelo, "sklearn"
    except AttributeError as e:
        raise RuntimeError(f"Modelo {pkl_path} incompatible: {e}")

def predecir_real(modelo, X_input, y_log):
    booster    = modelo.get_booster()
    feat_names = booster.feature_names
    X_use      = X_input[feat_names] if feat_names else X_input
    y_pred_log  = modelo.predict(X_use)
    y_pred_real = np.expm1(y_pred_log)
    y_real      = np.expm1(y_log.values)
    return y_real, y_pred_real

resultados_live = []

print("\n" + "=" * 65)
print("EVALUACION EN TEST SET (2024-2025)")
print("=" * 65)

# Primero M1 (desde JSON)
print(f"\n  [JSON] M1 - BCRP Baseline  (métricas guardadas del entrenamiento original)")
print(f"    MAE:  S/. {resultado_m1['MAE']:>12,.2f}")
print(f"    RMSE: S/. {resultado_m1['RMSE']:>12,.2f}")
print(f"    MAPE: {resultado_m1['MAPE']:>12.2f}%")
print(f"    R2:   {resultado_m1['R2']:>12.4f}")

for m in MODELOS_LIVE:
    pkl_path = os.path.join(MODELS_DIR, m["pkl"])
    if not os.path.exists(pkl_path):
        print(f"\n[SKIP] {m['nombre']} -> {m['pkl']} no encontrado")
        continue

    modelo, tipo = cargar_modelo_live(pkl_path)
    y_real, y_pred = predecir_real(modelo, X_test, y_test_log)

    status_mape = (
        "IAAO (<10%)"         if mean_absolute_percentage_error(y_real, y_pred)*100 < MAPE_IAAO else
        "Supera benchmark"    if mean_absolute_percentage_error(y_real, y_pred)*100 < MAPE_BENCH else
        "No supera benchmark"
    )

    metricas = {
        "Modelo":      m["nombre"],
        "Descripcion": m["descripcion"],
        "MAE":         mean_absolute_error(y_real, y_pred),
        "RMSE":        np.sqrt(mean_squared_error(y_real, y_pred)),
        "MAPE":        mean_absolute_percentage_error(y_real, y_pred) * 100,
        "R2":          r2_score(y_real, y_pred),
        "color":       m["color"],
        "y_real":      y_real,
        "y_pred":      y_pred,
    }
    resultados_live.append(metricas)
    print(f"\n  [{tipo.upper()}] {m['nombre']}")
    print(f"  {m['descripcion']}")
    print(f"    MAE:  S/. {metricas['MAE']:>12,.2f}")
    print(f"    RMSE: S/. {metricas['RMSE']:>12,.2f}")
    print(f"    MAPE: {metricas['MAPE']:>12.2f}%  [{status_mape}]")
    print(f"    R2:   {metricas['R2']:>12.4f}")

# Lista completa para tabla (M1 primero, luego M2 y M3)
resultados = [resultado_m1] + resultados_live

if not resultados:
    raise RuntimeError("No se encontro ningun modelo.")

# %% Tabla comparativa
print("\n" + "=" * 65)
print("TABLA COMPARATIVA RESUMIDA")
print("=" * 65)

df_comp = pd.DataFrame([
    {
        "Modelo":      r["Modelo"],
        "MAE (S/.)":   f"{r['MAE']:,.0f}",
        "RMSE (S/.)":  f"{r['RMSE']:,.0f}",
        "MAPE (%)":    f"{r['MAPE']:.2f}%",
        "R2":          f"{r['R2']:.4f}",
        "Benchmark":   f"OK ({MAPE_BENCH}%)" if r["MAPE"] < MAPE_BENCH else f"FALLO ({MAPE_BENCH}%)",
    }
    for r in resultados
])
print(df_comp.to_string(index=False))

csv_path = os.path.join(BASE_DIR, "reports", "comparativa_modelos.csv")
df_comp.to_csv(csv_path, index=False, encoding="utf-8-sig")
print(f"\n[OK] Tabla guardada: {csv_path}")

# %% FIGURA 1: Comparativa de metricas (barras agrupadas)
metricas_plot = ["MAE", "RMSE", "MAPE", "R2"]
titulos_plot  = ["MAE (S/.)", "RMSE (S/.)", "MAPE (%)", "R2"]
n_modelos     = len(resultados)
x             = np.arange(n_modelos)
nombres_cortos = [r["Modelo"].split(" - ")[-1] for r in resultados]
colores        = [r["color"] for r in resultados]

fig, axes = plt.subplots(1, 4, figsize=(18, 5))
fig.suptitle(
    "Comparativa de Modelos XGBoost — Evaluacion en Test Set (2024–2025)",
    fontsize=14, fontweight="bold", y=1.02
)

for ax, met, tit in zip(axes, metricas_plot, titulos_plot):
    valores = [r[met] for r in resultados]
    bars    = ax.bar(x, valores, color=colores, alpha=0.88, edgecolor="white", linewidth=1.2)

    for bar, val in zip(bars, valores):
        if met in ["MAE", "RMSE"]:
            fmt = f"S/. {val:,.0f}"
        elif met == "MAPE":
            fmt = f"{val:.2f}%"
        else:
            fmt = f"{val:.4f}"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(valores) * 0.01,
            fmt, ha="center", va="bottom", fontsize=8.5, fontweight="bold"
        )

    if met == "MAPE":
        ax.axhline(MAPE_BENCH, color="#CC2929", linestyle="--",
                   linewidth=1.3, label=f"Benchmark {MAPE_BENCH}% (Oporto 2024)")
        ax.axhline(MAPE_IAAO, color="#1A7A3B", linestyle="--",
                   linewidth=1.3, label=f"IAAO {MAPE_IAAO}%")
        ax.legend(fontsize=7.5, loc="upper right")
    elif met == "R2":
        ax.axhline(0.80, color="#CC2929", linestyle="--",
                   linewidth=1.3, label="R2=0.80 (target)")
        ax.legend(fontsize=7.5, loc="upper right")

    ax.set_title(tit, fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"M{i+1}" for i in range(n_modelos)], fontsize=10)
    ax.set_ylim(bottom=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

leyenda = [
    mpatches.Patch(color=r["color"], alpha=0.88, label=f"M{i+1}: {nombres_cortos[i]}")
    for i, r in enumerate(resultados)
]
fig.legend(
    handles=leyenda,
    loc="lower center",
    ncol=n_modelos,
    bbox_to_anchor=(0.5, -0.04),
    frameon=True,
    fontsize=9.5,
)

plt.tight_layout()
out1 = os.path.join(REPORTS_DIR, "comparativa_modelos.png")
plt.savefig(out1, dpi=150, bbox_inches="tight")
plt.close()
print(f"\n[OK] Figura 1 guardada: {out1}")

# %% FIGURA 2: Real vs Predicho (solo M2 y M3 — M1 no tiene predicciones re-evaluables)
resultados_plot = [r for r in resultados if r["y_real"] is not None]
n_plot = len(resultados_plot)
fig, axes = plt.subplots(1, n_plot, figsize=(6 * n_plot, 6))
if n_plot == 1:
    axes = [axes]

fig.suptitle(
    "Real vs Predicho — Comparativa por Modelo (Test 2024-2025)",
    fontsize=13, fontweight="bold"
)

for ax, r in zip(axes, resultados_plot):
    y_real = r["y_real"]
    y_pred = r["y_pred"]
    lim_lo = min(y_real.min(), y_pred.min()) * 0.95
    lim_hi = max(y_real.max(), y_pred.max()) * 1.05

    ax.scatter(y_real, y_pred, alpha=0.12, s=4, color=r["color"])
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], "r--", linewidth=1.5, label="Prediccion perfecta")
    ax.set_xlim(lim_lo, lim_hi)
    ax.set_ylim(lim_lo, lim_hi)
    ax.set_xlabel("Valor Real (S/.)", fontsize=9)
    ax.set_ylabel("Valor Predicho (S/.)", fontsize=9)
    ax.set_title(f"{r['Modelo']}\nMAPE={r['MAPE']:.2f}% | R2={r['R2']:.4f}", fontsize=9.5)
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

plt.tight_layout()
out2 = os.path.join(REPORTS_DIR, "comparativa_real_vs_predicho.png")
plt.savefig(out2, dpi=150, bbox_inches="tight")
plt.close()
print(f"[OK] Figura 2 guardada: {out2}")

# %% FIGURA 3: Distribucion de error porcentual absoluto (solo M2 y M3)
fig, axes = plt.subplots(1, n_plot, figsize=(5 * n_plot, 4))
if n_plot == 1:
    axes = [axes]

fig.suptitle(
    "Distribucion del Error Porcentual Absoluto — Comparativa",
    fontsize=13, fontweight="bold"
)

for ax, r in zip(axes, resultados_plot):
    ape = np.abs((r["y_real"] - r["y_pred"]) / r["y_real"]) * 100
    ape_clip = np.clip(ape, 0, 60)
    ax.hist(ape_clip, bins=60, color=r["color"], alpha=0.80, edgecolor="white")
    ax.axvline(r["MAPE"], color="black", linestyle="--", linewidth=1.5,
               label=f"MAPE={r['MAPE']:.2f}%")
    ax.axvline(MAPE_BENCH, color="#CC2929", linestyle="--", linewidth=1.2,
               label=f"Bench {MAPE_BENCH}%")
    ax.set_xlabel("APE (%) — truncado en 60%", fontsize=9)
    ax.set_ylabel("Frecuencia", fontsize=9)
    ax.set_title(r["Modelo"].split(" - ")[-1], fontsize=9.5)
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

plt.tight_layout()
out3 = os.path.join(REPORTS_DIR, "comparativa_distribucion_error.png")
plt.savefig(out3, dpi=150, bbox_inches="tight")
plt.close()
print(f"[OK] Figura 3 guardada: {out3}")

# %% Resumen final
print("\n" + "=" * 65)
print("RESUMEN EJECUTIVO — COMPARATIVA FINAL")
print("=" * 65)

mejor_mape = min(resultados, key=lambda r: r["MAPE"])
mejor_r2   = max(resultados, key=lambda r: r["R2"])

print(f"\n  Mejor MAPE:  {mejor_mape['Modelo']}  ->  {mejor_mape['MAPE']:.2f}%")
print(f"  Mejor R2:    {mejor_r2['Modelo']}  ->  {mejor_r2['R2']:.4f}")
print(f"\n  Benchmark Oporto et al. (2024): {MAPE_BENCH}%")

for r in resultados:
    estado = "SUPERA" if r["MAPE"] < MAPE_BENCH else "NO SUPERA"
    print(f"    {r['Modelo']:<45} MAPE={r['MAPE']:.2f}%  [{estado}]")

print(f"\n  Archivos generados:")
print(f"    {out1}")
print(f"    {out2}")
print(f"    {out3}")
print(f"    {csv_path}")

print("\n" + "=" * 65)
print("[OK] COMPARATIVA COMPLETADA")
print("=" * 65)
