# =============================================================================
# 07_VALIDACION_PREDICCION.py
# =============================================================================
# Validacion de predicciones del Modelo M3 (XGBoost - Fuente BCRP)
#
# CRISP-DM: Fase 5 - Evaluacion y Despliegue
#
# Objetivo: verificar que el modelo genera predicciones coherentes con
#           la realidad del mercado inmobiliario de Lima Metropolitana,
#           usando casos de prueba conocidos y una comparativa con
#           precios reales del Test Set (2024-2025).
#
# Fuente de entrenamiento: BCRP 2016-2023 (fuente unica oficial)
# Input:  xgboost_venta_ampliado.pkl
#         features_metadata.json
#         test.csv (precios reales para comparar)
#         train.csv (referencia de distribucion por distrito)
# Output: tabla de validacion en consola + graficos SHAP waterfall
# =============================================================================

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

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURACION
# =============================================================================
BASE_DIR    = r"d:\NuevaCarpetaLool\python_modelo_tesis"
DATA_DIR    = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR  = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports", "figures")
os.makedirs(REPORTS_DIR, exist_ok=True)

MAPE_V2 = 15.04  # MAPE oficial del modelo V2 en test set (2024-2025)

# IPC Lima Metropolitana — factor de reconversion a Soles Nominales 2026
# Fuente: INEI/BCRP serie PN01270PM, Base Dic 2009 = 100
# Valor exacto del Excel BCRP: 169.1771184 -> redondeado a 169.18
# Ultimo periodo oficial disponible al momento del estudio: Q1-2026
# Configurado como parametro en config/model_config.json para actualizacion
# sin necesidad de re-desplegar el servicio.
IPC_2026_Q1 = 169.18

# =============================================================================
# 1. CARGAR MODELO Y METADATA
# =============================================================================
print("=" * 65)
print("  VALIDACION DE PREDICCIONES - Modelo V2 (BCRP 2016-2025)")
print("=" * 65)

model_path = os.path.join(MODELS_DIR, "xgboost_venta_v2.pkl")
modelo = joblib.load(model_path)
print(f"\n[OK] Modelo cargado: {os.path.basename(model_path)}")

with open(os.path.join(DATA_DIR, "features_metadata.json")) as f:
    metadata = json.load(f)

FEATURE_COLS = metadata["features"]
ENCODING_MAP = metadata["encoding_map_distrito"]
MEDIA_GLOBAL = metadata["media_global_target"]

print(f"[OK] Features cargadas: {len(FEATURE_COLS)}")
print(f"[OK] Distritos con encoding: {len(ENCODING_MAP)}")
print(f"\nDistritos disponibles:")
for d in sorted(ENCODING_MAP.keys()):
    print(f"  - {d}  (precio medio train: S/. {ENCODING_MAP[d]:,.0f})")

# =============================================================================
# 2. TABLA CONTEXTUAL POR DISTRITO (NSE, criminalidad, distancias)
# =============================================================================
df_train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))

CONTEXT_COLS = [
    "pct_NSE_A", "pct_NSE_B", "pct_NSE_C", "pct_NSE_D", "pct_NSE_E",
    "tasa_robo", "tasa_hurto",
    "poblacion_proyectada", "area_distrito_km2", "distancia_centro_km",
    "dist_colegio_km", "dist_hospital_km", "dist_estacion_transporte_km",
    "dist_centro_comercial_km", "dist_parque_km", "dist_universidad_km",
    "densidad_hab_km2",
]

# Usar el CSV precomputado de contexto distrital
# (generado desde el Excel maestro, filtrando train 2016-2023)
CONTEXT_REF_PATH = os.path.join(DATA_DIR, "distrito_contexto_ref.csv")
if os.path.exists(CONTEXT_REF_PATH):
    distrito_context = pd.read_csv(CONTEXT_REF_PATH)
    print(f"[OK] Referencia contextual cargada: {CONTEXT_REF_PATH}")
else:
    # Fallback: leer desde el Excel maestro (mas lento)
    print("[INFO] distrito_contexto_ref.csv no encontrado. Leyendo desde Excel...")
    EXCEL_PATH = os.path.join(BASE_DIR, "data", "raw", "dataset_entrenamiento_final_imputado.xlsx")
    df_raw = pd.read_excel(EXCEL_PATH)
    df_train_raw = df_raw[df_raw["Anio"] <= 2023]
    distrito_context = (
        df_train_raw.sort_values("Anio", ascending=False)
        .groupby("Distrito")[CONTEXT_COLS]
        .first()
        .reset_index()
    )
    distrito_context.to_csv(CONTEXT_REF_PATH, index=False)
    print(f"[OK] CSV de referencia generado y guardado en {CONTEXT_REF_PATH}")

# Fallback de medias globales (para distritos fuera del encoding)
global_means = {c: distrito_context[c].mean() for c in CONTEXT_COLS}
print(f"[OK] Referencia contextual para {len(distrito_context)} distritos")

# =============================================================================
# 3. FUNCION PRINCIPAL DE PREDICCION
# =============================================================================

def predecir(distrito, superficie, habitaciones, banios, garajes, piso,
             antiguedad, vista_exterior=1, anio=2025, trimestre=2,
             precio_nominal_referencia=None, verbose=True):
    """Predice el precio de un inmueble en Lima Metropolitana (Soles Constantes)."""

    if distrito not in ENCODING_MAP:
        disponibles = sorted(ENCODING_MAP.keys())
        sugeridos = [d for d in disponibles if distrito.lower() in d.lower()]
        raise ValueError(
            f"Distrito '{distrito}' no encontrado.\n"
            f"Sugeridos: {sugeridos}\nDisponibles: {disponibles}"
        )

    ctx = distrito_context[distrito_context["Distrito"] == distrito]
    ctx_vals = ctx.iloc[0][CONTEXT_COLS].to_dict() if not ctx.empty else global_means

    fila = {
        "Anio":                        anio,
        "Trimestre":                   trimestre,
        "Superficie":                  superficie,
        "Habitaciones":                habitaciones,
        "Banios":                      banios,
        "Garajes":                     garajes,
        "Piso":                        piso,
        "Vista_Exterior":              vista_exterior,
        "Antiguedad":                  antiguedad,
        "pct_NSE_A":                   ctx_vals["pct_NSE_A"],
        "pct_NSE_B":                   ctx_vals["pct_NSE_B"],
        "pct_NSE_C":                   ctx_vals["pct_NSE_C"],
        "pct_NSE_D":                   ctx_vals["pct_NSE_D"],
        "pct_NSE_E":                   ctx_vals["pct_NSE_E"],
        "tasa_robo":                   ctx_vals["tasa_robo"],
        "tasa_hurto":                  ctx_vals["tasa_hurto"],
        "poblacion_proyectada":        ctx_vals["poblacion_proyectada"],
        "area_distrito_km2":           ctx_vals["area_distrito_km2"],
        "distancia_centro_km":         ctx_vals["distancia_centro_km"],
        "dist_colegio_km":             ctx_vals["dist_colegio_km"],
        "dist_hospital_km":            ctx_vals["dist_hospital_km"],
        "dist_estacion_transporte_km": ctx_vals["dist_estacion_transporte_km"],
        "dist_centro_comercial_km":    ctx_vals["dist_centro_comercial_km"],
        "dist_parque_km":              ctx_vals["dist_parque_km"],
        "dist_universidad_km":         ctx_vals["dist_universidad_km"],
        "densidad_hab_km2":            ctx_vals["densidad_hab_km2"],
        "periodo_numerico":            anio * 4 + trimestre,
        "m2_por_habitacion":           superficie / (habitaciones + 1),
        "ratio_banios_hab":            banios / (habitaciones + 0.1),
        "tiene_garaje":                1 if garajes > 0 else 0,
        "es_piso_alto":                1 if piso >= 8 else 0,
        "superficie_cuadrado":         (superficie / 100) ** 2,
        "distrito_encoded":            ENCODING_MAP[distrito],
    }

    X = pd.DataFrame([fila])[FEATURE_COLS]
    pred_const = np.expm1(modelo.predict(X)[0])  # Soles CONSTANTES (Base Dic 2009 = 100)

    # Reconversion a soles NOMINALES 2026 — multiplicar por factor IPC
    # Formula: Precio_Nominal = Precio_Const * (IPC_actual / 100)
    pred_nominal = pred_const * (IPC_2026_Q1 / 100)

    pred_m2_const   = pred_const   / superficie
    pred_m2_nominal = pred_nominal / superficie

    # Intervalos de confianza en ambas unidades
    ic_inf_const   = pred_const   * (1 - MAPE_V2 / 100)
    ic_sup_const   = pred_const   * (1 + MAPE_V2 / 100)
    ic_inf_nominal = pred_nominal * (1 - MAPE_V2 / 100)
    ic_sup_nominal = pred_nominal * (1 + MAPE_V2 / 100)

    error_pct = None
    if precio_nominal_referencia is not None:
        # precio_nominal_referencia viene en soles constantes (de test.csv)
        # -> comparar contra pred_const para MAPE correcto
        error_pct = abs(precio_nominal_referencia - pred_const) / precio_nominal_referencia * 100

    if verbose:
        print(f"\n{'─'*65}")
        print(f"  {habitaciones}D/{banios}B · {superficie}m2 · {distrito} · Piso {piso} · {antiguedad} anios")
        print(f"{'─'*65}")
        print(f"  [MODELO — Soles Constantes Base Dic2009]")
        print(f"  Precio Predicho (Const.):  S/. {pred_const:>12,.0f}")
        print(f"  Precio por m2  (Const.):  S/. {pred_m2_const:>12,.0f} /m2")
        print(f"  Intervalo +-{MAPE_V2}%:        S/. {ic_inf_const:,.0f} - S/. {ic_sup_const:,.0f}")
        print()
        print(f"  [USUARIO FINAL — Soles Nominales 2026  x{IPC_2026_Q1/100:.4f}]")
        print(f"  Precio Predicho (Nominal): S/. {pred_nominal:>12,.0f}")
        print(f"  Precio por m2  (Nominal): S/. {pred_m2_nominal:>12,.0f} /m2")
        print(f"  Intervalo +-{MAPE_V2}%:        S/. {ic_inf_nominal:,.0f} - S/. {ic_sup_nominal:,.0f}")
        if error_pct is not None:
            ok = "OK" if error_pct < MAPE_V2 else "FUERA"
            print(f"\n  Error vs precio real (soles const.): {error_pct:.1f}%  [{ok}]")
        print(f"{'─'*65}")

    return {
        "distrito":         distrito,
        "superficie":        superficie,
        "pred_soles_const":  pred_const,
        "pred_soles_nominal":pred_nominal,
        "pred_m2_const":     pred_m2_const,
        "pred_m2_nominal":   pred_m2_nominal,
        "ic_inf_const":      ic_inf_const,
        "ic_sup_const":      ic_sup_const,
        "ic_inf_nominal":    ic_inf_nominal,
        "ic_sup_nominal":    ic_sup_nominal,
        "error_pct":         error_pct,
        "X":                 X,
    }

# =============================================================================
# 4. BLOQUE A — CASOS SINTETICOS
# =============================================================================
print("\n\n" + "=" * 65)
print("  BLOQUE A - CASOS SINTETICOS (sin precio de referencia)")
print("=" * 65)

casos_sinteticos = [
    {"nombre": "Dpto. estandar San Miguel (3D/2B/75m2)",
     "distrito": "San Miguel",   "superficie": 75,  "habitaciones": 3,
     "banios": 2, "garajes": 1,  "piso": 5,  "antiguedad": 8,  "anio": 2025},
    {"nombre": "Flat Miraflores (1D/1B/50m2)",
     "distrito": "Miraflores",   "superficie": 50,  "habitaciones": 1,
     "banios": 1, "garajes": 0,  "piso": 3,  "antiguedad": 15, "anio": 2025},
    {"nombre": "Dpto. premium San Isidro (3D/3B/150m2)",
     "distrito": "San Isidro",   "superficie": 150, "habitaciones": 3,
     "banios": 3, "garajes": 2,  "piso": 12, "antiguedad": 3,  "anio": 2025},
    {"nombre": "Dpto. economico Comas (2D/1B/55m2)",
     "distrito": "Comas",        "superficie": 55,  "habitaciones": 2,
     "banios": 1, "garajes": 0,  "piso": 2,  "antiguedad": 20, "anio": 2025},
    {"nombre": "Dpto. Surco (3D/2B/100m2 con cochera)",
     "distrito": "Surco",        "superficie": 100, "habitaciones": 3,
     "banios": 2, "garajes": 1,  "piso": 6,  "antiguedad": 10, "anio": 2025},
]

resultados_sinteticos = []
for caso in casos_sinteticos:
    nombre = caso.pop("nombre")
    print(f"\n  {nombre}")
    r = predecir(**caso)
    r["nombre"] = nombre
    resultados_sinteticos.append(r)

# =============================================================================
# 5. BLOQUE B — VALIDACION CON PRECIOS REALES DEL TEST SET
# =============================================================================
print("\n\n" + "=" * 65)
print("  BLOQUE B - VALIDACION CON TEST SET REAL (2024-2025)")
print("=" * 65)

df_test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

distritos_muestra = [
    "Miraflores", "San Borja", "San Isidro", "Surco",
    "San Miguel", "La Molina", "Jesus Maria", "Chorrillos",
    "Ate Vitarte", "Los Olivos",
]

casos_reales = []
if "Distrito" in df_test.columns:
    for distrito in distritos_muestra:
        subset = df_test[df_test["Distrito"] == distrito]
        if subset.empty:
            continue
        mediana_precio = subset["Precio_Soles_Const"].median()
        idx = (subset["Precio_Soles_Const"] - mediana_precio).abs().idxmin()
        row = subset.loc[idx]
        casos_reales.append({
            "distrito": distrito,
            "superficie": row["Superficie"],
            "habitaciones": int(row["Habitaciones"]),
            "banios": int(row["Banios"]),
            "garajes": int(row["Garajes"]),
            "piso": int(row["Piso"]),
            "antiguedad": int(row["Antiguedad"]),
            "vista_exterior": int(row.get("Vista_Exterior", 1)),
            "anio": int(row["Anio"]),
            "trimestre": int(row["Trimestre"]),
            "precio_nominal_referencia": row["Precio_Soles_Const"],
        })
else:
    print("[WARN] Columna 'Distrito' no encontrada en test.csv. Saltando Bloque B.")

resultados_reales = []
for caso in casos_reales:
    nombre = f"{caso['distrito']} ({int(caso['superficie'])}m2 {caso['habitaciones']}D/{caso['banios']}B)"
    print(f"\n  {nombre}")
    r = predecir(**caso)
    r["nombre"] = nombre
    r["precio_real"] = caso["precio_nominal_referencia"]
    resultados_reales.append(r)

# =============================================================================
# 6. TABLA RESUMEN
# =============================================================================
if resultados_reales:
    print("\n\n" + "=" * 65)
    print("  TABLA RESUMEN - BLOQUE B")
    print("=" * 65)
    print(f"{'Caso':<42} {'Real S/.':>12} {'Pred S/.':>12} {'Error%':>8}")
    print("-" * 78)

    errores = []
    for r in resultados_reales:
        real = r.get("precio_real", 0)
        pred = r["pred_soles_const"]
        err  = r["error_pct"] or 0
        ok   = "OK" if err < MAPE_V2 else "!!"
        errores.append(err)
        print(f"{r['nombre']:<42} {real:>12,.0f} {pred:>12,.0f} {err:>6.1f}% {ok}")

    print("-" * 78)
    mape_muestra = np.mean(errores)
    print(f"  MAPE promedio en muestra ({len(errores)} casos):  {mape_muestra:.2f}%")
    print(f"  MAPE oficial V2 (23,741 obs test):          {MAPE_V2:.2f}%")

    # Grafico
    reales = [r["precio_real"] for r in resultados_reales]
    preds  = [r["pred_soles_const"] for r in resultados_reales]
    names  = [r["nombre"].split(" (")[0] for r in resultados_reales]

    fig, ax = plt.subplots(figsize=(9, 9))
    mn = min(min(reales), min(preds)) * 0.9
    mx = max(max(reales), max(preds)) * 1.1
    ax.plot([mn, mx], [mn, mx], "r--", lw=1.5, label="Prediccion perfecta")
    for real, pred, name in zip(reales, preds, names):
        color = "#2ecc71" if abs(pred - real) / real < MAPE_V2 / 100 else "#e74c3c"
        ax.scatter(real, pred, color=color, s=130, zorder=5)
        ax.annotate(name, (real, pred), textcoords="offset points",
                    xytext=(7, 3), fontsize=7.5, alpha=0.9)
    ax.set_xlabel("Precio Real (Soles Constantes)", fontsize=11)
    ax.set_ylabel("Precio Predicho (Soles Constantes)", fontsize=11)
    ax.set_title(f"Validacion: Real vs Predicho - Muestra Test Set\n"
                 f"MAPE muestra: {mape_muestra:.2f}% | MAPE modelo: {MAPE_V2}%", fontsize=11)
    from matplotlib.lines import Line2D
    leyenda_extra = [
        Line2D([0],[0], marker='o', color='w', markerfacecolor='#2ecc71',
               markersize=10, label=f'Error < {MAPE_V2}% [OK]'),
        Line2D([0],[0], marker='o', color='w', markerfacecolor='#e74c3c',
               markersize=10, label=f'Error >= {MAPE_V2}% [!!]'),
    ]
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles + leyenda_extra, fontsize=9)
    plt.tight_layout()
    out = os.path.join(REPORTS_DIR, "validacion_real_vs_predicho.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n[OK] Grafico guardado: {out}")

# =============================================================================
# 7. SHAP WATERFALL — Casos sinteticos
# =============================================================================
print("\n\n" + "=" * 65)
print("  SHAP WATERFALL - Explicacion casos sinteticos")
print("=" * 65)

try:
    import shap
    explainer = shap.TreeExplainer(modelo)
    for r in resultados_sinteticos:
        sv = explainer(r["X"])
        fig, ax = plt.subplots(figsize=(11, 7))
        shap.plots.waterfall(sv[0], max_display=15, show=False)
        plt.title(f"SHAP: {r['nombre']}\nPrediccion: S/. {r['pred_soles_const']:,.0f}",
                  fontsize=10, pad=12)
        plt.tight_layout()
        slug = r["nombre"].replace(" ", "_").replace("/","").replace("(","").replace(")","")[:45]
        out = os.path.join(REPORTS_DIR, f"shap_waterfall_{slug}.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  [OK] {r['nombre']} -> {os.path.basename(out)}")
except ImportError:
    print("  [SKIP] shap no instalado. pip install shap")

# =============================================================================
# 8. RESUMEN FINAL
# =============================================================================
print("\n\n" + "=" * 65)
print("  RESUMEN FINAL")
print("=" * 65)
print(f"  Modelo:              xgboost_venta_v2.pkl (V2)")
print(f"  MAPE oficial:        {MAPE_V2}%  (test set 23,741 obs)")
if resultados_reales:
    print(f"  MAPE en muestra:     {mape_muestra:.2f}%  ({len(resultados_reales)} casos)")
print()
print("  Interpretacion del error esperado:")
print(f"    S/. 400,000  ->  margen +-S/. {400000*MAPE_V2/100:,.0f}")
print(f"    S/. 600,000  ->  margen +-S/. {600000*MAPE_V2/100:,.0f}")
print(f"    S/. 200,000  ->  margen +-S/. {200000*MAPE_V2/100:,.0f}")
print()
if resultados_reales:
    aprobado = mape_muestra < 17.89
    print(f"  Estado: {'APROBADO' if aprobado else 'REVISAR'} (benchmark: 17.89%)")
print("=" * 65)
