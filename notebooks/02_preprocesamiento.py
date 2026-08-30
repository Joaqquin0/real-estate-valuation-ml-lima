# =============================================================================
# 02_PREPROCESAMIENTO.py
# =============================================================================
# Preparación de Datos — Modelo Predictivo XGBoost
# Valoración Inmobiliaria en Lima Metropolitana
#
# CRISP-DM: Fase 2 — Preparación de los Datos
# Historia Técnica: TH-01 (Preprocesar datos para modelo de venta)
#
# Decisiones basadas en el EDA (01_EDA_exploratorio.py):
#   - 1,036 filas con NSE nulo → XGBoost los maneja nativamente
#   - Eliminar IPC (corr 0.97 con Anio + precio ya deflactado)
#   - Eliminar tasa_denuncias (corr 0.93 con tasa_hurto, es su suma)
#   - Eliminar Tipo_Cambio (precio ya en soles constantes)
#   - Log-transform del target (skewness 5.06)
#   - Target Encoding para Distrito (26 categorías)
#   - Split temporal: Train 2016-2023 / Test 2024-2025
# =============================================================================

# %% [markdown]
# # 🔧 Preprocesamiento de Datos
# ## Pipeline reproducible para el modelo XGBoost
# **Input**: `dataset_entrenamiento_final_imputado.xlsx` (67,914 × 40)
# **Output**: `train.csv`, `test.csv` listos para entrenar

# %% Instalación (solo en Colab)
# !pip install -q openpyxl

# %% Importaciones
import pandas as pd
import numpy as np
import json
import os
import warnings

warnings.filterwarnings("ignore")

# %% [markdown]
# ## 1. Configuración

# %% Configuración centralizada
# =============================================
# Rutas locales
# =============================================
BASE_DIR = r"d:\NuevaCarpetaLool\python_modelo_tesis"
DATASET_PATH = os.path.join(BASE_DIR, "data", "raw", "dataset_entrenamiento_final_imputado.xlsx")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "processed")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Variable target
TARGET = "Precio_Soles_Const"

# Columnas a ELIMINAR (no entran al modelo)
COLS_ELIMINAR = [
    # Identificadores
    "ID",
    "Ubigeo",
    # Targets alternativos (son versiones del precio que no usamos)
    "Precio_Dolares",
    "Precio_Soles",
    "precio_dolares_m2",
    # Multicolinealidad detectada en EDA
    "IPC",              # corr 0.969 con Anio + precio ya deflactado por IPC
    "Tipo_Cambio",      # precio ya en soles constantes, TC absorbido
    "tasa_denuncias",   # corr 0.928 con tasa_hurto (es su suma con tasa_robo)
]

# Flags de imputación (guardar para análisis de sensibilidad, no como features)
COLS_FLAGS = [
    "nse_imputado",
    "tasas_criminalidad_imputada",
    "poblacion_imputada",
]

# Split temporal
TRAIN_YEARS = list(range(2016, 2024))  # 2016–2023
TEST_YEARS = list(range(2024, 2026))   # 2024–2025

# Reproducibilidad
RANDOM_SEED = 42

# %% [markdown]
# ## 2. Carga de Datos

# %% Cargar dataset
df = pd.read_excel(DATASET_PATH)
print(f"Dataset original: {df.shape[0]:,} filas x {df.shape[1]} columnas")

# %% [markdown]
# ## 3. Separar Flags de Imputación
# Los flags se guardan aparte para análisis de sensibilidad posterior
# (comparar métricas con/sin datos imputados), pero NO entran como features.

# %% Guardar flags
cols_flags_existentes = [c for c in COLS_FLAGS if c in df.columns]
df_flags = df[["Anio", "Distrito"] + cols_flags_existentes].copy()
print(f"Flags de imputación separados: {cols_flags_existentes}")

# %% [markdown]
# ## 4. Eliminar Columnas No Predictivas
#
# **Razones de eliminación:**
# - `ID`, `Ubigeo`: identificadores sin valor predictivo
# - `Precio_Dolares`, `Precio_Soles`, `precio_dolares_m2`: son el target en otras unidades
# - `IPC`: correlación 0.969 con `Anio` + el target ya está deflactado por IPC (circular)
# - `Tipo_Cambio`: el precio ya está en soles constantes
# - `tasa_denuncias`: correlación 0.928 con `tasa_hurto` (es la suma de robo + hurto)
# - Flags de imputación: solo para análisis de sensibilidad

# %% Eliminar columnas
cols_a_eliminar = [c for c in COLS_ELIMINAR + COLS_FLAGS if c in df.columns]
df = df.drop(columns=cols_a_eliminar)

print(f"\nColumnas eliminadas ({len(cols_a_eliminar)}):")
for c in cols_a_eliminar:
    print(f"  [X] {c}")
print(f"\nDataset despues de limpieza: {df.shape[0]:,} filas x {df.shape[1]} columnas")
print(f"Columnas restantes: {list(df.columns)}")

# %% [markdown]
# ## 5. Verificación de Tipos de Datos

# %% Verificar tipos
print("\n--- TIPOS DE DATOS ACTUALES ---")
print(df.dtypes.to_string())

# Verificar que Vista_Exterior sea numérica (0/1)
if "Vista_Exterior" in df.columns:
    vals_unicos = df["Vista_Exterior"].unique()
    print(f"\nVista_Exterior valores unicos: {sorted(vals_unicos)}")
    if df["Vista_Exterior"].dtype == "object":
        print("  -> Convirtiendo Vista_Exterior a numerico...")
        df["Vista_Exterior"] = df["Vista_Exterior"].map({"Si": 1, "No": 0, True: 1, False: 0})

# %% [markdown]
# ## 6. Feature Engineering
# Features de confort espacial, calidad y relaciones no lineales (Paso A)

# %% 6.1 Crear periodo_numerico (tendencia temporal continua)
df["periodo_numerico"] = df["Anio"] * 4 + df["Trimestre"]
print(f"[OK] Feature creada: 'periodo_numerico' [rango: {df['periodo_numerico'].min()} - {df['periodo_numerico'].max()}]")

# %% 6.2 Ratios de Confort y Arquitectura (adecuados para tasaciones BCRP)
df["m2_por_habitacion"] = df["Superficie"] / (df["Habitaciones"] + 1)
df["ratio_banios_hab"]  = df["Banios"] / (df["Habitaciones"] + 0.1)
df["tiene_garaje"]      = (df["Garajes"] > 0).astype(int)
df["es_piso_alto"]      = (df["Piso"] >= 8).astype(int)
df["superficie_cuadrado"] = (df["Superficie"] / 100.0) ** 2  # Escalado para estabilidad

print("[OK] Nuevas features de calidad creadas: m2_por_habitacion, ratio_banios_hab, tiene_garaje, es_piso_alto, superficie_cuadrado")

# %% [markdown]
# ## 7. Split Temporal (ANTES del Target Encoding)
#
# ⚠️ **Importante**: El split se hace ANTES del Target Encoding para que el
# encoding se calcule SOLO con datos de train. Si se calcula con todo el
# dataset, habría data leakage (el modelo "vería" información del futuro).

# %% Split temporal
df_train = df[df["Anio"].isin(TRAIN_YEARS)].copy()
df_test = df[df["Anio"].isin(TEST_YEARS)].copy()

# También separar los flags para análisis de sensibilidad
df_flags_train = df_flags[df_flags["Anio"].isin(TRAIN_YEARS)].copy()
df_flags_test = df_flags[df_flags["Anio"].isin(TEST_YEARS)].copy()

# %% 7.1 Limpieza de Outliers Extremos SOLO en Train (Paso B)
# Nota metodológica: Se recortan colas extremas (p0.5 y p99.5) SOLO en train para
# evitar que XGBoost gaste ramas en casos atípicos extremos, preservando el test intacto.
n_train_antes = len(df_train)
p_sup_min, p_sup_max = df_train["Superficie"].quantile(0.005), df_train["Superficie"].quantile(0.995)
p_pre_min, p_pre_max = df_train[TARGET].quantile(0.005), df_train[TARGET].quantile(0.995)

mask_train_valid = (
    (df_train["Superficie"] >= p_sup_min) & (df_train["Superficie"] <= p_sup_max) &
    (df_train[TARGET] >= p_pre_min) & (df_train[TARGET] <= p_pre_max)
)
df_train = df_train[mask_train_valid].copy()
df_flags_train = df_flags_train.loc[df_train.index].copy()

print(f"\n--- LIMPIEZA DE OUTLIERS EN TRAIN (0.5% - 99.5%) ---")
print(f"  Superficie valida en train: [{p_sup_min:.1f} m2 - {p_sup_max:.1f} m2]")
print(f"  Precio valido en train:     [S/. {p_pre_min:,.0f} - S/. {p_pre_max:,.0f}]")
print(f"  Filas Train: {n_train_antes:,} -> {len(df_train):,} ({n_train_antes - len(df_train)} filas atipicas filtradas)")

total = len(df_train) + len(df_test)
print(f"\n--- SPLIT TEMPORAL FINAL ---")
print(f"  Train ({min(TRAIN_YEARS)}-{max(TRAIN_YEARS)}): {len(df_train):,} filas ({len(df_train)/total*100:.1f}%)")
print(f"  Test  ({min(TEST_YEARS)}-{max(TEST_YEARS)}): {len(df_test):,} filas ({len(df_test)/total*100:.1f}%)")

# Verificar no solapamiento
assert len(set(df_train["Anio"].unique()) & set(df_test["Anio"].unique())) == 0, \
    "ERROR: Hay anios solapados entre train y test!"
print("  [OK] Sin solapamiento temporal verificado")

# %% [markdown]
# ## 8. Target Encoding para Distrito
#
# Se usa **Target Encoding con suavizado bayesiano** en lugar de One-Hot
# porque hay 26 distritos (One-Hot agregaría 25 columnas sparse).
#
# **Suavizado bayesiano**: mezcla la media del distrito con la media global,
# ponderada por el número de observaciones. Distritos con pocas observaciones
# se "acercan" a la media global, evitando overfitting.
#
# **Calculado SOLO con datos de train** para evitar data leakage.

# %% Target Encoding
COL_DISTRITO = "Distrito"
SMOOTHING = 10.0

# Calcular estadísticas SOLO en train
media_global = df_train[TARGET].mean()
stats = df_train.groupby(COL_DISTRITO)[TARGET].agg(["mean", "count"])
stats["distrito_encoded"] = (
    (stats["count"] * stats["mean"] + SMOOTHING * media_global)
    / (stats["count"] + SMOOTHING)
)
encoding_map = stats["distrito_encoded"].to_dict()

# Aplicar a ambos sets
df_train["distrito_encoded"] = df_train[COL_DISTRITO].map(encoding_map)
df_test["distrito_encoded"] = df_test[COL_DISTRITO].map(encoding_map)

# Distritos no vistos en train → media global
n_unseen = df_test["distrito_encoded"].isna().sum()
if n_unseen > 0:
    print(f"[WARN] {n_unseen} registros en test con distrito no visto en train -> media global")
    df_test["distrito_encoded"] = df_test["distrito_encoded"].fillna(media_global)

print(f"\n--- TARGET ENCODING: DISTRITO ---")
print(f"  Distritos codificados: {len(encoding_map)}")
print(f"  Media global (train): S/. {media_global:,.0f}")
print(f"  Smoothing factor: {SMOOTHING}")

# Mostrar encoding
encoding_df = pd.DataFrame({
    "Distrito": encoding_map.keys(),
    "Encoding (S/.)": [f"S/. {v:,.0f}" for v in encoding_map.values()],
    "Observaciones": stats["count"].values,
}).sort_values("Encoding (S/.)", ascending=False)
print(f"\n{encoding_df.to_string(index=False)}")

# Eliminar columna texto original de Distrito
# (Guardar para referencia antes de eliminar)
distrito_train = df_train[COL_DISTRITO].copy()
distrito_test = df_test[COL_DISTRITO].copy()

df_train = df_train.drop(columns=[COL_DISTRITO])
df_test = df_test.drop(columns=[COL_DISTRITO])

# %% [markdown]
# ## 9. Log-Transform del Target
#
# El target tiene skewness de **5.06** (fuertemente sesgado a la derecha).
# XGBoost trabaja mejor con targets más simétricos.
#
# Aplicamos `log1p(y)` para entrenar, y `expm1(pred)` para volver a escala real.
# Esto comprime los valores extremos y mejora las predicciones.

# %% Log-transform
print(f"\n--- LOG-TRANSFORM DEL TARGET ---")
print(f"  Antes  - Skewness: {df_train[TARGET].skew():.2f}")

df_train[f"{TARGET}_log"] = np.log1p(df_train[TARGET])
df_test[f"{TARGET}_log"] = np.log1p(df_test[TARGET])

print(f"  Despues - Skewness: {df_train[f'{TARGET}_log'].skew():.2f}")
print(f"  [OK] Se usara '{TARGET}_log' como target para entrenamiento")
print(f"  [OK] Se aplicara np.expm1() a las predicciones para volver a S/. reales")

# %% [markdown]
# ## 10. Verificación Final del Dataset

# %% Resumen final
print("\n" + "=" * 60)
print("RESUMEN DEL PREPROCESAMIENTO")
print("=" * 60)

print(f"\nDIMENSIONES FINALES:")
print(f"  Train: {df_train.shape[0]:,} filas x {df_train.shape[1]} columnas")
print(f"  Test:  {df_test.shape[0]:,} filas x {df_test.shape[1]} columnas")

# Features que entran al modelo (todo excepto targets)
target_cols = [TARGET, f"{TARGET}_log"]
feature_cols = [c for c in df_train.columns if c not in target_cols]

print(f"\nFEATURES ({len(feature_cols)}):")
for i, col in enumerate(feature_cols, 1):
    nulos_train = df_train[col].isna().sum()
    nulos_test = df_test[col].isna().sum()
    nulo_str = f" [WARN nulos: {nulos_train}/{nulos_test}]" if (nulos_train + nulos_test) > 0 else ""
    print(f"  {i:2d}. {col}{nulo_str}")

print(f"\nTARGETS:")
print(f"  - {TARGET} (escala real en S/.)")
print(f"  - {TARGET}_log (log-transformado, usar para entrenar)")

# Nulos finales
nulos_train = df_train[feature_cols].isna().sum().sum()
nulos_test = df_test[feature_cols].isna().sum().sum()
print(f"\nNULOS EN FEATURES:")
print(f"  Train: {nulos_train:,} nulos totales")
print(f"  Test:  {nulos_test:,} nulos totales")
if nulos_train > 0 or nulos_test > 0:
    print(f"  -> XGBoost maneja nulos nativamente (no se imputan)")

# %% [markdown]
# ## 11. Guardar Datasets Procesados

# %% Guardar train y test
train_path = os.path.join(OUTPUT_DIR, "train.csv")
test_path = os.path.join(OUTPUT_DIR, "test.csv")

df_train.to_csv(train_path, index=False)
df_test.to_csv(test_path, index=False)

print(f"\nARCHIVOS GUARDADOS:")
print(f"  {train_path} ({df_train.shape[0]:,} x {df_train.shape[1]})")
print(f"  {test_path} ({df_test.shape[0]:,} x {df_test.shape[1]})")

# %% Guardar metadata de features (para reproducibilidad)
metadata = {
    "target": TARGET,
    "target_log": f"{TARGET}_log",
    "features": feature_cols,
    "n_features": len(feature_cols),
    "encoding_map_distrito": {k: float(v) for k, v in encoding_map.items()},
    "smoothing_factor": SMOOTHING,
    "media_global_target": float(media_global),
    "train_years": TRAIN_YEARS,
    "test_years": TEST_YEARS,
    "train_rows": len(df_train),
    "test_rows": len(df_test),
    "columnas_eliminadas": cols_a_eliminar,
    "random_seed": RANDOM_SEED,
    "log_transform_aplicado": True,
    "skewness_original": float(df_test[TARGET].skew()),
}

metadata_path = os.path.join(OUTPUT_DIR, "features_metadata.json")
with open(metadata_path, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

print(f"  {metadata_path} (metadata de features y encoding)")

# %% Guardar flags de imputación (para análisis de sensibilidad)
flags_path = os.path.join(OUTPUT_DIR, "flags_imputacion.csv")
df_flags_combined = pd.concat([
    df_flags_train.assign(split="train"),
    df_flags_test.assign(split="test"),
])
df_flags_combined.to_csv(flags_path, index=False)
print(f"  {flags_path} (flags para análisis de sensibilidad)")

# %% Guardar distrito como referencia
ref_path = os.path.join(OUTPUT_DIR, "distrito_referencia.csv")
pd.DataFrame({
    "split": ["train"] * len(distrito_train) + ["test"] * len(distrito_test),
    "Distrito": pd.concat([distrito_train, distrito_test]).values,
}).to_csv(ref_path, index=False)
print(f"  {ref_path} (nombres de distrito para evaluación)")

# %% [markdown]
# ## 12. Verificación de Integridad

# %% Verificar que los archivos se pueden releer
print("\nVERIFICACION DE INTEGRIDAD:")

df_train_check = pd.read_csv(train_path)
df_test_check = pd.read_csv(test_path)

print(f"  Train releido: {df_train_check.shape} [OK]")
print(f"  Test releido: {df_test_check.shape} [OK]")

# Verificar que no hay data leakage temporal
years_train = set(df_train_check["Anio"].unique())
years_test = set(df_test_check["Anio"].unique())
overlap = years_train & years_test
assert len(overlap) == 0, f"DATA LEAKAGE! Anios en ambos sets: {overlap}"
print(f"  Sin data leakage temporal [OK]")

# Verificar que el target log existe
assert f"{TARGET}_log" in df_train_check.columns, "Falta target log en train"
assert f"{TARGET}_log" in df_test_check.columns, "Falta target log en test"
print(f"  Target log-transformado presente [OK]")

print(f"\n{'=' * 60}")
print(f"[OK] PREPROCESAMIENTO COMPLETADO")
print(f"{'=' * 60}")
print(f"\nSiguiente paso: 03_entrenamiento_xgboost.py")
