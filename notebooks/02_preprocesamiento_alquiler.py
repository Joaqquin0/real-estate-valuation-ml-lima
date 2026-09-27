# =============================================================================
# 02_PREPROCESAMIENTO_ALQUILER.py
# =============================================================================
# Preparación de Datos — Modelo Predictivo XGBoost para Alquiler
# Valoración Inmobiliaria en Lima Metropolitana
#
# CRISP-DM: Fase 2 — Preparación de los Datos
#
# Decisiones metodológicas:
#   - Merge dinámico con tabla maestra de contexto: distrito_anio_contexto.csv
#   - 22 distritos representativos compartidos con el modelo de venta
#   - Imputación de variables BCRP con nulos/ND (Piso, Vista, Antigüedad)
#   - Ratios de confort y arquitectura (m2_por_habitacion, tiene_garaje, etc.)
#   - Eliminación de multicolinealidad (IPC, Tipo_Cambio, tasa_denuncias)
#   - Split temporal: Train 2016-2023 / Test 2024-2025
#   - Target Encoding bayesiano para Distrito calculado SOLO en train
#   - Log-transform del target Alquiler_Soles_Const
# =============================================================================

import pandas as pd
import numpy as np
import json
import os
import warnings

warnings.filterwarnings("ignore")

# %% 1. Configuración de Rutas y Constantes
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, "dataset_entrenamineto_alquiler_2025.xlsx")
CONTEXT_PATH = os.path.join(BASE_DIR, "data", "processed", "distrito_anio_contexto.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "processed")

os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET = "Alquiler_Soles_Const"

# Columnas a eliminar del modelo final
COLS_ELIMINAR = [
    "Alquiler_Dolares",
    "Alquiler_Soles",
    "Tipo_Cambio",
    "IPC",
    "tasa_denuncias",
]

COLS_FLAGS = [
    "nse_imputado",
    "tasas_criminalidad_imputada",
    "poblacion_imputada",
]

TRAIN_YEARS = list(range(2016, 2024))  # 2016–2023
TEST_YEARS = list(range(2024, 2026))   # 2024–2025
RANDOM_SEED = 42
SMOOTHING = 10.0

print("=" * 60)
print("INICIANDO PREPROCESAMIENTO DE ALQUILER")
print("=" * 60)

# %% 2. Carga y Normalización del Dataset BCRP de Alquiler
print("\n--- 1. Carga de datos BCRP ---")
df = pd.read_excel(DATASET_PATH)
print(f"Dataset BCRP Alquiler crudo: {df.shape[0]:,} filas x {df.shape[1]} columnas")

# Normalizar nombres de columnas
df = df.rename(columns={
    df.columns[1]: "Anio",
    df.columns[2]: "Trimestre",
    df.columns[3]: "Alquiler_Dolares",
    df.columns[4]: "Tipo_Cambio",
    df.columns[5]: "IPC",
    df.columns[6]: "Alquiler_Soles",
    df.columns[7]: TARGET,
    df.columns[8]: "Distrito",
    df.columns[9]: "Superficie",
    df.columns[10]: "Habitaciones",
    df.columns[11]: "Banios",
    df.columns[12]: "Garajes",
    df.columns[13]: "Piso",
    df.columns[14]: "Vista_Exterior",
    df.columns[15]: "Antiguedad",
})

# Descartar columna índice de excel si existe
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

# %% 3. Filtrado de Distritos Atípicos (<= 1 registro)
conteos = df["Distrito"].value_counts()
distritos_invalidos = conteos[conteos <= 1].index.tolist()
if distritos_invalidos:
    print(f"Excluyendo distritos con <= 1 registro: {distritos_invalidos}")
    df = df[~df["Distrito"].isin(distritos_invalidos)].copy()
print(f"Filas tras filtrar distritos atípicos: {df.shape[0]:,} en {df['Distrito'].nunique()} distritos")

# %% 4. Limpieza e Imputación de Columnas Físicas BCRP
print("\n--- 2. Limpieza de columnas físicas con nulos / ND ---")

cols_fisicas = ["Piso", "Vista_Exterior", "Antiguedad", "Habitaciones", "Banios", "Garajes", "Superficie", TARGET]

# Convertir valores tipo "N/D", "ND", espacios a NaN y forzar float
for col in cols_fisicas:
    if col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].replace(["N/D", "ND", "n/d", "nd", " ", "", "None"], np.nan)
        df[col] = pd.to_numeric(df[col], errors="coerce")

# 4.1 Imputación de Vista_Exterior (Moda por distrito, fallback 1.0)
print(f"Nulos en Vista_Exterior antes de imputar: {df['Vista_Exterior'].isna().sum():,}")
moda_vista_distrito = df.groupby("Distrito")["Vista_Exterior"].apply(
    lambda s: s.mode().iloc[0] if not s.mode().empty else 1.0
).to_dict()
df["Vista_Exterior"] = df["Vista_Exterior"].fillna(df["Distrito"].map(moda_vista_distrito)).fillna(1.0)

# 4.2 Imputación de Piso (Mediana por distrito; si piso == 0 o NaN, imputar mediana)
print(f"Nulos en Piso antes de imputar: {df['Piso'].isna().sum():,}")
piso_validos = df["Piso"].replace(0, np.nan)
mediana_piso_distrito = piso_validos.groupby(df["Distrito"]).median().fillna(2.0).to_dict()
piso_imputado = df["Distrito"].map(mediana_piso_distrito).fillna(2.0)
df["Piso"] = piso_validos.fillna(piso_imputado)

# 4.3 Imputación de Antiguedad (Mediana por distrito)
print(f"Nulos en Antiguedad antes de imputar: {df['Antiguedad'].isna().sum():,}")
mediana_antiguedad_distrito = df.groupby("Distrito")["Antiguedad"].median().fillna(df["Antiguedad"].median()).to_dict()
df["Antiguedad"] = df["Antiguedad"].fillna(df["Distrito"].map(mediana_antiguedad_distrito))

# 4.4 Imputación de Habitaciones, Banios, Garajes
for col in ["Habitaciones", "Banios", "Garajes"]:
    mediana_col = df.groupby("Distrito")[col].median().fillna(df[col].median()).to_dict()
    df[col] = df[col].fillna(df["Distrito"].map(mediana_col))

print("Imputación física completada. Verificando nulos restantes en BCRP:")
print(df[cols_fisicas].isnull().sum().to_dict())

# %% 5. Merge Dinámico con Tabla Maestra de Contexto por (Distrito, Anio)
print("\n--- 3. Fusión con datos de contexto distrital por (Distrito, Anio) ---")
df_ctx = pd.read_csv(CONTEXT_PATH)
print(f"Tabla de contexto: {df_ctx.shape[0]} filas x {df_ctx.shape[1]} columnas")

# Realizar merge
n_antes = len(df)
df = df.merge(df_ctx, on=["Distrito", "Anio"], how="inner")
print(f"Merge exitoso: {len(df):,} filas conservadas de {n_antes:,} (coincidencia 100%)")

# %% 6. Separar Flags de Imputación (para análisis de sensibilidad)
cols_flags_existentes = [c for c in COLS_FLAGS if c in df.columns]
df_flags = df[["Anio", "Distrito"] + cols_flags_existentes].copy()
print(f"Flags de imputación separados: {cols_flags_existentes}")

# %% 7. Eliminar Columnas Multicolineales y No Predictivas
cols_a_eliminar = [c for c in COLS_ELIMINAR + COLS_FLAGS if c in df.columns]
df = df.drop(columns=cols_a_eliminar)
print(f"Columnas eliminadas ({len(cols_a_eliminar)}): {cols_a_eliminar}")

# %% 8. Feature Engineering (Idéntico a Venta para Coherencia)
print("\n--- 4. Feature Engineering ---")
df["periodo_numerico"] = df["Anio"] * 4 + df["Trimestre"]
df["m2_por_habitacion"] = df["Superficie"] / (df["Habitaciones"] + 1)
df["ratio_banios_hab"]  = df["Banios"] / (df["Habitaciones"] + 0.1)
df["tiene_garaje"]      = (df["Garajes"] > 0).astype(int)
df["es_piso_alto"]      = (df["Piso"] >= 8).astype(int)
df["superficie_cuadrado"] = (df["Superficie"] / 100.0) ** 2

print("[OK] Features creadas: periodo_numerico, m2_por_habitacion, ratio_banios_hab, tiene_garaje, es_piso_alto, superficie_cuadrado")

# %% 9. Split Temporal
print("\n--- 5. Split Temporal ---")
df_train = df[df["Anio"].isin(TRAIN_YEARS)].copy()
df_test = df[df["Anio"].isin(TEST_YEARS)].copy()

df_flags_train = df_flags[df_flags["Anio"].isin(TRAIN_YEARS)].copy()
df_flags_test = df_flags[df_flags["Anio"].isin(TEST_YEARS)].copy()

# Limpieza de outliers extremos SOLO en train (0.5% - 99.5%)
n_train_antes = len(df_train)
p_sup_min, p_sup_max = df_train["Superficie"].quantile(0.005), df_train["Superficie"].quantile(0.995)
p_alq_min, p_alq_max = df_train[TARGET].quantile(0.005), df_train[TARGET].quantile(0.995)

mask_train_valid = (
    (df_train["Superficie"] >= p_sup_min) & (df_train["Superficie"] <= p_sup_max) &
    (df_train[TARGET] >= p_alq_min) & (df_train[TARGET] <= p_alq_max)
)
df_train = df_train[mask_train_valid].copy()
df_flags_train = df_flags_train.loc[df_train.index].copy()

total = len(df_train) + len(df_test)
print(f"  Filas Train ({min(TRAIN_YEARS)}-{max(TRAIN_YEARS)}): {len(df_train):,} ({len(df_train)/total*100:.1f}%) [filtrados {n_train_antes - len(df_train)} outliers]")
print(f"  Filas Test  ({min(TEST_YEARS)}-{max(TEST_YEARS)}): {len(df_test):,} ({len(df_test)/total*100:.1f}%)")

assert len(set(df_train["Anio"].unique()) & set(df_test["Anio"].unique())) == 0, "DATA LEAKAGE TEMPORAL!"

# %% 10. Target Encoding para Distrito (Calculado SOLO en Train)
print("\n--- 6. Target Encoding para Distrito ---")
COL_DISTRITO = "Distrito"
media_global = df_train[TARGET].mean()
stats = df_train.groupby(COL_DISTRITO)[TARGET].agg(["mean", "count"])
stats["distrito_encoded"] = (
    (stats["count"] * stats["mean"] + SMOOTHING * media_global)
    / (stats["count"] + SMOOTHING)
)
encoding_map = stats["distrito_encoded"].to_dict()

df_train["distrito_encoded"] = df_train[COL_DISTRITO].map(encoding_map)
df_test["distrito_encoded"] = df_test[COL_DISTRITO].map(encoding_map).fillna(media_global)

# Guardar referencias de distrito antes de dropear
distrito_train = df_train[COL_DISTRITO].copy()
distrito_test = df_test[COL_DISTRITO].copy()

df_train = df_train.drop(columns=[COL_DISTRITO])
df_test = df_test.drop(columns=[COL_DISTRITO])

# %% 11. Log-Transform del Target
print("\n--- 7. Log-Transform del Target ---")
df_train[f"{TARGET}_log"] = np.log1p(df_train[TARGET])
df_test[f"{TARGET}_log"] = np.log1p(df_test[TARGET])
print(f"Skewness original en Train: {df_train[TARGET].skew():.2f}")
print(f"Skewness tras log1p en Train: {df_train[f'{TARGET}_log'].skew():.2f}")

# %% 12. Guardar Datasets y Metadata de Alquiler
print("\n--- 8. Guardando datasets procesados ---")
train_path = os.path.join(OUTPUT_DIR, "train_alquiler.csv")
test_path = os.path.join(OUTPUT_DIR, "test_alquiler.csv")

df_train.to_csv(train_path, index=False)
df_test.to_csv(test_path, index=False)
print(f"  [OK] Train guardado: {train_path} ({df_train.shape})")
print(f"  [OK] Test guardado:  {test_path} ({df_test.shape})")

# Metadata de features
target_cols = [TARGET, f"{TARGET}_log"]
feature_cols = [c for c in df_train.columns if c not in target_cols]

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
    "distritos_evaluados": list(encoding_map.keys()),
}

metadata_path = os.path.join(OUTPUT_DIR, "features_metadata_alquiler.json")
with open(metadata_path, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)
print(f"  [OK] Metadata guardada: {metadata_path}")

# Flags de imputación
flags_path = os.path.join(OUTPUT_DIR, "flags_imputacion_alquiler.csv")
df_flags_combined = pd.concat([
    df_flags_train.assign(split="train"),
    df_flags_test.assign(split="test"),
])
df_flags_combined.to_csv(flags_path, index=False)
print(f"  [OK] Flags guardados: {flags_path}")

# Referencia de distritos
ref_path = os.path.join(OUTPUT_DIR, "distrito_referencia_alquiler.csv")
pd.DataFrame({
    "split": ["train"] * len(distrito_train) + ["test"] * len(distrito_test),
    "Distrito": pd.concat([distrito_train, distrito_test]).values,
}).to_csv(ref_path, index=False)
print(f"  [OK] Referencia distritos guardada: {ref_path}")

print("\n" + "=" * 60)
print("[OK] PREPROCESAMIENTO DE ALQUILER COMPLETADO CON ÉXITO")
print("=" * 60)
