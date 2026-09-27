"""
Configuración centralizada del proyecto.
Modelo predictivo XGBoost para valoración inmobiliaria - Lima Metropolitana.
"""

import os

# ============================================================
# RUTAS
# ============================================================
# En Google Colab, cambiar BASE_DIR a la ruta de tu Google Drive montado.
# Ejemplo: BASE_DIR = "/content/drive/MyDrive/python_modelo_tesis"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports", "figures")

# Archivo fuente Venta (BCRP base)
DATASET_FILENAME = "dataset_entrenamiento_venta_2025.xlsx"
DATASET_PATH = os.path.join(DATA_RAW_DIR, DATASET_FILENAME)

# Archivo fuente Alquiler (BCRP base)
DATASET_ALQUILER_FILENAME = "dataset_entrenamineto_alquiler_2025.xlsx"
DATASET_ALQUILER_PATH = os.path.join(DATA_RAW_DIR, DATASET_ALQUILER_FILENAME)
DISTRICT_CONTEXT_PATH = os.path.join(DATA_PROCESSED_DIR, "distrito_anio_contexto.csv")

# Archivos procesados Venta
TRAIN_PATH = os.path.join(DATA_PROCESSED_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_PROCESSED_DIR, "test.csv")
FEATURES_METADATA_PATH = os.path.join(DATA_PROCESSED_DIR, "features_metadata.json")

# Archivos procesados Alquiler
TRAIN_ALQUILER_PATH = os.path.join(DATA_PROCESSED_DIR, "train_alquiler.csv")
TEST_ALQUILER_PATH = os.path.join(DATA_PROCESSED_DIR, "test_alquiler.csv")
FEATURES_METADATA_ALQUILER_PATH = os.path.join(DATA_PROCESSED_DIR, "features_metadata_alquiler.json")

# Modelos guardados
MODEL_VENTA_PATH = os.path.join(MODELS_DIR, "xgboost_venta_v2.pkl")
MODEL_ALQUILER_PATH = os.path.join(MODELS_DIR, "xgboost_alquiler_v1.pkl")

# ============================================================
# VARIABLES TARGET
# ============================================================
TARGET_COL = "Precio_Soles_Const"
TARGET_ALQUILER_COL = "Alquiler_Soles_Const"

# ============================================================
# COLUMNAS POR CATEGORÍA
# ============================================================

# Columnas a eliminar (no entran al modelo)
COLS_DROP = [
    "ID",
    "Ubigeo",
    # Targets alternativos (no se usan como features)
    "Precio_Dolares",
    "Precio_Soles",
    "precio_dolares_m2",
]

# Flags de imputación (solo para análisis de sensibilidad, no como features)
COLS_FLAGS_IMPUTACION = [
    "nse_imputado",
    "tasas_criminalidad_imputada",
    "poblacion_imputada",
]

# Columna categórica principal (requiere encoding)
COL_DISTRITO = "Distrito"

# Features físicas del inmueble
COLS_FISICAS = [
    "Superficie",
    "Habitaciones",
    "Banios",
    "Garajes",
    "Piso",
    "Vista_Exterior",
    "Antiguedad",
]

# Features macroeconómicas
COLS_MACRO = [
    "Tipo_Cambio",
    "IPC",
]

# Features de nivel socioeconómico
COLS_NSE = [
    "pct_NSE_A",
    "pct_NSE_B",
    "pct_NSE_C",
    "pct_NSE_D",
    "pct_NSE_E",
]

# Features de seguridad ciudadana
COLS_SEGURIDAD = [
    "tasa_denuncias",
    "tasa_robo",
    "tasa_hurto",
]

# Features demográficas
COLS_DEMOGRAFICAS = [
    "poblacion_proyectada",
    "area_distrito_km2",
    "densidad_hab_km2",
]

# Features geoespaciales
COLS_GEOESPACIALES = [
    "distancia_centro_km",
    "dist_colegio_km",
    "dist_hospital_km",
    "dist_estacion_transporte_km",
    "dist_centro_comercial_km",
    "dist_parque_km",
    "dist_universidad_km",
]

# Features temporales
COLS_TEMPORALES = [
    "Anio",
    "Trimestre",
]

# ============================================================
# SPLIT TEMPORAL
# ============================================================
TRAIN_YEARS = list(range(2016, 2024))  # 2016–2023
TEST_YEARS = list(range(2024, 2026))   # 2024–2025
SPLIT_COL = "Anio"

# ============================================================
# HIPERPARÁMETROS XGBoost (defaults iniciales)
# ============================================================
XGBOOST_DEFAULT_PARAMS = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 5,
    "min_child_weight": 3,
    "random_state": 42,
    "n_jobs": -1,
}

# Espacio de búsqueda para Optuna/GridSearch
XGBOOST_TUNING_SPACE = {
    "n_estimators": [100, 300, 500, 800],
    "max_depth": [4, 6, 8, 10],
    "learning_rate": [0.01, 0.05, 0.1],
    "subsample": [0.7, 0.8, 0.9],
    "colsample_bytree": [0.7, 0.8, 0.9],
    "reg_alpha": [0, 0.1, 1],
    "reg_lambda": [1, 5, 10],
    "min_child_weight": [1, 3, 5],
}

# ============================================================
# MÉTRICAS Y UMBRALES DE ACEPTACIÓN
# ============================================================
MAPE_BENCHMARK = 17.89   # Oporto et al. (2024) — superar este valor
MAPE_IDEAL = 10.0        # Estándar IAAO
R2_THRESHOLD = 0.80      # Mínimo aceptable

# ============================================================
# REPRODUCIBILIDAD
# ============================================================
RANDOM_SEED = 42
CV_FOLDS = 5  # TimeSeriesSplit folds
