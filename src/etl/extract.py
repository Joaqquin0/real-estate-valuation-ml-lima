"""
Módulo de Extracción de Datos (Extract).
Carga los archivos primarios de Excel del BCRP y la tabla de contexto distrital.
"""
import logging
from pathlib import Path
import pandas as pd
from config import DATASET_PATH, DATASET_ALQUILER_PATH, DISTRICT_CONTEXT_PATH

logger = logging.getLogger(__name__)

def extract_contexto_distrital(filepath: str = None) -> pd.DataFrame:
    """Extrae la tabla maestra de contexto distrital-anual (202 registros)."""
    path = Path(filepath or DISTRICT_CONTEXT_PATH)
    if not path.exists():
        raise FileNotFoundError(f"Archivo de contexto distrital no encontrado: {path}")
    logger.info(f"Extrayendo contexto distrital desde: {path.name}")
    df = pd.read_csv(path)
    logger.info(f"Contexto distrital cargado: {len(df)} filas, {len(df.columns)} columnas.")
    return df

def extract_raw_venta(filepath: str = None) -> pd.DataFrame:
    """Extrae el archivo Excel primario del BCRP para venta inmobiliaria."""
    path = Path(filepath or DATASET_PATH)
    if not path.exists():
        raise FileNotFoundError(f"Archivo de venta BCRP no encontrado: {path}")
    logger.info(f"Extrayendo dataset de venta desde: {path.name}...")
    df = pd.read_excel(path)
    logger.info(f"Dataset venta cargado: {len(df)} filas crudas, {len(df.columns)} columnas.")
    return df

def extract_raw_alquiler(filepath: str = None) -> pd.DataFrame:
    """Extrae el archivo Excel primario del BCRP para alquiler inmobiliario."""
    path = Path(filepath or DATASET_ALQUILER_PATH)
    if not path.exists():
        raise FileNotFoundError(f"Archivo de alquiler BCRP no encontrado: {path}")
    logger.info(f"Extrayendo dataset de alquiler desde: {path.name}...")
    df = pd.read_excel(path)
    logger.info(f"Dataset alquiler cargado: {len(df)} filas crudas, {len(df.columns)} columnas.")
    return df
