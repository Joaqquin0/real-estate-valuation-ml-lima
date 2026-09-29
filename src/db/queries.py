"""
Consultas SQL analíticas para alimentar el Servicio de Machine Learning desde PostgreSQL.
Realiza los JOINs entre las tablas de inmuebles y el contexto distrital automáticamente.
"""
import logging
import pandas as pd
from src.db.connection import get_engine

logger = logging.getLogger(__name__)

def load_training_dataset_venta(split: str = None) -> pd.DataFrame:
    """
    Carga el dataset de venta consolidado con sus variables contextuales desde PostgreSQL.
    
    Args:
        split: 'TRAIN', 'TEST' o None (para ambos).
    """
    engine = get_engine()
    split_filter = f"WHERE v.split_dataset = '{split.upper()}'" if split else ""
    
    query = f"""
    SELECT 
        v.id AS inmueble_id,
        v.anio,
        v.trimestre,
        d.nombre AS distrito,
        v.superficie_m2,
        v.habitaciones,
        v.banos,
        v.garajes,
        v.piso,
        v.antiguedad_anios,
        v.vista_exterior,
        v.precio_soles_nominal,
        v.precio_soles_const,
        v.split_dataset,
        -- Contexto Distrital Anual
        c.pct_nse_a, c.pct_nse_b, c.pct_nse_c, c.pct_nse_d, c.pct_nse_e,
        c.tasa_robo, c.tasa_hurto, c.tasa_denuncias,
        c.poblacion_proyectada, c.densidad_hab_km2,
        c.distancia_centro_km, c.dist_colegio_km, c.dist_hospital_km,
        c.dist_estacion_transporte_km, c.dist_centro_comercial_km,
        c.dist_parque_km, c.dist_universidad_km
    FROM dataset_inmuebles_venta v
    JOIN distritos d ON v.distrito_id = d.id
    JOIN distrito_anio_contexto c ON v.distrito_id = c.distrito_id AND v.anio = c.anio
    {split_filter}
    ORDER BY v.anio ASC, v.trimestre ASC;
    """
    logger.info("Cargando dataset de venta desde PostgreSQL con JOIN contextual...")
    df = pd.read_sql(query, con=engine)
    logger.info(f"Dataset venta cargado desde BD: {len(df):,} filas x {len(df.columns)} columnas.")
    return df

def load_training_dataset_alquiler(split: str = None) -> pd.DataFrame:
    """
    Carga el dataset de alquiler consolidado con sus variables contextuales desde PostgreSQL.
    
    Args:
        split: 'TRAIN', 'TEST' o None (para ambos).
    """
    engine = get_engine()
    split_filter = f"WHERE a.split_dataset = '{split.upper()}'" if split else ""
    
    query = f"""
    SELECT 
        a.id AS inmueble_id,
        a.anio,
        a.trimestre,
        d.nombre AS distrito,
        a.superficie_m2,
        a.habitaciones,
        a.banos,
        a.garajes,
        a.piso,
        a.antiguedad_anios,
        a.vista_exterior,
        a.alquiler_soles_nominal,
        a.alquiler_soles_const,
        a.split_dataset,
        -- Contexto Distrital Anual
        c.pct_nse_a, c.pct_nse_b, c.pct_nse_c, c.pct_nse_d, c.pct_nse_e,
        c.tasa_robo, c.tasa_hurto, c.tasa_denuncias,
        c.poblacion_proyectada, c.densidad_hab_km2,
        c.distancia_centro_km, c.dist_colegio_km, c.dist_hospital_km,
        c.dist_estacion_transporte_km, c.dist_centro_comercial_km,
        c.dist_parque_km, c.dist_universidad_km
    FROM dataset_inmuebles_alquiler a
    JOIN distritos d ON a.distrito_id = d.id
    JOIN distrito_anio_contexto c ON a.distrito_id = c.distrito_id AND a.anio = c.anio
    {split_filter}
    ORDER BY a.anio ASC, a.trimestre ASC;
    """
    logger.info("Cargando dataset de alquiler desde PostgreSQL con JOIN contextual...")
    df = pd.read_sql(query, con=engine)
    logger.info(f"Dataset alquiler cargado desde BD: {len(df):,} filas x {len(df.columns)} columnas.")
    return df

def get_historial_ejecuciones(limit: int = 10) -> pd.DataFrame:
    """Retorna las últimas corridas registradas en pipeline_ejecuciones."""
    engine = get_engine()
    query = f"""
    SELECT id, tipo_ejecucion, ejecutado_por, iniciado_en, finalizado_en, 
           duracion_segundos, registros_leidos, registros_guardados, registros_filtrados,
           errores, metricas_salida, observaciones, estado
    FROM pipeline_ejecuciones
    ORDER BY iniciado_en DESC
    LIMIT {limit};
    """
    return pd.read_sql(query, con=engine)
