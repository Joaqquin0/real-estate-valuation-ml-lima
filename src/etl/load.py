"""
Módulo de Carga a Base de Datos (Load).
Inserta y sincroniza datos masivos en PostgreSQL con alta eficiencia (bulk loading).
"""
import logging
import pandas as pd
from psycopg2.extras import execute_values
from src.db.connection import get_connection, get_engine

logger = logging.getLogger(__name__)

def load_distritos(df_distritos: pd.DataFrame) -> dict:
    """
    Inserta o actualiza los 22 distritos en la tabla 'distritos'.
    Retorna un diccionario {nombre_distrito: id_db}.
    """
    conn = get_connection()
    distrito_id_map = {}
    try:
        with conn.cursor() as cur:
            query = """
                INSERT INTO distritos (nombre, ubigeo, area_km2)
                VALUES %s
                ON CONFLICT (nombre) 
                DO UPDATE SET 
                    ubigeo = EXCLUDED.ubigeo,
                    area_km2 = EXCLUDED.area_km2
                RETURNING id, nombre;
            """
            data = [(row["nombre"], row["ubigeo"], float(row["area_km2"])) for _, row in df_distritos.iterrows()]
            execute_values(cur, query, data)
            for row in cur.fetchall():
                distrito_id_map[row[1]] = row[0]
        conn.commit()
        logger.info(f"Distritos cargados/actualizados: {len(distrito_id_map)} mapeos obtenidos.")
        return distrito_id_map
    except Exception as e:
        conn.rollback()
        logger.error(f"Error al cargar distritos: {e}")
        raise
    finally:
        conn.close()

def load_contexto_distrital(df_contexto: pd.DataFrame):
    """Inserta o actualiza la tabla 'distrito_anio_contexto' (202 registros)."""
    conn = get_connection()
    try:
        columnas = list(df_contexto.columns)
        cols_str = ", ".join(columnas)
        update_str = ", ".join([f"{col} = EXCLUDED.{col}" for col in columnas if col not in ["distrito_id", "anio"]])
        
        query = f"""
            INSERT INTO distrito_anio_contexto ({cols_str})
            VALUES %s
            ON CONFLICT (distrito_id, anio)
            DO UPDATE SET {update_str};
        """
        
        # Convertir NaN de Pandas a None de Python para PostgreSQL
        data = [tuple(None if pd.isna(v) else v for v in row) for row in df_contexto.itertuples(index=False)]
        
        with conn.cursor() as cur:
            execute_values(cur, query, data, page_size=500)
        conn.commit()
        logger.info(f"Contexto distrital cargado: {len(df_contexto)} registros sincronizados.")
    except Exception as e:
        conn.rollback()
        logger.error(f"Error al cargar contexto distrital: {e}")
        raise
    finally:
        conn.close()

def load_inmuebles_venta(df_venta: pd.DataFrame, truncate: bool = False):
    """Carga masiva de ofertas de venta en 'dataset_inmuebles_venta'."""
    engine = get_engine()
    if truncate:
        logger.info("Truncando tabla 'dataset_inmuebles_venta' antes de la carga...")
        with engine.begin() as conn:
            conn.exec_driver_sql("TRUNCATE TABLE dataset_inmuebles_venta RESTART IDENTITY CASCADE;")
            
    logger.info(f"Cargando {len(df_venta):,} transacciones de venta en PostgreSQL...")
    df_venta.to_sql(
        name="dataset_inmuebles_venta",
        con=engine,
        if_exists="append",
        index=False,
        chunksize=5000,
        method="multi"
    )
    logger.info("Carga de dataset_inmuebles_venta completada exitosamente.")

def load_inmuebles_alquiler(df_alquiler: pd.DataFrame, truncate: bool = False):
    """Carga masiva de contratos de alquiler en 'dataset_inmuebles_alquiler'."""
    engine = get_engine()
    if truncate:
        logger.info("Truncando tabla 'dataset_inmuebles_alquiler' antes de la carga...")
        with engine.begin() as conn:
            conn.exec_driver_sql("TRUNCATE TABLE dataset_inmuebles_alquiler RESTART IDENTITY CASCADE;")
            
    logger.info(f"Cargando {len(df_alquiler):,} contratos de alquiler en PostgreSQL...")
    df_alquiler.to_sql(
        name="dataset_inmuebles_alquiler",
        con=engine,
        if_exists="append",
        index=False,
        chunksize=5000,
        method="multi"
    )
    logger.info("Carga de dataset_inmuebles_alquiler completada exitosamente.")

def load_auditoria_flags(df_flags: pd.DataFrame, truncate: bool = False):
    """Carga de flags de auditoría para trazabilidad científica y análisis de sensibilidad."""
    engine = get_engine()
    if truncate:
        with engine.begin() as conn:
            conn.exec_driver_sql("TRUNCATE TABLE auditoria_flags_imputacion RESTART IDENTITY;")
            
    df_flags.to_sql(
        name="auditoria_flags_imputacion",
        con=engine,
        if_exists="append",
        index=False,
        chunksize=5000,
        method="multi"
    )
    logger.info(f"Flags de auditoría cargados: {len(df_flags):,} registros.")
