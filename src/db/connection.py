"""
Gestor de conexiones y operaciones DDL en PostgreSQL.
Soporta SQLAlchemy (para bulk loading con Pandas) y psycopg2 directo (para DDL y transacciones).
"""
import os
import logging
from pathlib import Path
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from sqlalchemy import create_engine, Engine, text
from src.db.config import DatabaseConfig

logger = logging.getLogger(__name__)

_engine = None

def get_engine() -> Engine:
    """Retorna una instancia singleton del SQLAlchemy Engine."""
    global _engine
    if _engine is None:
        url = DatabaseConfig.get_sqlalchemy_url()
        _engine = create_engine(
            url,
            pool_size=DatabaseConfig.POOL_SIZE,
            max_overflow=DatabaseConfig.MAX_OVERFLOW,
            echo=False
        )
    return _engine

def get_connection(dbname: str = None):
    """Retorna una conexión activa de psycopg2."""
    dsn = DatabaseConfig.get_dsn(dbname=dbname)
    return psycopg2.connect(**dsn)

def check_connection() -> bool:
    """Verifica si el servidor de PostgreSQL está accesible."""
    try:
        conn = get_connection(dbname="postgres")
        conn.close()
        return True
    except Exception as e:
        logger.warning(f"No se pudo conectar a PostgreSQL: {e}")
        return False

def ensure_database_exists(target_db: str = None) -> bool:
    """
    Verifica si la base de datos objetivo existe; si no existe, la crea
    conectándose a la base de datos de mantenimiento 'postgres'.
    """
    db_name = target_db or DatabaseConfig.NAME
    try:
        conn = get_connection(dbname="postgres")
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM pg_database WHERE datname = %s;", (db_name,))
            exists = cur.fetchone() is not None
            if not exists:
                logger.info(f"Creando base de datos '{db_name}'...")
                cur.execute(f'CREATE DATABASE "{db_name}";')
                logger.info(f"Base de datos '{db_name}' creada exitosamente.")
            else:
                logger.info(f"La base de datos '{db_name}' ya existe.")
        conn.close()
        return True
    except Exception as e:
        logger.error(f"Error al verificar/crear base de datos '{db_name}': {e}")
        raise

def init_db(sql_file_path: str = None) -> bool:
    """
    Ejecuta el script SQL DDL para crear las tablas, constraints e índices.
    """
    ensure_database_exists()
    
    if sql_file_path is None:
        base_dir = Path(__file__).resolve().parent.parent.parent
        sql_file_path = base_dir / "sql" / "schema_training_db.sql"
        
    sql_path = Path(sql_file_path)
    if not sql_path.exists():
        raise FileNotFoundError(f"Archivo de esquema no encontrado: {sql_path}")
        
    logger.info(f"Ejecutando script DDL: {sql_path.name}...")
    with open(sql_path, "r", encoding="utf-8") as f:
        ddl_script = f.read()
        
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(ddl_script)
        conn.commit()
        logger.info("Esquema de base de datos inicializado exitosamente.")
        return True
    except Exception as e:
        conn.rollback()
        logger.error(f"Error al ejecutar DDL: {e}")
        raise
    finally:
        conn.close()
