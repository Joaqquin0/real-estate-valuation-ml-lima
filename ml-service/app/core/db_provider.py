"""
app/core/db_provider.py
────────────────────────
Abstracción para la lectura de datos de entrenamiento desde PostgreSQL.

Implementa el patrón ITrainingDataProvider para desacoplar el servicio
de entrenamiento del proveedor de datos concreto (PostgreSQL en Fase 1,
extensible a otras fuentes en fases futuras).

El JOIN utilizado replica exactamente la consulta documentada en:
  GUIA_ENTRENAMIENTO_ML_DESDE_POSTGRES.md — Sección 1

Dependencias:
  pip install psycopg2-binary  (driver de PostgreSQL para Python)

Variables de entorno requeridas:
  DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Literal

import pandas as pd

# Importación condicional para no requerir psycopg2 si solo se usa el servicio de inferencia
try:
    import psycopg2
    import psycopg2.extras
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False


# ─── Interfaz abstracta ───────────────────────────────────────────────────────

class ITrainingDataProvider(ABC):
    """
    Interfaz para la lectura de datos de entrenamiento.

    Separado de IContextDataProvider (datos de inferencia) para mantener
    el principio de responsabilidad única: este proveedor solo se usa
    durante el pipeline de reentrenamiento.
    """

    @abstractmethod
    def load_dataset_venta(
        self,
        split: Literal["TRAIN", "TEST", "ALL"] = "ALL",
    ) -> pd.DataFrame:
        """
        Retorna el dataset de venta con el JOIN contextual aplicado.

        Args:
            split: 'TRAIN' (2016-2023), 'TEST' (2024-2025), 'ALL' (ambos).

        Returns:
            DataFrame con todas las columnas del JOIN (características del inmueble
            + variables contextuales del distrito para el año de la transacción).
        """
        ...

    @abstractmethod
    def check_connection(self) -> bool:
        """Verifica que la conexión a la fuente de datos esté activa."""
        ...


# ─── Implementación PostgreSQL (Fase 1) ───────────────────────────────────────

# SQL que implementa el JOIN contextual documentado en la guía.
# Une: características físicas del inmueble + contexto distrital del año exacto.
_QUERY_VENTA_TEMPLATE = """
SELECT
    v.id            AS inmueble_id,
    v.anio,
    v.trimestre,
    d.nombre        AS distrito,
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
    -- Variables Contextuales Anuales (JOIN con distrito_anio_contexto)
    c.pct_nse_a,
    c.pct_nse_b,
    c.pct_nse_c,
    c.pct_nse_d,
    c.pct_nse_e,
    c.tasa_robo,
    c.tasa_hurto,
    c.poblacion_proyectada,
    c.densidad_hab_km2,
    c.distancia_centro_km,
    c.dist_colegio_km,
    c.dist_hospital_km,
    c.dist_estacion_transporte_km,
    c.dist_centro_comercial_km,
    c.dist_parque_km,
    c.dist_universidad_km
FROM dataset_inmuebles_venta v
JOIN distritos d
    ON v.distrito_id = d.id
JOIN distrito_anio_contexto c
    ON v.distrito_id = c.distrito_id AND v.anio = c.anio
{where_clause}
ORDER BY v.anio ASC, v.trimestre ASC;
"""


class PostgreSQLTrainingProvider(ITrainingDataProvider):
    """
    Proveedor de datos de entrenamiento desde PostgreSQL (inmobiliaria_ml_db).

    Lee las tablas:
      - dataset_inmuebles_venta  → características físicas + precio + split_dataset
      - distritos                → nombre del distrito
      - distrito_anio_contexto   → NSE, seguridad, demografía, distancias POI

    No accede a: auditoria_flags_imputacion ni pipeline_ejecuciones
    (esas tablas son exclusivas del pipeline ETL, según la guía).
    """

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        db_name: str | None = None,
        user: str | None = None,
        password: str | None = None,
    ) -> None:
        if not PSYCOPG2_AVAILABLE:
            raise ImportError(
                "psycopg2 no está instalado. "
                "Ejecuta: pip install psycopg2-binary"
            )

        self.conn_params = {
            "host":     host     or os.getenv("DB_HOST", "localhost"),
            "port":     port     or int(os.getenv("DB_PORT", "5432")),
            "dbname":   db_name  or os.getenv("DB_NAME", "inmobiliaria_ml_db"),
            "user":     user     or os.getenv("DB_USER", "postgres"),
            "password": password or os.getenv("DB_PASSWORD", ""),
        }

    def _get_connection(self):
        """Abre y retorna una nueva conexión a PostgreSQL."""
        return psycopg2.connect(**self.conn_params)

    def check_connection(self) -> bool:
        """Verifica que PostgreSQL esté accesible."""
        try:
            conn = self._get_connection()
            conn.close()
            return True
        except Exception:
            return False

    def load_dataset_venta(
        self,
        split: Literal["TRAIN", "TEST", "ALL"] = "ALL",
    ) -> pd.DataFrame:
        """
        Ejecuta el JOIN contextual y retorna un DataFrame limpio.

        El split respeta la partición temporal definida en la DB:
          - TRAIN: anios 2016-2023
          - TEST:  anios 2024-2025
          - ALL:   ambos (para análisis, no para entrenamiento directo)

        Raises:
            RuntimeError: Si la consulta falla o retorna 0 filas.
        """
        if split == "ALL":
            where_clause = ""
        else:
            where_clause = f"WHERE v.split_dataset = '{split}'"

        query = _QUERY_VENTA_TEMPLATE.format(where_clause=where_clause)

        conn = self._get_connection()
        try:
            df = pd.read_sql_query(query, conn)
        finally:
            conn.close()

        if df.empty:
            raise RuntimeError(
                f"La consulta para split='{split}' no retornó filas. "
                "Verifica que la BD tenga datos cargados."
            )

        print(
            f"[PostgreSQLTrainingProvider] Cargados {len(df):,} registros "
            f"(split={split}) desde '{self.conn_params['dbname']}'."
        )
        return df
