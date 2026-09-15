"""
app/core/data_provider.py
─────────────────────────
Abstracción de la fuente de datos contextuales de distrito.

Fase 1 → CSVContextProvider (lee distrito_contexto_ref.csv en memoria)
Fase 2 → MongoContextProvider (hereda la misma interfaz, sin tocar
          prediccion_service.py ni ningún router)

El cambio de fase es una sola línea en app/core/model_loader.py.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


# ─── Columnas contextuales requeridas ─────────────────────────────────────────

CONTEXT_COLS: list[str] = [
    "pct_NSE_A", "pct_NSE_B", "pct_NSE_C", "pct_NSE_D", "pct_NSE_E",
    "tasa_robo", "tasa_hurto",
    "poblacion_proyectada", "area_distrito_km2", "densidad_hab_km2",
    "distancia_centro_km",
    "dist_colegio_km", "dist_hospital_km", "dist_estacion_transporte_km",
    "dist_centro_comercial_km", "dist_parque_km", "dist_universidad_km",
]


# ─── Interfaz ─────────────────────────────────────────────────────────────────

class IContextDataProvider(ABC):
    """
    Contrato que deben cumplir todos los proveedores de contexto distrital.
    prediccion_service.py solo depende de esta interfaz.
    """

    @abstractmethod
    def get_distrito_context(self, distrito: str) -> dict[str, Any]:
        """
        Devuelve un diccionario con las variables contextuales del distrito.
        Si el distrito no existe, devuelve los promedios globales (fallback).
        """
        ...

    @abstractmethod
    def get_distritos_disponibles(self) -> list[str]:
        """Lista de distritos con datos contextuales disponibles."""
        ...

    @abstractmethod
    def get_global_means(self) -> dict[str, float]:
        """Medias globales de todas las variables contextuales (usado como fallback)."""
        ...


# ─── Fase 1: CSV ──────────────────────────────────────────────────────────────

class CSVContextProvider(IContextDataProvider):
    """
    Lee distrito_contexto_ref.csv en memoria al iniciar el servicio.
    Carga única — sin I/O por request.
    """

    def __init__(self, csv_path: str) -> None:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"[CSVContextProvider] No se encontró el archivo de contexto: {csv_path}\n"
                "Asegúrate de que 'data/distrito_contexto_ref.csv' existe "
                "(se genera con el notebook 02_preprocesamiento.py)."
            )
        self._df: pd.DataFrame = pd.read_csv(csv_path)
        self._global_means: dict[str, float] = {
            col: float(self._df[col].mean())
            for col in CONTEXT_COLS
            if col in self._df.columns
        }
        print(
            f"[CSVContextProvider] Cargado: {len(self._df)} distritos "
            f"desde {os.path.basename(csv_path)}"
        )

    def get_distrito_context(self, distrito: str) -> dict[str, Any]:
        fila = self._df[self._df["Distrito"] == distrito]
        if fila.empty:
            print(
                f"[CSVContextProvider] Distrito '{distrito}' sin contexto. "
                "Usando medias globales como fallback."
            )
            return self._global_means.copy()
        return fila.iloc[0][CONTEXT_COLS].to_dict()

    def get_distritos_disponibles(self) -> list[str]:
        return sorted(self._df["Distrito"].dropna().tolist())

    def get_global_means(self) -> dict[str, float]:
        return self._global_means.copy()


# ─── Fase 2: MongoDB (stub — implementar en fase siguiente) ───────────────────

class MongoContextProvider(IContextDataProvider):
    """
    Placeholder para la Fase 2.
    Reemplaza CSVContextProvider sin tocar prediccion_service.py.

    Implementación pendiente:
      - Conectar a MongoDB con motor/pymongo
      - Colección: distrito_contexto (un doc por distrito)
      - Caché local en memoria para evitar queries por request
    """

    def __init__(self, connection_string: str, db: str, collection: str) -> None:
        raise NotImplementedError(
            "MongoContextProvider está pendiente para Fase 2. "
            "Usa DATA_PROVIDER=csv en el .env para Fase 1."
        )

    def get_distrito_context(self, distrito: str) -> dict[str, Any]:
        raise NotImplementedError

    def get_distritos_disponibles(self) -> list[str]:
        raise NotImplementedError

    def get_global_means(self) -> dict[str, float]:
        raise NotImplementedError
