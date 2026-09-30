"""
app/schemas/entrenamiento.py
────────────────────────────
Contratos Pydantic para el endpoint de reentrenamiento administrativo.

POST /api/v1/admin/entrenamiento/venta
  → Lanza un job de entrenamiento en background
  → Devuelve inmediatamente un job_id para consultar estado

GET /api/v1/admin/entrenamiento/estado/{job_id}
  → Retorna el estado actual del job de entrenamiento
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ─── Enums ────────────────────────────────────────────────────────────────────

class EstadoJob(str, Enum):
    PENDIENTE   = "pendiente"
    EN_PROGRESO = "en_progreso"
    COMPLETADO  = "completado"
    FALLIDO     = "fallido"


# ─── REQUEST ──────────────────────────────────────────────────────────────────

class EntrenamientoVentaRequest(BaseModel):
    """
    Parámetros opcionales para el job de reentrenamiento.

    Si se omiten, se usan los hiperparámetros del modelo actual
    registrados en config/model_config.json.
    """
    nombre_modelo: str = Field(
        default="xgboost_venta_v2",
        description="Nombre base del artefacto a generar (sin extensión). "
                    "Ejemplo: 'xgboost_venta_v3'.",
        examples=["xgboost_venta_v2"],
    )
    hiperparametros: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Hiperparámetros del XGBRegressor. Si se omite, se usan los actuales "
            "de model_config.json. "
            "Ejemplo: {\"n_estimators\": 500, \"max_depth\": 6, \"learning_rate\": 0.05}"
        ),
        examples=[None],
    )
    shap_top_n: int = Field(
        default=10,
        ge=5, le=33,
        description="Número de contribuciones SHAP a incluir en la respuesta de inferencia.",
    )

    model_config = {"json_schema_extra": {
        "example": {
            "nombre_modelo": "xgboost_venta_v2",
            "hiperparametros": None,
            "shap_top_n": 10,
        }
    }}


# ─── RESPONSE — Job iniciado ──────────────────────────────────────────────────

class EntrenamientoIniciadoResponse(BaseModel):
    """Respuesta inmediata al iniciar un job de reentrenamiento (HTTP 202)."""
    job_id: str = Field(
        description="Identificador único del job. Usar para consultar estado."
    )
    estado: EstadoJob = Field(
        default=EstadoJob.PENDIENTE,
        description="Estado inicial del job.",
    )
    nombre_modelo: str = Field(
        description="Nombre base del artefacto a generar."
    )
    iniciado_en: datetime = Field(
        description="Timestamp UTC de inicio del job."
    )
    mensaje: str = Field(
        description="Descripción del job iniciado."
    )
    consultar_estado_en: str = Field(
        description="URL relativa para consultar el estado del job."
    )


# ─── RESPONSE — Estado del Job ────────────────────────────────────────────────

class MetricasEntrenamiento(BaseModel):
    """Métricas calculadas sobre el test set tras el reentrenamiento."""
    mae: float | None = Field(default=None, description="Error Absoluto Medio (Soles Constantes).")
    rmse: float | None = Field(default=None, description="Raíz del Error Cuadrático Medio.")
    mape_pct: float | None = Field(default=None, description="MAPE en % sobre el test set 2024-2025.")
    r2: float | None = Field(default=None, description="Coeficiente de determinación R².")
    n_train: int | None = Field(default=None, description="Filas usadas en entrenamiento (TRAIN split).")
    n_test: int | None = Field(default=None, description="Filas usadas en evaluación (TEST split).")
    supera_benchmark: bool | None = Field(
        default=None,
        description="True si MAPE < 15% (benchmark referencial de la tesis Oporto et al., 2024).",
    )


class ArtifactosGenerados(BaseModel):
    """Artefactos producidos y actualizados tras el reentrenamiento exitoso."""
    modelo_pkl: str = Field(description="Ruta relativa del modelo serializado.")
    params_json: str = Field(description="Ruta relativa de los hiperparámetros.")
    metricas_json: str = Field(description="Ruta relativa de las métricas oficiales.")
    model_config_json: str = Field(description="Ruta relativa de la config del servicio (actualizada).")
    modelo_recargado_en_memoria: bool = Field(
        description="True si el modelo en memoria del servicio fue reemplazado sin restart."
    )

    model_config = {"protected_namespaces": ()}



class EstadoEntrenamientoResponse(BaseModel):
    """Estado detallado de un job de reentrenamiento."""
    job_id: str
    estado: EstadoJob
    nombre_modelo: str
    iniciado_en: datetime
    completado_en: datetime | None = Field(default=None)
    duracion_segundos: float | None = Field(default=None)
    progreso: str = Field(
        description="Descripción textual del paso actual del pipeline de entrenamiento."
    )
    metricas: MetricasEntrenamiento | None = Field(
        default=None,
        description="Métricas del modelo. Solo disponibles cuando estado='completado'.",
    )
    artefactos: ArtifactosGenerados | None = Field(
        default=None,
        description="Artefactos generados. Solo disponibles cuando estado='completado'.",
    )
    error: str | None = Field(
        default=None,
        description="Mensaje de error detallado. Solo presente cuando estado='fallido'.",
    )
