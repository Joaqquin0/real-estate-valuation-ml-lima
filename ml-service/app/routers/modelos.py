"""
app/routers/modelos.py
──────────────────────
Router administrativo de gobernanza de modelos:
  - Benchmark comparativo (Producción vs Candidatos)
  - Historial de versiones en MLflow Model Registry
  - Promoción formal a Producción con Hot-Reload seguro en memoria RAM
"""

from __future__ import annotations

import os
from typing import Literal

from fastapi import APIRouter, Depends, Header, HTTPException, status

from app.core.model_loader import model_state
from app.schemas.modelos import (
    BenchmarkResponse,
    HistorialModelosResponse,
    PromoverModeloRequest,
    PromoverModeloResponse,
)
from app.services.model_registry_service import (
    obtener_benchmark,
    obtener_historial_versiones,
    promover_modelo_a_produccion,
)

router = APIRouter(
    prefix="/api/v1/admin/modelos",
    tags=["Administración — Gobernanza y Modelos"],
)


def _verificar_admin_token(x_admin_token: str | None = Header(None)) -> None:
    """Valida el token de administración si está configurado en .env."""
    expected = os.getenv("ADMIN_TOKEN", "").strip()
    if not expected:
        return
    if x_admin_token != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": "token_invalido",
                "message": "El token de administrador es inválido.",
            },
        )


@router.get(
    "/benchmark/{tipo_operacion}",
    response_model=BenchmarkResponse,
    summary="Benchmark comparativo: Producción vs Candidatos",
    description=(
        "Compara las métricas del modelo actualmente en producción (y memoria RAM) "
        "contra las versiones candidatas registradas en MLflow tras los reentrenamientos.\n\n"
        "Calcula diferencias relativas (ΔMAPE, ΔR²) y genera dictamen automático de aprobación."
    ),
)
def get_benchmark(
    tipo_operacion: Literal["venta", "alquiler"],
    _: None = Depends(_verificar_admin_token),
) -> BenchmarkResponse:
    return obtener_benchmark(tipo_operacion=tipo_operacion, state=model_state)


@router.get(
    "/historial/{tipo_operacion}",
    response_model=HistorialModelosResponse,
    summary="Historial de versiones en Model Registry",
    description="Lista todas las versiones del modelo registradas en MLflow con sus etapas y métricas.",
)
def get_historial(
    tipo_operacion: Literal["venta", "alquiler"],
    _: None = Depends(_verificar_admin_token),
) -> HistorialModelosResponse:
    return obtener_historial_versiones(tipo_operacion=tipo_operacion)


@router.post(
    "/promover",
    response_model=PromoverModeloResponse,
    summary="Aprobar y promover modelo candidato a Producción",
    description=(
        "Acción humana de gobernanza: el administrador aprueba una versión candidata.\n\n"
        "1. MLflow marca la versión como 'Production' y archiva la versión anterior.\n"
        "2. El microservicio descarga el artefacto y lo carga directamente en **memoria RAM**.\n"
        "3. Reconstruye el TreeExplainer de SHAP en RAM sin reiniciar el servidor (Hot-Reload)."
    ),
)
def promover_modelo(
    request: PromoverModeloRequest,
    _: None = Depends(_verificar_admin_token),
) -> PromoverModeloResponse:
    return promover_modelo_a_produccion(
        tipo_operacion=request.tipo_operacion,
        version=request.version,
        motivo=request.motivo,
        state=model_state,
    )
