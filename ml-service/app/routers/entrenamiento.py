"""
app/routers/entrenamiento.py
────────────────────────────
Router administrativo para el reentrenamiento del modelo.

Endpoints:
  POST /api/v1/admin/entrenamiento/venta
    → Lanza un job de reentrenamiento en background (HTTP 202 inmediato)
    → Requiere header: X-Admin-Token: <valor de ADMIN_TOKEN en .env>

  GET /api/v1/admin/entrenamiento/estado/{job_id}
    → Consulta el estado y progreso del job

Seguridad:
  Autenticación mediante header X-Admin-Token (token estático configurado en .env).
  En una arquitectura productiva esto debería ser JWT con rol 'admin'.

Diseño del reentrenamiento:
  - El pipeline corre en background (thread daemon) para no bloquear la API.
  - Solo puede haber 1 job activo a la vez (semáforo en entrenamiento_service.py).
  - El modelo en memoria se recarga automáticamente al finalizar (hot-reload).
  - No se persiste nada en BD — los artefactos van a models/ y config/.
"""

from __future__ import annotations

import os
from datetime import datetime

from fastapi import APIRouter, Depends, Header, HTTPException, status

from app.core.model_loader import model_state
from app.schemas.entrenamiento import (
    EntrenamientoIniciadoResponse,
    EntrenamientoVentaRequest,
    EstadoEntrenamientoResponse,
    EstadoJob,
    MetricasEntrenamiento,
    ArtifactosGenerados,
)
from app.services.entrenamiento_service import (
    get_job_status,
    get_latest_job,
    iniciar_entrenamiento_venta,
    list_all_jobs,
)

router = APIRouter(
    prefix="/api/v1/admin/entrenamiento",
    tags=["Administración — Entrenamiento"],
)


# ─── Dependencia de autenticación admin ───────────────────────────────────────

def _verificar_admin_token(x_admin_token: str | None = Header(None)) -> None:
    """
    Valida el header X-Admin-Token contra la variable de entorno ADMIN_TOKEN si está definida.
    Si ADMIN_TOKEN no está configurado, permite el acceso (entorno de desarrollo/pruebas).
    """
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


# ─── Endpoints ────────────────────────────────────────────────────────────────

@router.post(
    "/venta",
    response_model=EntrenamientoIniciadoResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Iniciar reentrenamiento del modelo de venta",
    description=(
        "Lanza el pipeline de reentrenamiento del modelo XGBoost de venta "
        "utilizando los datos actuales en PostgreSQL (`inmobiliaria_ml_db`).\n\n"
        "**Flujo del pipeline** (corre en background):\n"
        "1. Conecta a PostgreSQL y ejecuta el JOIN contextual\n"
        "2. Construye features con Training-Serving Parity\n"
        "3. Aplica Target Encoding del distrito (solo sobre split TRAIN)\n"
        "4. Transforma el target: `y = log1p(precio_soles_const)`\n"
        "5. Entrena el XGBRegressor con los hiperparámetros indicados\n"
        "6. Evalúa métricas sobre el TEST set (2024-2025)\n"
        "7. Exporta artefactos: `.pkl`, `_params.json`, `_metricas.json`, `model_config.json`\n"
        "8. Recarga el modelo en memoria del servicio (hot-reload sin restart)\n\n"
        "Retorna inmediatamente un `job_id` para consultar el progreso.\n\n"
        "**Autenticación:** Requiere header `X-Admin-Token`.\n\n"
        "**Restricción:** Solo puede haber 1 job activo a la vez (HTTP 409 si hay uno en curso)."
    ),
    responses={
        202: {"description": "Job iniciado exitosamente. Consultar estado con el job_id retornado."},
        401: {"description": "Token de administrador inválido."},
        409: {"description": "Ya hay un job de entrenamiento en curso."},
        503: {"description": "ADMIN_TOKEN no configurado en el servidor."},
    },
)
def iniciar_entrenamiento(
    request: EntrenamientoVentaRequest,
    _: None = Depends(_verificar_admin_token),
) -> EntrenamientoIniciadoResponse:
    """
    Inicia el reentrenamiento en background y retorna el job_id (HTTP 202).
    """
    job_id = iniciar_entrenamiento_venta(
        nombre_modelo=request.nombre_modelo,
        hiperparametros=request.hiperparametros,
        shap_top_n=request.shap_top_n,
        state=model_state,
    )

    return EntrenamientoIniciadoResponse(
        job_id=job_id,
        estado=EstadoJob.EN_PROGRESO,
        nombre_modelo=request.nombre_modelo,
        iniciado_en=datetime.utcnow(),
        mensaje=(
            f"Job de reentrenamiento '{request.nombre_modelo}' iniciado. "
            "El pipeline corre en background y puede tardar varios minutos."
        ),
        consultar_estado_en=f"/api/v1/admin/entrenamiento/estado/{job_id}",
    )


@router.get(
    "/jobs",
    summary="Listar historial de jobs de entrenamiento",
    description="Retorna la lista de todos los jobs de entrenamiento ejecutados en la sesión actual.",
)
def listar_jobs(_: None = Depends(_verificar_admin_token)) -> list[dict]:
    return list_all_jobs()


@router.get(
    "/estado",
    summary="Consultar estado del último job o disponibilidad",
    description="Retorna el estado del último job de entrenamiento registrado, o un mensaje indicando que no hay jobs activos.",
)
def ultimo_estado_entrenamiento(_: None = Depends(_verificar_admin_token)) -> dict:
    job_data = get_latest_job()
    if not job_data:
        return {
            "mensaje": "No se ha ejecutado ningún job de reentrenamiento en esta sesión.",
            "disponible_para_entrenar": True,
        }
    return job_data


@router.get(
    "/estado/{job_id}",
    response_model=EstadoEntrenamientoResponse,
    summary="Consultar estado del job de entrenamiento",
    description=(
        "Retorna el estado actual, progreso y métricas de un job de reentrenamiento.\n\n"
        "**Estados posibles:**\n"
        "- `en_progreso`: El pipeline está corriendo (revisa `progreso` para el paso actual).\n"
        "- `completado`: El modelo fue reentrenado y recargado en memoria. "
        "Revisa `metricas` y `artefactos`.\n"
        "- `fallido`: Ocurrió un error. Revisa `error` para el detalle.\n\n"
        "**Autenticación:** Requiere header `X-Admin-Token`."
    ),
    responses={
        200: {"description": "Estado del job retornado correctamente."},
        401: {"description": "Token de administrador inválido."},
        404: {"description": "job_id no encontrado."},
    },
)
def estado_entrenamiento(
    job_id: str,
    _: None = Depends(_verificar_admin_token),
) -> EstadoEntrenamientoResponse:
    """
    Consulta el estado de un job de reentrenamiento por su job_id.
    """
    job_data = get_job_status(job_id)

    # Convertir MetricasEntrenamiento si viene como objeto o dict
    metricas = job_data.get("metricas")
    if isinstance(metricas, dict):
        metricas = MetricasEntrenamiento(**metricas)

    artefactos = job_data.get("artefactos")
    if isinstance(artefactos, dict):
        artefactos = ArtifactosGenerados(**artefactos)

    return EstadoEntrenamientoResponse(
        job_id=job_data["job_id"],
        estado=job_data["estado"],
        nombre_modelo=job_data["nombre_modelo"],
        iniciado_en=job_data["iniciado_en"],
        completado_en=job_data.get("completado_en"),
        duracion_segundos=job_data.get("duracion_segundos"),
        progreso=job_data.get("progreso", ""),
        metricas=metricas,
        artefactos=artefactos,
        error=job_data.get("error"),
    )
