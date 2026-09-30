"""
app/routers/health.py
─────────────────────
Endpoints de estado y metadata del servicio.

GET /health                → estado básico del servicio (para load balancer / k8s probe)
GET /api/v1/modelo/info    → metadata completa del modelo activo
GET /api/v1/distritos      → lista de distritos disponibles para predicción
"""

from __future__ import annotations

from fastapi import APIRouter

from app.core.model_loader import model_state
from app.schemas.modelo_info import (
    DistritosResponse,
    IpcInfo,
    MetricasModelo,
    ModeloInfoResponse,
)

router = APIRouter(tags=["Estado y Metadata"])


@router.get(
    "/health",
    summary="Health check",
    description="Verifica que el servicio y el modelo están operativos. "
                "Útil para probes de Kubernetes o balanceadores de carga.",
)
def health_check() -> dict:
    modelo_ok = model_state.modelo is not None
    return {
        "status": "ok" if modelo_ok else "degraded",
        "modelo": model_state.config.get("modelo_version", "desconocido") if modelo_ok else None,
        "ipc_periodo": (
            model_state.config.get("ipc_actual", {}).get("periodo")
            if modelo_ok else None
        ),
    }


@router.get(
    "/api/v1/modelo/info",
    response_model=ModeloInfoResponse,
    summary="Información del modelo activo",
    description=(
        "Devuelve la metadata completa del modelo cargado: versión, métricas, "
        "IPC vigente, fuente de datos activa y lista de distritos disponibles."
    ),
)
def modelo_info() -> ModeloInfoResponse:
    cfg = model_state.config
    ipc_cfg = cfg.get("ipc_actual", {})

    distritos = (
        model_state.data_provider.listar_distritos()  # type: ignore[union-attr]
        if model_state.data_provider
        else sorted(model_state.encoding_map.keys())
    )

    return ModeloInfoResponse(
        version=cfg.get("modelo_version", "v2"),
        archivo=cfg.get("modelo_archivo", "xgboost_venta_v2.pkl"),
        metricas=MetricasModelo(
            mape_pct=cfg.get("mape_test", 0.0),
            r2=cfg.get("r2_test", 0.0),
            mae_soles=cfg.get("mae_test", 0.0),
            rmse_soles=cfg.get("rmse_test", 0.0),
            train_periodo=cfg.get("train_periodo", "2016-2023"),
            test_periodo=cfg.get("test_periodo", "2024-2025"),
        ),
        ipc_actual=IpcInfo(
            valor=ipc_cfg.get("valor", 0.0),
            periodo=ipc_cfg.get("periodo", ""),
            fuente=ipc_cfg.get("fuente", ""),
            nota=ipc_cfg.get("nota", ""),
        ),
        n_features=cfg.get("n_features", 33),
        n_distritos=cfg.get("n_distritos", 22),
        data_provider="postgresql",
        distritos_disponibles=distritos,
    )


@router.get(
    "/api/v1/distritos",
    response_model=DistritosResponse,
    summary="Distritos disponibles",
    description=(
        "Lista de los distritos de Lima Metropolitana disponibles en el modelo. "
        "El campo 'distrito' del request de predicción debe usar exactamente "
        "uno de estos nombres (case-sensitive)."
    ),
)
def listar_distritos() -> DistritosResponse:
    distritos = (
        model_state.data_provider.listar_distritos()  # type: ignore[union-attr]
        if model_state.data_provider
        else sorted(model_state.encoding_map.keys())
    )
    return DistritosResponse(distritos=distritos, total=len(distritos))

