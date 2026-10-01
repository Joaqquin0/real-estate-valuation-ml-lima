"""
app/services/model_registry_service.py
──────────────────────────────────────
Servicio de gobernanza y benchmarking para modelos en MLflow Model Registry.

Funcionalidades:
  1. Comparativa de Benchmark: Modelo en Producción vs Modelos Candidatos
     - Cálculo de diferencias: ΔMAPE (%), ΔR², ΔMAE, ΔRMSE
     - Dictamen automático según estándares de la tesis (MAPE < 15% venta / 17.89% alquiler)
  2. Historial de versiones del Model Registry
  3. Promoción controlada a 'Production' con Hot-Reload seguro en memoria RAM
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Literal

import mlflow
import mlflow.xgboost
import shap
from fastapi import HTTPException, status

from app.core.mlflow_client import mlflow_manager
from app.schemas.modelos import (
    BenchmarkResponse,
    ComparativaCandidato,
    DeltaMetricas,
    HistorialModelosResponse,
    MetricasModelo,
    PromoverModeloResponse,
    VersionModeloInfo,
)

if TYPE_CHECKING:
    from app.core.model_loader import ModelState


# Benchmarks metodológicos oficiales de la tesis
BENCHMARKS_MAPE = {
    "venta": 15.0,     # Oporto et al. (2024) / Metodología Venta
    "alquiler": 17.89, # Oporto et al. (2024) / Metodología Alquiler
}


def _get_model_name(tipo_operacion: Literal["venta", "alquiler"]) -> str:
    """Obtiene el nombre oficial registrado en MLflow según el tipo de operación."""
    return os.getenv(
        f"MLFLOW_MODEL_NAME_{tipo_operacion.upper()}",
        f"xgboost_{tipo_operacion}",
    )


def _build_version_info(
    model_name: str,
    version: str,
    stage: str,
    run_id: str | None = None,
    tags: dict[str, str] | None = None,
    created_at_timestamp: int | float | None = None,
) -> VersionModeloInfo:
    """Construye un VersionModeloInfo consultando las métricas del run en MLflow."""
    metricas = MetricasModelo()
    tags_dict = tags or {}

    if run_id and mlflow_manager.check_connection():
        try:
            run_data = mlflow_manager.get_run_metrics_and_params(run_id)
            m = run_data.get("metrics", {})
            metricas = MetricasModelo(
                mape_pct=m.get("mape_pct") or m.get("mape"),
                r2=m.get("r2"),
                mae=m.get("mae"),
                rmse=m.get("rmse"),
                n_train=int(m.get("n_train")) if "n_train" in m else None,
                n_test=int(m.get("n_test")) if "n_test" in m else None,
            )
            # Combinar tags del run con los de la versión
            run_tags = run_data.get("tags", {})
            tags_dict = {**run_tags, **tags_dict}
        except Exception as e:
            print(f"[ModelRegistryService] Advertencia al leer métricas del run {run_id}: {e}")

    creado_en = None
    if created_at_timestamp:
        try:
            creado_en = datetime.fromtimestamp(created_at_timestamp / 1000.0, tz=timezone.utc)
        except Exception:
            pass

    return VersionModeloInfo(
        nombre_modelo=model_name,
        version=str(version),
        stage=stage,
        run_id=run_id,
        metricas=metricas,
        tags=tags_dict,
        creado_en=creado_en,
        fuente_artefacto=f"models:/{model_name}/{version}",
    )


def obtener_benchmark(
    tipo_operacion: Literal["venta", "alquiler"],
    state: "ModelState",
) -> BenchmarkResponse:
    """
    Construye la comparativa completa de benchmark entre la versión de producción
    activa en RAM y las versiones candidatas registradas en MLflow.
    """
    model_name = _get_model_name(tipo_operacion)
    benchmark_mape = BENCHMARKS_MAPE.get(tipo_operacion, 15.0)

    # 1. Obtener información de la versión actualmente en Producción
    prod_info: VersionModeloInfo | None = None

    if mlflow_manager.check_connection():
        prod_ver = mlflow_manager.get_production_version(model_name)
        if prod_ver:
            prod_info = _build_version_info(
                model_name=model_name,
                version=prod_ver.version,
                stage="Production",
                run_id=prod_ver.run_id,
                tags=getattr(prod_ver, "tags", {}),
                created_at_timestamp=getattr(prod_ver, "creation_timestamp", None),
            )

    # Si MLflow no tiene aún una versión en Production, usar la configuración activa en memoria
    if prod_info is None or prod_info.metricas.mape_pct is None:
        cfg = state.config if tipo_operacion == "venta" else state.config_alquiler
        prod_info = VersionModeloInfo(
            nombre_modelo=model_name,
            version=cfg.get("modelo_version", "v2"),
            stage="Production",
            run_id=None,
            metricas=MetricasModelo(
                mape_pct=cfg.get("mape_test"),
                r2=cfg.get("r2_test"),
                mae=cfg.get("mae_test"),
                rmse=cfg.get("rmse_test"),
            ),
            tags={"fuente": "config_local"},
            fuente_artefacto="memoria_activa",
        )

    # 2. Obtener versiones candidatas (no Production)
    candidatos_evaluados: list[ComparativaCandidato] = []

    if mlflow_manager.check_connection():
        candidates = mlflow_manager.get_latest_candidate_versions(model_name, limit=5)

        for cand_ver in candidates:
            cand_info = _build_version_info(
                model_name=model_name,
                version=cand_ver.version,
                stage="Candidate",
                run_id=cand_ver.run_id,
                tags=getattr(cand_ver, "tags", {}),
                created_at_timestamp=getattr(cand_ver, "creation_timestamp", None),
            )

            # Calcular deltas frente a Producción
            delta_mape = None
            delta_r2 = None
            delta_mae = None
            delta_rmse = None
            mejora_mape = False
            mejora_r2 = False

            cand_m = cand_info.metricas
            prod_m = prod_info.metricas

            if cand_m.mape_pct is not None and prod_m.mape_pct is not None:
                delta_mape = round(cand_m.mape_pct - prod_m.mape_pct, 4)
                mejora_mape = delta_mape < 0

            if cand_m.r2 is not None and prod_m.r2 is not None:
                delta_r2 = round(cand_m.r2 - prod_m.r2, 4)
                mejora_r2 = delta_r2 > 0

            if cand_m.mae is not None and prod_m.mae is not None:
                delta_mae = round(cand_m.mae - prod_m.mae, 2)

            if cand_m.rmse is not None and prod_m.rmse is not None:
                delta_rmse = round(cand_m.rmse - prod_m.rmse, 2)

            # Dictamen automático
            if cand_m.mape_pct is not None and cand_m.mape_pct <= benchmark_mape and (mejora_mape or mejora_r2):
                recomendacion = "RECOMENDADO_PARA_PRODUCCION"
                motivo = (
                    f"El candidato mejora el desempeño frente a producción "
                    f"(ΔMAPE: {delta_mape:+.2f}%, ΔR²: {delta_r2:+.4f}) y supera el benchmark exigido ({benchmark_mape}%)."
                )
            elif cand_m.mape_pct is not None and cand_m.mape_pct > benchmark_mape:
                recomendacion = "NO_RECOMENDADO"
                motivo = (
                    f"El modelo candidato tiene un MAPE de {cand_m.mape_pct:.2f}%, "
                    f"el cual no alcanza el umbral de aprobación ({benchmark_mape}%)."
                )
            else:
                recomendacion = "REVISION_MANUAL"
                motivo = "Desempeño similar o métricas incompletas. Se recomienda inspección visual en panel de control."

            candidatos_evaluados.append(
                ComparativaCandidato(
                    candidato=cand_info,
                    deltas=DeltaMetricas(
                        delta_mape_pct=delta_mape,
                        delta_r2=delta_r2,
                        delta_mae=delta_mae,
                        delta_rmse=delta_rmse,
                        mejora_mape=mejora_mape,
                        mejora_r2=mejora_r2,
                    ),
                    recomendacion=recomendacion,
                    motivo=motivo,
                )
            )

    return BenchmarkResponse(
        tipo_operacion=tipo_operacion,
        benchmark_referencial_mape=benchmark_mape,
        modelo_produccion=prod_info,
        candidatos=candidatos_evaluados,
        total_candidatos=len(candidatos_evaluados),
    )


def obtener_historial_versiones(
    tipo_operacion: Literal["venta", "alquiler"],
) -> HistorialModelosResponse:
    """Lista todas las versiones registradas para el modelo en MLflow."""
    model_name = _get_model_name(tipo_operacion)

    if not mlflow_manager.check_connection():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "mlflow_no_disponible",
                "message": "El servidor MLflow no está accesible para consultar el historial.",
            },
        )

    filter_str = f"name = '{model_name}'"
    all_versions = mlflow_manager.client.search_model_versions(
        filter_string=filter_str,
        order_by=["version_number DESC"],
        max_results=50,
    )

    versiones_info = [
        _build_version_info(
            model_name=model_name,
            version=v.version,
            stage=v.current_stage,
            run_id=v.run_id,
            tags=getattr(v, "tags", {}),
            created_at_timestamp=getattr(v, "creation_timestamp", None),
        )
        for v in all_versions
    ]

    return HistorialModelosResponse(
        tipo_operacion=tipo_operacion,
        nombre_modelo=model_name,
        total_versiones=len(versiones_info),
        versiones=versiones_info,
    )


def promover_modelo_a_produccion(
    tipo_operacion: Literal["venta", "alquiler"],
    version: str | int,
    motivo: str,
    state: "ModelState",
) -> PromoverModeloResponse:
    """
    1. Promueve la versión solicitada en MLflow Model Registry a stage 'Production'.
    2. Descarga el modelo directamente desde MLflow a la memoria RAM.
    3. Reconstruye el SHAP TreeExplainer en RAM.
    4. Actualiza atómicamente el ModelState global (Hot-Reload sin downtime).
    """
    model_name = _get_model_name(tipo_operacion)
    version_str = str(version)

    if not mlflow_manager.check_connection():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "mlflow_no_disponible",
                "message": "El servidor MLflow no está accesible para procesar la promoción.",
            },
        )

    # Identificar versión anterior en producción para archivar
    prod_anterior = mlflow_manager.get_production_version(model_name)
    ver_anterior_str = prod_anterior.version if prod_anterior else None

    # 1. Promover en MLflow
    try:
        mlflow_manager.promote_version_to_production(
            model_name=model_name,
            version=version_str,
            archive_existing=True,
            motivo=motivo,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "error_promocion_mlflow",
                "message": f"No se pudo promover la versión {version_str} en MLflow: {e}",
            },
        )

    # 2. Descargar y Cargar en RAM
    model_uri = f"models:/{model_name}/{version_str}"
    print(f"[ModelRegistryService] Cargando modelo en RAM desde {model_uri}...")

    try:
        nuevo_modelo = mlflow.xgboost.load_model(model_uri)
        print("[ModelRegistryService] Construyendo SHAP TreeExplainer en RAM...")
        nuevo_explainer = shap.TreeExplainer(nuevo_modelo)

        # 3. Hot-reload atómico en memoria
        if tipo_operacion == "venta":
            state.modelo = nuevo_modelo
            state.explainer = nuevo_explainer
            if hasattr(nuevo_modelo, "feature_names_in_"):
                state.feature_cols = list(nuevo_modelo.feature_names_in_)
            state.config["modelo_version"] = f"v{version_str}_mlflow"
            state.config["promoted_at"] = datetime.now(timezone.utc).isoformat()
        else:
            state.modelo_alquiler = nuevo_modelo
            state.explainer_alquiler = nuevo_explainer
            if hasattr(nuevo_modelo, "feature_names_in_"):
                state.feature_cols_alquiler = list(nuevo_modelo.feature_names_in_)
            state.config_alquiler["modelo_version"] = f"v{version_str}_mlflow"
            state.config_alquiler["promoted_at"] = datetime.now(timezone.utc).isoformat()

        print(f"[ModelRegistryService] [OK] Hot-Reload completado en RAM para {tipo_operacion} (versión {version_str}).")

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "error_carga_ram",
                "message": f"El modelo se promovió en MLflow pero falló al cargarse en RAM: {e}",
            },
        )

    return PromoverModeloResponse(
        exito=True,
        mensaje=f"Modelo {model_name} v{version_str} promovido exitosamente a Producción y cargado en memoria RAM.",
        tipo_operacion=tipo_operacion,
        version_promovida=version_str,
        version_anterior_archivada=ver_anterior_str,
        modelo_cargado_en_ram=True,
    )
