"""
app/services/entrenamiento_service.py
───────────────────────────────────────
Pipeline de reentrenamiento del modelo XGBoost de venta, ejecutado
bajo petición del administrador del sistema.

Flujo completo del pipeline (se ejecuta en un thread de background):
  1. Conectar a PostgreSQL y cargar datos de entrenamiento (JOIN contextual)
  2. Construir features con la misma lógica que prediccion_service.py
     (Training-Serving Parity — ver GUIA_ENTRENAMIENTO_ML_DESDE_POSTGRES.md §D)
  3. Target encoding del distrito calculado SOLO sobre TRAIN (sin data leakage)
  4. Transformación logarítmica del target: y = log1p(precio_soles_const)
  5. Entrenamiento del XGBRegressor con los hiperparámetros indicados
  6. Evaluación de métricas sobre el TEST set (2024-2025)
  7. Exportar artefactos:
       - models/{nombre}.pkl
       - models/{nombre}_params.json
       - models/{nombre}_metricas.json
       - config/model_config.json (actualizado con nuevas métricas y encoding_map)
  8. Recargar el modelo en memoria del servicio (hot-reload sin restart)

Restricciones:
  - NO escribe en pipeline_ejecuciones ni auditoria_flags_imputacion
    (esas tablas son exclusivas del pipeline ETL)
  - Solo un job de entrenamiento puede correr a la vez
  - El hot-reload actualiza model_state global de forma thread-safe
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import HTTPException
from xgboost import XGBRegressor

from app.schemas.entrenamiento import (
    ArtifactosGenerados,
    EstadoJob,
    MetricasEntrenamiento,
)

# Guard para importar TYPE_CHECKING sin ciclos
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from app.core.model_loader import ModelState


# ─── Registry de jobs en memoria ─────────────────────────────────────────────
# Diccionario que mapea job_id → estado del job.
# En una arquitectura distribuida esto debería ser Redis, pero para el caso
# de un solo servidor es suficiente en memoria.

_job_registry: dict[str, dict[str, Any]] = {}
_job_lock = threading.Lock()

# Semáforo para garantizar que solo corra 1 job a la vez
_training_semaphore = threading.Semaphore(1)


# ─── Acceso al registry ───────────────────────────────────────────────────────

def get_job_status(job_id: str) -> dict[str, Any]:
    """Retorna el estado de un job. HTTPException 404 si no existe."""
    with _job_lock:
        if job_id not in _job_registry:
            raise HTTPException(
                status_code=404,
                detail={"error": "job_no_encontrado", "job_id": job_id},
            )
        return dict(_job_registry[job_id])


def get_latest_job() -> dict[str, Any] | None:
    """Retorna el último job registrado o None si no hay ninguno."""
    with _job_lock:
        if not _job_registry:
            return None
        ultimo_id = list(_job_registry.keys())[-1]
        return dict(_job_registry[ultimo_id])


def list_all_jobs() -> list[dict[str, Any]]:
    """Retorna la lista de todos los jobs registrados."""
    with _job_lock:
        return [dict(j) for j in _job_registry.values()]


def _update_job(job_id: str, **kwargs: Any) -> None:
    """Actualiza campos del job en el registry de forma thread-safe."""
    with _job_lock:
        _job_registry[job_id].update(kwargs)


# ─── Feature Engineering (Training-Serving Parity) ───────────────────────────

def _construir_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construye el conjunto de features con idéntica lógica a prediccion_service.py.

    IMPORTANTE: cualquier cambio aquí debe replicarse en prediccion_service.py
    para mantener Training-Serving Parity (ver Guía §D).

    Args:
        df: DataFrame con columnas crudas del JOIN de PostgreSQL.

    Returns:
        DataFrame con las features engineered listas para el modelo.
    """
    df = df.copy()

    # Renombrar columnas de la BD al nombre de feature del modelo
    df.rename(columns={
        "superficie_m2":   "Superficie",
        "habitaciones":    "Habitaciones",
        "banos":           "Banios",
        "garajes":         "Garajes",
        "piso":            "Piso",
        "antiguedad_anios":"Antiguedad",
        "vista_exterior":  "Vista_Exterior",
        "anio":            "Anio",
        "trimestre":       "Trimestre",
        "pct_nse_a":       "pct_NSE_A",
        "pct_nse_b":       "pct_NSE_B",
        "pct_nse_c":       "pct_NSE_C",
        "pct_nse_d":       "pct_NSE_D",
        "pct_nse_e":       "pct_NSE_E",
    }, inplace=True)

    # Features engineered
    df["periodo_numerico"]  = df["Anio"] * 4 + df["Trimestre"]
    df["m2_por_habitacion"] = df["Superficie"] / (df["Habitaciones"] + 1)
    df["ratio_banios_hab"]  = df["Banios"] / (df["Habitaciones"] + 0.1)
    df["tiene_garaje"]      = (df["Garajes"] > 0).astype(float)
    df["es_piso_alto"]      = (df["Piso"] >= 8).astype(float)
    df["superficie_cuadrado"] = (df["Superficie"] / 100) ** 2

    # Vista_Exterior como float
    df["Vista_Exterior"] = df["Vista_Exterior"].astype(float)

    return df


def _target_encoding_distrito(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    col_distrito: str = "distrito",
    col_target: str = "Precio_Soles_Const",
    smoothing: float = 10.0,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float], float]:
    """
    Target Encoding suavizado del distrito.

    Calculado SOLO sobre df_train para evitar data leakage temporal.
    Los distritos no vistos en train reciben la media global.

    Returns:
        (df_train, df_test, encoding_map, media_global)
    """
    media_global = float(df_train[col_target].mean())

    stats = df_train.groupby(col_distrito)[col_target].agg(["mean", "count"])
    stats["distrito_encoded"] = (
        (stats["count"] * stats["mean"] + smoothing * media_global)
        / (stats["count"] + smoothing)
    )
    encoding_map: dict[str, float] = stats["distrito_encoded"].to_dict()

    df_train = df_train.copy()
    df_test  = df_test.copy()

    df_train["distrito_encoded"] = df_train[col_distrito].map(encoding_map)
    df_test["distrito_encoded"]  = df_test[col_distrito].map(encoding_map).fillna(media_global)

    n_unseen = df_test["distrito_encoded"].isna().sum()
    if n_unseen > 0:
        print(f"[EntrenamientoService] {n_unseen} distritos no vistos en test -> media global")

    return df_train, df_test, encoding_map, media_global


def _calcular_metricas(
    y_real: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """
    Calcula MAE, RMSE, MAPE y R² en escala original (soles constantes).
    y_real y y_pred están en escala logarítmica → se aplica expm1 antes.
    """
    y_real_orig = np.expm1(y_real)
    y_pred_orig = np.expm1(y_pred)

    mae  = float(np.mean(np.abs(y_real_orig - y_pred_orig)))
    rmse = float(np.sqrt(np.mean((y_real_orig - y_pred_orig) ** 2)))
    mape = float(np.mean(np.abs((y_real_orig - y_pred_orig) / (y_real_orig + 1e-8))) * 100)
    ss_res = np.sum((y_real_orig - y_pred_orig) ** 2)
    ss_tot = np.sum((y_real_orig - np.mean(y_real_orig)) ** 2)
    r2   = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    return {"mae": mae, "rmse": rmse, "mape_pct": mape, "r2": r2}


# ─── Pipeline principal (se ejecuta en background thread) ─────────────────────

def _run_training_pipeline(
    job_id: str,
    nombre_modelo: str,
    hiperparametros: dict[str, Any] | None,
    shap_top_n: int,
    state: "ModelState",
) -> None:
    """
    Pipeline completo de reentrenamiento. Ejecutado en un thread secundario.

    Actualiza el registry de jobs en cada paso para que el endpoint de estado
    pueda reportar el progreso en tiempo real.
    """
    inicio = time.time()

    # Hiperparámetros por defecto del modelo actual
    DEFAULT_PARAMS: dict[str, Any] = {
        "n_estimators":     500,
        "max_depth":        6,
        "learning_rate":    0.05,
        "subsample":        0.8,
        "colsample_bytree": 0.8,
        "reg_alpha":        0.1,
        "reg_lambda":       1.0,
        "min_child_weight": 3,
        "random_state":     42,
        "n_jobs":          -1,
        "tree_method":     "hist",
    }
    params = hiperparametros or DEFAULT_PARAMS

    try:
        # ── Paso 1: Conectar a DB y cargar datos ──────────────────────────────
        _update_job(job_id, progreso="[1/8] Conectando a PostgreSQL y cargando datos...")
        print(f"[EntrenamientoService][{job_id}] Paso 1: Cargando datos de PostgreSQL...")

        from app.core.db_provider import PostgreSQLTrainingProvider
        db_provider = PostgreSQLTrainingProvider()

        if not db_provider.check_connection():
            raise ConnectionError(
                "No se pudo conectar a PostgreSQL. "
                "Verifica DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD en .env"
            )

        df_train_raw = db_provider.load_dataset_venta(split="TRAIN")
        df_test_raw  = db_provider.load_dataset_venta(split="TEST")

        _update_job(
            job_id,
            progreso=f"[1/8] Datos cargados: {len(df_train_raw):,} train / {len(df_test_raw):,} test.",
        )

        # ── Paso 2: Feature Engineering ───────────────────────────────────────
        _update_job(job_id, progreso="[2/8] Construyendo features (Training-Serving Parity)...")
        print(f"[EntrenamientoService][{job_id}] Paso 2: Feature Engineering...")

        df_train = _construir_features(df_train_raw)
        df_test  = _construir_features(df_test_raw)

        # ── Paso 3: Target Encoding del distrito ──────────────────────────────
        _update_job(job_id, progreso="[3/8] Aplicando Target Encoding del distrito...")
        print(f"[EntrenamientoService][{job_id}] Paso 3: Target Encoding...")

        # Target en escala original para calcular el encoding
        df_train["Precio_Soles_Const"] = df_train_raw["precio_soles_const"]
        df_test["Precio_Soles_Const"]  = df_test_raw["precio_soles_const"]

        df_train, df_test, encoding_map, media_global = _target_encoding_distrito(
            df_train, df_test,
            col_distrito="distrito",
            col_target="Precio_Soles_Const",
        )

        # ── Paso 4: Preparar X e y ────────────────────────────────────────────
        _update_job(job_id, progreso="[4/8] Preparando matrices X e y (log1p target)...")
        print(f"[EntrenamientoService][{job_id}] Paso 4: Preparando X/y...")

        FEATURE_COLS = [
            "Anio", "Trimestre", "Superficie", "Habitaciones", "Banios",
            "Garajes", "Piso", "Vista_Exterior", "Antiguedad",
            "pct_NSE_A", "pct_NSE_B", "pct_NSE_C", "pct_NSE_D", "pct_NSE_E",
            "tasa_robo", "tasa_hurto", "poblacion_proyectada", "area_distrito_km2",
            "distancia_centro_km", "dist_colegio_km", "dist_hospital_km",
            "dist_estacion_transporte_km", "dist_centro_comercial_km",
            "dist_parque_km", "dist_universidad_km", "densidad_hab_km2",
            "periodo_numerico", "m2_por_habitacion", "ratio_banios_hab",
            "tiene_garaje", "es_piso_alto", "superficie_cuadrado",
            "distrito_encoded",
        ]

        # Verificar que todas las features existen
        missing = [c for c in FEATURE_COLS if c not in df_train.columns]
        if missing:
            raise ValueError(
                f"Faltan columnas en el dataset de entrenamiento: {missing}. "
                "Revisa el JOIN en db_provider.py o la guía de entrenamiento."
            )

        X_train = df_train[FEATURE_COLS].astype(float)
        X_test  = df_test[FEATURE_COLS].astype(float)

        # Transformación logarítmica del target (Soles Constantes)
        y_train = np.log1p(df_train["Precio_Soles_Const"].astype(float).values)
        y_test  = np.log1p(df_test["Precio_Soles_Const"].astype(float).values)

        # ── Paso 5: Entrenamiento XGBoost ─────────────────────────────────────
        _update_job(job_id, progreso=f"[5/8] Entrenando XGBRegressor (params: n_estimators={params.get('n_estimators', '?')})...")
        print(f"[EntrenamientoService][{job_id}] Paso 5: Entrenando modelo...")

        modelo = XGBRegressor(**params)
        modelo.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        # ── Paso 6: Métricas sobre el test set ───────────────────────────────
        _update_job(job_id, progreso="[6/8] Evaluando métricas sobre test set 2024-2025...")
        print(f"[EntrenamientoService][{job_id}] Paso 6: Calculando métricas...")

        y_pred_test = modelo.predict(X_test)
        metricas_dict = _calcular_metricas(y_test, y_pred_test)
        metricas_dict["n_train"] = len(X_train)
        metricas_dict["n_test"]  = len(X_test)
        metricas_dict["supera_benchmark"] = metricas_dict["mape_pct"] < 15.0

        print(
            f"[EntrenamientoService][{job_id}] Metricas: "
            f"MAPE={metricas_dict['mape_pct']:.2f}% | R2={metricas_dict['r2']:.4f}"
        )

        # ── Paso 7: Exportar artefactos ───────────────────────────────────────
        _update_job(job_id, progreso="[7/8] Exportando artefactos (pkl, params, metricas, config)...")
        print(f"[EntrenamientoService][{job_id}] Paso 7: Guardando artefactos...")

        models_dir = os.getenv("MODELS_DIR", "models")
        config_path = os.getenv("CONFIG_PATH", "config/model_config.json")
        os.makedirs(models_dir, exist_ok=True)

        # 7a. Modelo serializado
        pkl_path = os.path.join(models_dir, f"{nombre_modelo}.pkl")
        joblib.dump(modelo, pkl_path)

        # 7b. Hiperparámetros
        params_path = os.path.join(models_dir, f"{nombre_modelo}_params.json")
        with open(params_path, "w", encoding="utf-8") as f:
            json.dump(params, f, indent=2, default=str)

        # 7c. Métricas oficiales
        metricas_path = os.path.join(models_dir, f"{nombre_modelo}_metricas.json")
        with open(metricas_path, "w", encoding="utf-8") as f:
            json.dump(metricas_dict, f, indent=2)

        # 7d. Actualizar config del servicio (model_config.json)
        # Lee la config actual y actualiza solo los campos relevantes
        if os.path.exists(config_path):
            with open(config_path, encoding="utf-8") as f:
                config_actual = json.load(f)
        else:
            config_actual = {}

        now_utc = datetime.now(timezone.utc)
        quarter = (now_utc.month - 1) // 3 + 1
        ahora_str = f"{now_utc.year}-Q{quarter}"
        # Extraer versión del nombre
        version = nombre_modelo.split("_")[-1] if "_" in nombre_modelo else "v2"

        config_actualizada = {
            **config_actual,
            "modelo_version":  version,
            "modelo_archivo":  f"{nombre_modelo}.pkl",
            "mape_test":       round(metricas_dict["mape_pct"], 4),
            "r2_test":         round(metricas_dict["r2"], 4),
            "mae_test":        round(metricas_dict["mae"], 2),
            "rmse_test":       round(metricas_dict["rmse"], 2),
            "n_features":      len(FEATURE_COLS),
            "feature_cols":    FEATURE_COLS,
            "encoding_map_distrito": encoding_map,
            "media_global_target":   media_global,
            "shap_top_n":      shap_top_n,
            "n_distritos":     len(encoding_map),
            "train_periodo":   "2016-2023",
            "test_periodo":    "2024-2025",
            "retrained_at":    now_utc.isoformat(),
        }

        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config_actualizada, f, indent=2, ensure_ascii=False)

        # 7e. Actualizar features_metadata.json
        metadata_path = os.getenv("METADATA_PATH", "data/features_metadata.json")
        metadata_actualizada = {
            "features": FEATURE_COLS,
            "encoding_map_distrito": encoding_map,
            "media_global_target": media_global,
            "target": "Precio_Soles_Const",
            "transformacion": "log1p",
            "updated_at": now_utc.isoformat(),
        }
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata_actualizada, f, indent=2, ensure_ascii=False)

        # ── Paso 8: Hot-reload del modelo en memoria ──────────────────────────
        _update_job(job_id, progreso="[8/8] Recargando modelo en memoria del servicio (hot-reload)...")
        print(f"[EntrenamientoService][{job_id}] Paso 8: Hot-reload en memoria...")

        from app.core.model_loader import cargar_modelo
        cargar_modelo(state, model_path_override=pkl_path)

        duracion = round(time.time() - inicio, 2)

        artefactos = ArtifactosGenerados(
            modelo_pkl=pkl_path,
            params_json=params_path,
            metricas_json=metricas_path,
            model_config_json=config_path,
            modelo_recargado_en_memoria=True,
        )
        metricas_obj = MetricasEntrenamiento(**metricas_dict)

        _update_job(
            job_id,
            estado=EstadoJob.COMPLETADO,
            progreso="[8/8] Entrenamiento completado exitosamente.",
            completado_en=datetime.now(timezone.utc),
            duracion_segundos=duracion,
            metricas=metricas_obj,
            artefactos=artefactos,
        )
        print(
            f"[EntrenamientoService][{job_id}] Completado en {duracion}s. "
            f"MAPE={metricas_dict['mape_pct']:.2f}%"
        )

    except Exception as exc:
        duracion = round(time.time() - inicio, 2)
        error_msg = str(exc)
        print(f"[EntrenamientoService][{job_id}] ERROR: {error_msg}")
        _update_job(
            job_id,
            estado=EstadoJob.FALLIDO,
            progreso="Pipeline de entrenamiento fallido.",
            completado_en=datetime.now(timezone.utc),
            duracion_segundos=duracion,
            error=error_msg,
        )
    finally:
        _training_semaphore.release()


# ─── Función pública: iniciar job ─────────────────────────────────────────────

def iniciar_entrenamiento_venta(
    nombre_modelo: str,
    hiperparametros: dict[str, Any] | None,
    shap_top_n: int,
    state: "ModelState",
) -> str:
    """
    Registra y lanza el job de reentrenamiento en un thread de background.

    Returns:
        job_id: Identificador único del job iniciado.

    Raises:
        HTTPException 409: Si ya hay un job de entrenamiento en curso.
    """
    if not _training_semaphore.acquire(blocking=False):
        raise HTTPException(
            status_code=409,
            detail={
                "error": "entrenamiento_en_curso",
                "message": (
                    "Ya hay un job de entrenamiento en ejecución. "
                    "Espera a que termine antes de iniciar uno nuevo."
                ),
            },
        )

    job_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)

    with _job_lock:
        _job_registry[job_id] = {
            "job_id":             job_id,
            "estado":             EstadoJob.EN_PROGRESO,
            "nombre_modelo":      nombre_modelo,
            "iniciado_en":        now,
            "completado_en":      None,
            "duracion_segundos":  None,
            "progreso":           "[0/8] Iniciando pipeline de entrenamiento...",
            "metricas":           None,
            "artefactos":         None,
            "error":              None,
        }

    thread = threading.Thread(
        target=_run_training_pipeline,
        args=(job_id, nombre_modelo, hiperparametros, shap_top_n, state),
        daemon=True,
        name=f"entrenamiento-{job_id[:8]}",
    )
    thread.start()

    print(f"[EntrenamientoService] Job {job_id} iniciado en thread '{thread.name}'.")
    return job_id
