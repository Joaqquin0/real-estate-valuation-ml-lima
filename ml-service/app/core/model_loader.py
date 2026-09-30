"""
app/core/model_loader.py
────────────────────────
Singleton de carga del modelo XGBoost + TreeExplainer SHAP + metadata.

Se ejecuta UNA SOLA VEZ al arrancar la app (lifespan FastAPI).
El TreeExplainer tarda ~500ms en inicializarse — por eso nunca debe
crearse por request.

Fuente de datos: exclusivamente PostgreSQL (inmobiliaria_ml_db).
  - Contexto distrital → tabla distrito_anio_contexto (via PostgreSQLContextProvider)
  - Dataset de entrenamiento → tablas de venta/alquiler (via entrenamiento_service)

Thread-safety:
  - El modelo XGBoost y el TreeExplainer son seguros para lectura concurrente.
  - No hay escritura después del startup → sin locks necesarios.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

import joblib
import shap
from xgboost import XGBRegressor

from app.core.data_provider import (
    IContextDataProvider,
    PostgreSQLContextProvider,
)


# ─── Estado global del modelo ─────────────────────────────────────────────────

@dataclass
class ModelState:
    """Contenedor de todos los artefactos cargados al startup."""
    modelo: XGBRegressor | None = None
    explainer: shap.TreeExplainer | None = None
    feature_cols: list[str] = field(default_factory=list)
    encoding_map: dict[str, float] = field(default_factory=dict)
    media_global: float = 0.0
    config: dict[str, Any] = field(default_factory=dict)
    data_provider: IContextDataProvider | None = None


# Instancia singleton — importada por routers y services
model_state = ModelState()


# ─── Función de carga ─────────────────────────────────────────────────────────

def cargar_modelo(state: ModelState, model_path_override: str | None = None) -> None:
    """
    Carga todos los artefactos del modelo en el ModelState proporcionado.
    Lanza RuntimeError si algún archivo crítico no existe.
    """
    metadata_path = os.getenv("METADATA_PATH", "data/features_metadata.json")
    config_path   = os.getenv("CONFIG_PATH",   "config/model_config.json")
    models_dir    = os.getenv("MODELS_DIR",    "models")

    # ── 1. Config del modelo (IPC, MAPE, version) ─────────────────────────────
    if os.path.exists(config_path):
        with open(config_path, encoding="utf-8") as f:
            state.config = json.load(f)

    # Determinar ruta del modelo:
    # 1. model_path_override
    # 2. MODEL_PATH env var
    # 3. archivo indicado en config
    # 4. default v2
    config_model_file = state.config.get("modelo_archivo", "xgboost_venta_v2.pkl")
    config_model_path = os.path.join(models_dir, config_model_file)
    model_path = (
        model_path_override
        or os.getenv("MODEL_PATH")
        or (config_model_path if os.path.exists(config_model_path) else os.path.join(models_dir, "xgboost_venta_v2.pkl"))
    )

    # ── 2. Modelo XGBoost ─────────────────────────────────────────────────────
    if not os.path.exists(model_path):
        raise RuntimeError(
            f"Modelo no encontrado: {model_path}\n"
            f"Verifica que el modelo exista en {models_dir}/"
        )
    print(f"[ModelLoader] Cargando modelo: {model_path} ...")
    state.modelo = joblib.load(model_path)
    print(f"[ModelLoader] [OK] Modelo cargado ({os.path.getsize(model_path) / 1e6:.1f} MB)")

    # ── 3. SHAP TreeExplainer (la operación más costosa del startup) ──────────
    print("[ModelLoader] Inicializando SHAP TreeExplainer ...")
    state.explainer = shap.TreeExplainer(state.modelo)
    print("[ModelLoader] [OK] SHAP TreeExplainer listo")

    # ── 4. Features metadata ──────────────────────────────────────────────────
    if not os.path.exists(metadata_path):
        raise RuntimeError(
            f"Metadata no encontrada: {metadata_path}\n"
            "Copia 'features_metadata.json' en ml-service/data/"
        )
    with open(metadata_path, encoding="utf-8") as f:
        metadata = json.load(f)

    state.feature_cols   = metadata["features"]
    state.encoding_map   = metadata["encoding_map_distrito"]
    state.media_global   = metadata["media_global_target"]
    print(
        f"[ModelLoader] [OK] Metadata: {len(state.feature_cols)} features, "
        f"{len(state.encoding_map)} distritos"
    )

    if state.config:
        ipc_info = state.config.get("ipc_actual", {})
        print(
            f"[ModelLoader] [OK] Config: version={state.config.get('modelo_version', '?')}, "
            f"IPC={ipc_info.get('valor', '?')} ({ipc_info.get('periodo', '?')})"
        )

    # ── 5. Contexto distrital desde PostgreSQL ────────────────────────────────
    print("[ModelLoader] Cargando contexto distrital desde PostgreSQL ...")
    state.data_provider = PostgreSQLContextProvider()
    n_distritos = len(state.data_provider.listar_distritos())
    print(f"[ModelLoader] [OK] Contexto distrital: {n_distritos} distritos cargados")
    print("[ModelLoader] ===== Servicio ML listo =====")


def liberar_recursos(state: ModelState) -> None:
    """Limpieza al cerrar la app (opcional — Python GC lo maneja igual)."""
    state.modelo        = None
    state.explainer     = None
    state.data_provider = None
    print("[ModelLoader] Recursos liberados.")
