"""
app/core/model_loader.py
────────────────────────
Singleton de carga del modelo XGBoost + TreeExplainer SHAP + metadata.

Se ejecuta UNA SOLA VEZ al arrancar la app (lifespan FastAPI).
El TreeExplainer tarda ~500ms en inicializarse — por eso nunca debe
crearse por request.

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
    CSVContextProvider,
    IContextDataProvider,
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

def cargar_modelo(state: ModelState) -> None:
    """
    Carga todos los artefactos del modelo en el ModelState proporcionado.
    Lanza RuntimeError si algún archivo crítico no existe.
    """
    # Rutas desde variables de entorno (con defaults relativos al servicio)
    model_path    = os.getenv("MODEL_PATH",    "models/xgboost_venta_v2.pkl")
    metadata_path = os.getenv("METADATA_PATH", "data/features_metadata.json")
    context_path  = os.getenv("CONTEXT_CSV_PATH", "data/distrito_contexto_ref.csv")
    config_path   = os.getenv("CONFIG_PATH",   "config/model_config.json")

    # ── 1. Modelo XGBoost ─────────────────────────────────────────────────────
    if not os.path.exists(model_path):
        raise RuntimeError(
            f"Modelo no encontrado: {model_path}\n"
            "Copia 'xgboost_venta_v2.pkl' en ml-service/models/"
        )
    print(f"[ModelLoader] Cargando modelo: {model_path} ...")
    state.modelo = joblib.load(model_path)
    print(f"[ModelLoader] [OK] Modelo cargado ({os.path.getsize(model_path) / 1e6:.1f} MB)")

    # ── 2. SHAP TreeExplainer (la operación más costosa del startup) ──────────
    print("[ModelLoader] Inicializando SHAP TreeExplainer ...")
    state.explainer = shap.TreeExplainer(state.modelo)
    print("[ModelLoader] [OK] SHAP TreeExplainer listo")

    # ── 3. Features metadata ──────────────────────────────────────────────────
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

    # ── 4. Config del modelo (IPC, MAPE, versión) ─────────────────────────────
    if not os.path.exists(config_path):
        raise RuntimeError(
            f"Config no encontrada: {config_path}\n"
            "Verifica que existe 'config/model_config.json'"
        )
    with open(config_path, encoding="utf-8") as f:
        state.config = json.load(f)
    print(
        f"[ModelLoader] [OK] Config: version={state.config['modelo_version']}, "
        f"IPC={state.config['ipc_actual']['valor']} ({state.config['ipc_actual']['periodo']})"
    )

    # ── 5. Data provider (CSV fase 1, MongoDB fase 2) ─────────────────────────
    provider_type = os.getenv("DATA_PROVIDER", "csv").lower()
    if provider_type == "csv":
        state.data_provider = CSVContextProvider(context_path)
    elif provider_type == "mongo":
        from app.core.data_provider import MongoContextProvider  # type: ignore
        state.data_provider = MongoContextProvider(
            connection_string=os.getenv("MONGO_URI", ""),
            db=os.getenv("MONGO_DB", "tasacion_db"),
            collection=os.getenv("MONGO_COLLECTION_CONTEXTO", "distrito_contexto"),
        )
    else:
        raise ValueError(f"DATA_PROVIDER desconocido: '{provider_type}'. Usa 'csv' o 'mongo'.")

    print(f"[ModelLoader] [OK] Data provider: {provider_type.upper()}")
    print("[ModelLoader] ===== Servicio ML listo =====")


def liberar_recursos(state: ModelState) -> None:
    """Limpieza al cerrar la app (opcional — Python GC lo maneja igual)."""
    state.modelo    = None
    state.explainer = None
    state.data_provider = None
    print("[ModelLoader] Recursos liberados.")
