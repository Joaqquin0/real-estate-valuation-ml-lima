"""
Pipeline de entrenamiento del modelo XGBoost.
Alineado con TH-02: Entrenar modelo de venta.
"""

import json
import os
import time
from typing import Dict, Any, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from xgboost import XGBRegressor

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


def entrenar_baseline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: Dict[str, Any],
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[pd.Series] = None,
    early_stopping_rounds: int = 50,
) -> XGBRegressor:
    """
    Entrena un modelo XGBoost con parámetros dados (baseline).

    Si se provee un set de validación, usa early stopping.
    """
    modelo = XGBRegressor(**params)

    fit_params = {}
    if X_val is not None and y_val is not None:
        fit_params["eval_set"] = [(X_val, y_val)]
        fit_params["verbose"] = 50

    start = time.time()
    modelo.fit(X_train, y_train, **fit_params)
    elapsed = time.time() - start

    print(f"Modelo entrenado en {elapsed:.1f}s | "
          f"n_estimators usados: {modelo.best_iteration if hasattr(modelo, 'best_iteration') and modelo.best_iteration else params.get('n_estimators', '?')}")

    return modelo


def validacion_cruzada_temporal(
    X: pd.DataFrame,
    y: pd.Series,
    params: Dict[str, Any],
    n_splits: int = 5,
    scoring: str = "neg_mean_absolute_percentage_error",
) -> Dict[str, Any]:
    """
    Realiza validación cruzada temporal con TimeSeriesSplit.

    Returns:
        Dict con scores por fold y estadísticas.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    modelo = XGBRegressor(**params)

    scores = cross_val_score(
        modelo, X, y,
        cv=tscv,
        scoring=scoring,
        n_jobs=-1,
    )

    # MAPE viene negativo por convención de sklearn
    mape_scores = -scores * 100  # Convertir a porcentaje positivo

    resultado = {
        "scores_por_fold": mape_scores.tolist(),
        "mape_media": float(mape_scores.mean()),
        "mape_std": float(mape_scores.std()),
        "n_splits": n_splits,
    }

    print(f"Validación Cruzada Temporal ({n_splits} folds):")
    for i, s in enumerate(mape_scores):
        print(f"  Fold {i+1}: MAPE = {s:.2f}%")
    print(f"  Media: {resultado['mape_media']:.2f}% ± {resultado['mape_std']:.2f}%")

    return resultado


def tuning_optuna(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_trials: int = 50,
    n_splits: int = 5,
    random_state: int = 42,
    timeout: int = 600,
) -> Tuple[Dict[str, Any], Any]:
    """
    Tuning de hiperparámetros con Optuna.

    Args:
        n_trials: Número de combinaciones a probar.
        timeout: Tiempo máximo en segundos.

    Returns:
        Tupla (mejores_params, estudio_optuna).
    """
    if not OPTUNA_AVAILABLE:
        raise ImportError("Optuna no está instalado. Ejecuta: pip install optuna")

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "random_state": random_state,
            "n_jobs": -1,
        }

        tscv = TimeSeriesSplit(n_splits=n_splits)
        modelo = XGBRegressor(**params)

        scores = cross_val_score(
            modelo, X_train, y_train,
            cv=tscv,
            scoring="neg_mean_absolute_percentage_error",
            n_jobs=1,  # Ya usamos -1 dentro del modelo
        )

        return -scores.mean()  # Minimizar MAPE

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)

    best_params = study.best_trial.params
    best_params["random_state"] = random_state
    best_params["n_jobs"] = -1

    print(f"\nMejor MAPE encontrado: {study.best_value * 100:.2f}%")
    print(f"Mejores hiperparámetros: {json.dumps(best_params, indent=2)}")

    return best_params, study


def guardar_modelo(
    modelo: XGBRegressor,
    params: Dict[str, Any],
    metricas: Dict[str, Any],
    models_dir: str,
    nombre: str = "xgboost_venta",
) -> str:
    """
    Guarda el modelo entrenado y sus metadatos.

    Genera:
        - {nombre}.pkl: Modelo serializado
        - {nombre}_params.json: Hiperparámetros
        - {nombre}_metricas.json: Métricas de evaluación
    """
    os.makedirs(models_dir, exist_ok=True)

    # Guardar modelo
    model_path = os.path.join(models_dir, f"{nombre}.pkl")
    joblib.dump(modelo, model_path)

    # Guardar hiperparámetros
    params_path = os.path.join(models_dir, f"{nombre}_params.json")
    with open(params_path, "w") as f:
        json.dump(params, f, indent=2, default=str)

    # Guardar métricas
    metricas_path = os.path.join(models_dir, f"{nombre}_metricas.json")
    with open(metricas_path, "w") as f:
        json.dump(metricas, f, indent=2, default=str)

    print(f"Modelo guardado en: {model_path}")
    return model_path


def cargar_modelo(model_path: str) -> XGBRegressor:
    """Carga un modelo previamente guardado."""
    modelo = joblib.load(model_path)
    print(f"Modelo cargado: {model_path}")
    return modelo
