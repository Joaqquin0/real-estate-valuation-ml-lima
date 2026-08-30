"""
Funciones de evaluación del modelo.
Alineado con TH-03: Evaluar modelo de venta (MAE, RMSE, MAPE).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Optional
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
)


def calcular_metricas(
    y_real: pd.Series,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """
    Calcula las 4 métricas definidas en el TI (Tabla 18).

    Returns:
        Dict con MAE, RMSE, MAPE (%), R².
    """
    metricas = {
        "MAE": mean_absolute_error(y_real, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_real, y_pred)),
        "MAPE": mean_absolute_percentage_error(y_real, y_pred) * 100,
        "R2": r2_score(y_real, y_pred),
    }

    print("=" * 50)
    print("MÉTRICAS DE EVALUACIÓN")
    print("=" * 50)
    print(f"  MAE:   S/. {metricas['MAE']:,.2f}")
    print(f"  RMSE:  S/. {metricas['RMSE']:,.2f}")
    print(f"  MAPE:  {metricas['MAPE']:.2f}%")
    print(f"  R²:    {metricas['R2']:.4f}")
    print("=" * 50)

    return metricas


def evaluar_vs_benchmark(
    metricas: Dict[str, float],
    mape_benchmark: float = 17.89,
    r2_threshold: float = 0.80,
) -> bool:
    """
    Evalúa si el modelo supera los umbrales de aceptación técnica.

    Benchmark: Oporto et al. (2024) MAPE = 17.89%
    Estándar IAAO: MAPE < 10%
    """
    mape = metricas["MAPE"]
    r2 = metricas["R2"]

    print("\n📊 EVALUACIÓN VS BENCHMARK")
    print("-" * 40)

    # MAPE
    if mape < 10:
        print(f"  ✅ MAPE {mape:.2f}% < 10% (estándar IAAO) — EXCELENTE")
    elif mape < mape_benchmark:
        print(f"  ✅ MAPE {mape:.2f}% < {mape_benchmark}% (Oporto et al.) — SUPERA BENCHMARK")
    else:
        print(f"  ❌ MAPE {mape:.2f}% ≥ {mape_benchmark}% — NO SUPERA BENCHMARK")

    # R²
    if r2 > r2_threshold:
        print(f"  ✅ R² {r2:.4f} > {r2_threshold} — ACEPTABLE")
    else:
        print(f"  ❌ R² {r2:.4f} ≤ {r2_threshold} — INSUFICIENTE")

    aprobado = (mape < mape_benchmark) and (r2 > r2_threshold)
    print(f"\n  {'✅ MODELO APROBADO' if aprobado else '❌ MODELO REQUIERE MEJORAS'}")
    return aprobado


def plot_residuos(
    y_real: pd.Series,
    y_pred: np.ndarray,
    save_path: Optional[str] = None,
) -> None:
    """Gráfico de residuos vs predichos (homocedasticidad)."""
    residuos = y_real.values - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Residuos vs predichos
    axes[0].scatter(y_pred, residuos, alpha=0.3, s=5, color="#4A90D9")
    axes[0].axhline(y=0, color="red", linestyle="--", linewidth=1)
    axes[0].set_xlabel("Valor Predicho (S/.)")
    axes[0].set_ylabel("Residuo (S/.)")
    axes[0].set_title("Residuos vs Predichos")

    # Distribución de residuos
    axes[1].hist(residuos, bins=50, edgecolor="black", alpha=0.7, color="#4A90D9")
    axes[1].axvline(x=0, color="red", linestyle="--", linewidth=1)
    axes[1].set_xlabel("Residuo (S/.)")
    axes[1].set_ylabel("Frecuencia")
    axes[1].set_title("Distribución de Residuos")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Gráfico guardado: {save_path}")
    plt.show()


def plot_real_vs_predicho(
    y_real: pd.Series,
    y_pred: np.ndarray,
    save_path: Optional[str] = None,
) -> None:
    """Gráfico de valores reales vs predichos (línea de 45°)."""
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(y_real, y_pred, alpha=0.3, s=5, color="#4A90D9")

    # Línea de predicción perfecta
    min_val = min(y_real.min(), y_pred.min())
    max_val = max(y_real.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1.5, label="Predicción perfecta")

    ax.set_xlabel("Valor Real (S/.)")
    ax.set_ylabel("Valor Predicho (S/.)")
    ax.set_title("Real vs Predicho")
    ax.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Gráfico guardado: {save_path}")
    plt.show()


def error_por_distrito(
    y_real: pd.Series,
    y_pred: np.ndarray,
    distritos: pd.Series,
    save_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Calcula MAPE por distrito para identificar zonas problemáticas.

    Returns:
        DataFrame con MAPE por distrito ordenado.
    """
    df_eval = pd.DataFrame({
        "real": y_real.values,
        "pred": y_pred,
        "distrito": distritos.values,
    })

    df_eval["ape"] = np.abs((df_eval["real"] - df_eval["pred"]) / df_eval["real"]) * 100

    mape_distrito = (
        df_eval.groupby("distrito")
        .agg(
            MAPE=("ape", "mean"),
            n_registros=("ape", "count"),
            MAE=("real", lambda x: mean_absolute_error(
                df_eval.loc[x.index, "real"],
                df_eval.loc[x.index, "pred"]
            )),
        )
        .sort_values("MAPE", ascending=False)
    )

    print("\nMAPE por Distrito:")
    print(mape_distrito.to_string())

    if save_path:
        fig, ax = plt.subplots(figsize=(12, 6))
        mape_distrito["MAPE"].plot(kind="barh", ax=ax, color="#4A90D9")
        ax.axvline(x=17.89, color="red", linestyle="--", label="Benchmark 17.89%")
        ax.axvline(x=10, color="green", linestyle="--", label="Estándar IAAO 10%")
        ax.set_xlabel("MAPE (%)")
        ax.set_title("MAPE por Distrito")
        ax.legend()
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Gráfico guardado: {save_path}")
        plt.show()

    return mape_distrito


def error_por_rango_precio(
    y_real: pd.Series,
    y_pred: np.ndarray,
    n_bins: int = 5,
    save_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Calcula MAPE por rango de precio para identificar si el modelo
    falla más en inmuebles baratos o caros.
    """
    df_eval = pd.DataFrame({
        "real": y_real.values,
        "pred": y_pred,
    })

    df_eval["rango_precio"] = pd.qcut(df_eval["real"], q=n_bins, duplicates="drop")
    df_eval["ape"] = np.abs((df_eval["real"] - df_eval["pred"]) / df_eval["real"]) * 100

    mape_rango = (
        df_eval.groupby("rango_precio", observed=True)
        .agg(
            MAPE=("ape", "mean"),
            n_registros=("ape", "count"),
        )
    )

    print("\nMAPE por Rango de Precio:")
    print(mape_rango.to_string())

    return mape_rango
