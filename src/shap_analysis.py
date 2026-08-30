"""
Análisis de explicabilidad con SHAP.
Alineado con US-18: Ver explicación SHAP de venta.
"""

import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from xgboost import XGBRegressor


def crear_explainer(modelo: XGBRegressor) -> shap.TreeExplainer:
    """Crea un TreeExplainer optimizado para XGBoost."""
    explainer = shap.TreeExplainer(modelo)
    print("SHAP TreeExplainer creado correctamente.")
    return explainer


def calcular_shap_values(
    explainer: shap.TreeExplainer,
    X: pd.DataFrame,
) -> shap.Explanation:
    """
    Calcula los SHAP values para un conjunto de datos.

    Returns:
        Objeto shap.Explanation con los valores.
    """
    shap_values = explainer(X)
    print(f"SHAP values calculados: {shap_values.shape}")
    return shap_values


def plot_summary_beeswarm(
    shap_values: shap.Explanation,
    max_display: int = 20,
    save_path: Optional[str] = None,
) -> None:
    """
    Summary Plot (beeswarm): muestra la importancia global de cada variable
    junto con la dirección del efecto.
    """
    plt.figure(figsize=(12, 8))
    shap.plots.beeswarm(shap_values, max_display=max_display, show=False)
    plt.title("Importancia Global de Variables (SHAP Beeswarm)")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Gráfico guardado: {save_path}")
    plt.show()


def plot_bar_importance(
    shap_values: shap.Explanation,
    max_display: int = 15,
    save_path: Optional[str] = None,
) -> None:
    """
    Bar Plot: Top N features por importancia SHAP media absoluta.
    """
    plt.figure(figsize=(10, 6))
    shap.plots.bar(shap_values, max_display=max_display, show=False)
    plt.title(f"Top {max_display} Features por Importancia SHAP")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Gráfico guardado: {save_path}")
    plt.show()


def plot_dependence(
    shap_values: shap.Explanation,
    feature: str,
    interaction_feature: Optional[str] = None,
    save_path: Optional[str] = None,
) -> None:
    """
    Dependence Plot: relación entre una feature y su SHAP value.
    Opcionalmente coloreado por otra variable (interacción).
    """
    plt.figure(figsize=(10, 6))
    shap.plots.scatter(
        shap_values[:, feature],
        color=shap_values[:, interaction_feature] if interaction_feature else None,
        show=False,
    )
    plt.title(f"Dependencia SHAP: {feature}")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Gráfico guardado: {save_path}")
    plt.show()


def plot_waterfall_individual(
    shap_values: shap.Explanation,
    indice: int = 0,
    max_display: int = 15,
    save_path: Optional[str] = None,
) -> None:
    """
    Waterfall Plot: explicación de una predicción individual.
    Muestra cómo cada variable sube o baja el precio desde la predicción base.
    Ideal para incluir en la tesis como ejemplo de explicabilidad.
    """
    plt.figure(figsize=(10, 8))
    shap.plots.waterfall(shap_values[indice], max_display=max_display, show=False)
    plt.title(f"Explicación Individual — Inmueble #{indice}")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Gráfico guardado: {save_path}")
    plt.show()


def plot_force_individual(
    explainer: shap.TreeExplainer,
    shap_values_array: np.ndarray,
    X: pd.DataFrame,
    indice: int = 0,
    save_path: Optional[str] = None,
) -> None:
    """
    Force Plot: visualización horizontal de contribución por variable.
    """
    shap.initjs()
    force_plot = shap.force_plot(
        explainer.expected_value,
        shap_values_array[indice, :],
        X.iloc[indice, :],
    )

    if save_path:
        shap.save_html(save_path.replace(".png", ".html"), force_plot)
        print(f"Force plot guardado: {save_path.replace('.png', '.html')}")

    return force_plot


def generar_reporte_shap_completo(
    modelo: XGBRegressor,
    X: pd.DataFrame,
    reports_dir: str,
    max_display: int = 15,
) -> shap.Explanation:
    """
    Genera todos los gráficos SHAP de una sola vez y los guarda en reports/.

    Returns:
        shap.Explanation para uso posterior.
    """
    os.makedirs(reports_dir, exist_ok=True)

    print("Generando reporte SHAP completo...")
    print("=" * 50)

    # 1. Crear explainer y calcular valores
    explainer = crear_explainer(modelo)
    shap_values = calcular_shap_values(explainer, X)

    # 2. Summary beeswarm
    plot_summary_beeswarm(
        shap_values,
        max_display=max_display,
        save_path=os.path.join(reports_dir, "shap_summary_beeswarm.png"),
    )

    # 3. Bar importance
    plot_bar_importance(
        shap_values,
        max_display=max_display,
        save_path=os.path.join(reports_dir, "shap_bar_importance.png"),
    )

    # 4. Waterfall del primer caso
    plot_waterfall_individual(
        shap_values,
        indice=0,
        max_display=max_display,
        save_path=os.path.join(reports_dir, "shap_waterfall_ejemplo.png"),
    )

    # 5. Dependence: Superficie
    if "Superficie" in X.columns:
        plot_dependence(
            shap_values,
            feature="Superficie",
            save_path=os.path.join(reports_dir, "shap_dependence_superficie.png"),
        )

    print("=" * 50)
    print("Reporte SHAP completo generado.")

    return shap_values
