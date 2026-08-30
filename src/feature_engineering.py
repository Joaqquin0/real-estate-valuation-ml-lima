"""
Funciones de feature engineering.
Transformaciones de variables para mejorar el rendimiento del modelo.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Tuple
from sklearn.preprocessing import LabelEncoder


def crear_periodo_numerico(
    df: pd.DataFrame,
    col_anio: str = "Anio",
    col_trimestre: str = "Trimestre",
    nueva_col: str = "periodo_numerico",
) -> pd.DataFrame:
    """
    Crea una variable de tendencia temporal continua.
    periodo_numerico = Anio * 4 + Trimestre
    Captura la tendencia temporal de forma lineal.
    """
    df = df.copy()
    df[nueva_col] = df[col_anio] * 4 + df[col_trimestre]
    print(f"Feature creada: '{nueva_col}' "
          f"[rango: {df[nueva_col].min()} – {df[nueva_col].max()}]")
    return df


def target_encoding_distrito(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    col_distrito: str = "Distrito",
    col_target: str = "Precio_Soles_Const",
    nueva_col: str = "distrito_encoded",
    smoothing: float = 10.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aplica Target Encoding suavizado al distrito.

    Usa la media del target por distrito con regularización bayesiana
    para evitar overfitting en distritos con pocas observaciones.

    Se calcula SOLO con datos de train para evitar data leakage.

    Args:
        smoothing: Factor de suavizado (más alto = más regularización).

    Returns:
        Tupla (df_train, df_test) con la nueva columna.
    """
    df_train = df_train.copy()
    df_test = df_test.copy()

    # Media global del target en train
    media_global = df_train[col_target].mean()

    # Media y conteo por distrito en train
    stats = df_train.groupby(col_distrito)[col_target].agg(["mean", "count"])

    # Target encoding suavizado: (count * mean_distrito + smoothing * mean_global) / (count + smoothing)
    stats[nueva_col] = (
        (stats["count"] * stats["mean"] + smoothing * media_global)
        / (stats["count"] + smoothing)
    )

    encoding_map = stats[nueva_col].to_dict()

    # Aplicar a train y test
    df_train[nueva_col] = df_train[col_distrito].map(encoding_map)
    df_test[nueva_col] = df_test[col_distrito].map(encoding_map)

    # Distritos no vistos en train → media global
    n_unseen = df_test[nueva_col].isna().sum()
    if n_unseen > 0:
        print(f"⚠️ {n_unseen} registros en test con distrito no visto en train → media global")
        df_test[nueva_col] = df_test[nueva_col].fillna(media_global)

    print(f"Target Encoding aplicado: '{col_distrito}' → '{nueva_col}' "
          f"({len(encoding_map)} distritos)")

    return df_train, df_test, encoding_map


def evaluar_multicolinealidad(
    df: pd.DataFrame,
    cols: Optional[List[str]] = None,
    umbral_corr: float = 0.90,
) -> pd.DataFrame:
    """
    Identifica pares de features con correlación alta (posible multicolinealidad).

    Returns:
        DataFrame con pares correlacionados por encima del umbral.
    """
    if cols is not None:
        df_subset = df[cols]
    else:
        df_subset = df.select_dtypes(include=[np.number])

    corr_matrix = df_subset.corr().abs()

    # Tomar solo el triángulo superior
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    # Encontrar pares con alta correlación
    pares = []
    for col in upper.columns:
        for idx in upper.index:
            val = upper.loc[idx, col]
            if val > umbral_corr:
                pares.append({
                    "variable_1": idx,
                    "variable_2": col,
                    "correlacion": val,
                })

    resultado = pd.DataFrame(pares).sort_values("correlacion", ascending=False)
    print(f"Pares con correlación > {umbral_corr}: {len(resultado)}")
    return resultado


def log_transform_target(
    y: pd.Series,
    inverse: bool = False,
) -> pd.Series:
    """
    Aplica transformación logarítmica al target para normalizar distribución.
    Útil si el target tiene distribución sesgada a la derecha (común en precios).

    Args:
        inverse: Si True, aplica la transformación inversa (exp).
    """
    if inverse:
        return np.expm1(y)
    else:
        return np.log1p(y)
