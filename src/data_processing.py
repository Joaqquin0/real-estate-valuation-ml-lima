"""
Funciones de limpieza y preprocesamiento de datos.
Alineado con TH-01: Preprocesar datos para modelo de venta.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Optional


def cargar_dataset(filepath: str, sheet_name: int = 0) -> pd.DataFrame:
    """Carga el dataset desde un archivo Excel."""
    df = pd.read_excel(filepath, sheet_name=sheet_name)
    print(f"Dataset cargado: {df.shape[0]} filas × {df.shape[1]} columnas")
    return df


def reporte_calidad(df: pd.DataFrame) -> pd.DataFrame:
    """
    Genera un reporte de calidad de datos por columna.

    Returns:
        DataFrame con: tipo, nulos, % nulos, únicos, ejemplo.
    """
    reporte = pd.DataFrame({
        "tipo": df.dtypes,
        "nulos": df.isnull().sum(),
        "pct_nulos": (df.isnull().sum() / len(df) * 100).round(2),
        "unicos": df.nunique(),
        "ejemplo": df.iloc[0],
    })
    return reporte.sort_values("pct_nulos", ascending=False)


def eliminar_columnas(
    df: pd.DataFrame,
    cols_drop: List[str],
    cols_flags: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Elimina columnas no predictivas y separa flags de imputación.

    Args:
        df: DataFrame original.
        cols_drop: Columnas a eliminar definitivamente.
        cols_flags: Columnas de flags de imputación (se separan para análisis de sensibilidad).

    Returns:
        Tupla (df_limpio, df_flags) donde df_flags tiene ID + flags.
    """
    # Guardar flags para análisis posterior
    cols_flags_existentes = [c for c in cols_flags if c in df.columns]
    df_flags = df[cols_flags_existentes].copy()

    # Eliminar columnas
    cols_a_eliminar = [c for c in cols_drop + cols_flags if c in df.columns]
    df_limpio = df.drop(columns=cols_a_eliminar)
    print(f"Columnas eliminadas: {len(cols_a_eliminar)} → quedan {df_limpio.shape[1]}")
    return df_limpio, df_flags


def detectar_outliers_iqr(
    df: pd.DataFrame,
    columna: str,
    factor: float = 1.5,
) -> pd.Series:
    """
    Detecta outliers usando el método IQR.

    Returns:
        Serie booleana: True = outlier.
    """
    Q1 = df[columna].quantile(0.25)
    Q3 = df[columna].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - factor * IQR
    upper = Q3 + factor * IQR
    mask = (df[columna] < lower) | (df[columna] > upper)
    n_outliers = mask.sum()
    print(f"Outliers en '{columna}': {n_outliers} ({n_outliers/len(df)*100:.1f}%) "
          f"[rango válido: {lower:,.0f} – {upper:,.0f}]")
    return mask


def filtrar_outliers(
    df: pd.DataFrame,
    columna: str,
    metodo: str = "iqr",
    factor: float = 1.5,
    percentil_inf: float = 0.01,
    percentil_sup: float = 0.99,
) -> pd.DataFrame:
    """
    Filtra outliers de una columna usando IQR o percentiles.

    Args:
        metodo: 'iqr' o 'percentil'.
    """
    n_antes = len(df)

    if metodo == "iqr":
        mask_outlier = detectar_outliers_iqr(df, columna, factor)
        df_filtrado = df[~mask_outlier].copy()
    elif metodo == "percentil":
        lower = df[columna].quantile(percentil_inf)
        upper = df[columna].quantile(percentil_sup)
        df_filtrado = df[(df[columna] >= lower) & (df[columna] <= upper)].copy()
    else:
        raise ValueError(f"Método '{metodo}' no válido. Usar 'iqr' o 'percentil'.")

    n_despues = len(df_filtrado)
    print(f"Outliers removidos: {n_antes - n_despues} filas "
          f"({(n_antes - n_despues)/n_antes*100:.1f}%)")
    return df_filtrado


def split_temporal(
    df: pd.DataFrame,
    col_anio: str,
    train_years: List[int],
    test_years: List[int],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Realiza un split temporal cronológico (sin data leakage).

    Returns:
        Tupla (df_train, df_test).
    """
    df_train = df[df[col_anio].isin(train_years)].copy()
    df_test = df[df[col_anio].isin(test_years)].copy()

    total = len(df_train) + len(df_test)
    print(f"Split temporal:")
    print(f"  Train ({min(train_years)}-{max(train_years)}): "
          f"{len(df_train)} filas ({len(df_train)/total*100:.1f}%)")
    print(f"  Test  ({min(test_years)}-{max(test_years)}): "
          f"{len(df_test)} filas ({len(df_test)/total*100:.1f}%)")

    # Verificar que no hay solapamiento
    assert len(set(train_years) & set(test_years)) == 0, \
        "¡Hay años solapados entre train y test!"

    return df_train, df_test


def separar_features_target(
    df: pd.DataFrame,
    target_col: str,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Separa features (X) del target (y)."""
    X = df.drop(columns=[target_col])
    y = df[target_col]
    print(f"Features: {X.shape[1]} columnas | Target: '{target_col}'")
    return X, y
