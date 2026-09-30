"""
Módulo de Transformación y Limpieza de Datos (Transform).
Aplica las reglas de negocio, imputaciones documentadas y normalizaciones para PostgreSQL.
"""
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Diccionario oficial de UBIGEOS INEI para los 22 distritos modelados
UBIGEO_DISTRITOS = {
    "Ate Vitarte": "150103",
    "Barranco": "150104",
    "Bellavista": "070102",
    "Breña": "150105",
    "Carabayllo": "150106",
    "Cercado de Lima": "150101",
    "Chorrillos": "150108",
    "Comas": "150110",
    "Jesús María": "150113",
    "La Molina": "150114",
    "La Perla": "070104",
    "La Victoria": "150115",
    "Lince": "150116",
    "Los Olivos": "150117",
    "Magdalena": "150120",
    "Miraflores": "150122",
    "Pueblo Libre": "150121",
    "San Borja": "150130",
    "San Isidro": "150131",
    "San Miguel": "150136",
    "Surco": "150140",
    "Surquillo": "150141",
}

def transform_catalogo_distritos(df_contexto: pd.DataFrame) -> pd.DataFrame:
    """Genera el catálogo de los 22 distritos con UBIGEO y área territorial fija."""
    distritos_unicos = df_contexto[["Distrito", "area_distrito_km2"]].drop_duplicates().copy()
    distritos_unicos = distritos_unicos.rename(columns={
        "Distrito": "nombre",
        "area_distrito_km2": "area_km2"
    })
    distritos_unicos["ubigeo"] = distritos_unicos["nombre"].map(UBIGEO_DISTRITOS).fillna("150100")
    distritos_unicos = distritos_unicos[["nombre", "ubigeo", "area_km2"]].sort_values("nombre").reset_index(drop=True)
    logger.info(f"Catálogo de distritos transformado: {len(distritos_unicos)} distritos.")
    return distritos_unicos

def transform_contexto_distrital(df_contexto: pd.DataFrame, distrito_id_map: dict) -> pd.DataFrame:
    """Prepara la tabla distrito_anio_contexto con referencias a distrito_id."""
    df = df_contexto.copy()
    df["distrito_id"] = df["Distrito"].map(distrito_id_map)
    
    # Filtrar únicamente los distritos mapeados válidos
    df = df[df["distrito_id"].notna()].copy()
    df["distrito_id"] = df["distrito_id"].astype(int)
    df["anio"] = df["Anio"].astype(int)
    
    # Renombrar columnas a snake_case
    columnas_db = {
        "pct_NSE_A": "pct_nse_a",
        "pct_NSE_B": "pct_nse_b",
        "pct_NSE_C": "pct_nse_c",
        "pct_NSE_D": "pct_nse_d",
        "pct_NSE_E": "pct_nse_e",
        "tasa_robo": "tasa_robo",
        "tasa_hurto": "tasa_hurto",
        "tasa_denuncias": "tasa_denuncias",
        "poblacion_proyectada": "poblacion_proyectada",
        "area_distrito_km2": "area_distrito_km2",
        "densidad_hab_km2": "densidad_hab_km2",
        "distancia_centro_km": "distancia_centro_km",
        "dist_colegio_km": "dist_colegio_km",
        "dist_hospital_km": "dist_hospital_km",
        "dist_estacion_transporte_km": "dist_estacion_transporte_km",
        "dist_centro_comercial_km": "dist_centro_comercial_km",
        "dist_parque_km": "dist_parque_km",
        "dist_universidad_km": "dist_universidad_km",
    }
    
    cols_existentes = {k: v for k, v in columnas_db.items() if k in df.columns}
    df = df.rename(columns=cols_existentes)
    
    cols_finales = ["distrito_id", "anio"] + list(cols_existentes.values())
    df_result = df[cols_finales].drop_duplicates(subset=["distrito_id", "anio"]).copy()
    logger.info(f"Contexto distrital transformado: {len(df_result)} filas para PostgreSQL.")
    return df_result

def transform_venta(df_raw: pd.DataFrame, distrito_id_map: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Transforma el Excel crudo de venta BCRP a la estructura de dataset_inmuebles_venta.
    Retorna: (df_venta_limpio, df_auditoria_flags)
    """
    df = df_raw.copy()
    
    # Renombrar columnas posicionales BCRP
    df = df.rename(columns={
        df.columns[1]: "anio",
        df.columns[2]: "trimestre",
        df.columns[3]: "precio_dolares",
        df.columns[4]: "tipo_cambio",
        df.columns[5]: "ipc",
        df.columns[6]: "precio_soles_nominal",
        df.columns[7]: "precio_soles_const",
        df.columns[8]: "distrito",
        df.columns[9]: "superficie_m2",
        df.columns[10]: "habitaciones",
        df.columns[11]: "banos",
        df.columns[12]: "garajes",
        df.columns[13]: "piso",
        df.columns[14]: "vista_exterior",
        df.columns[15]: "antiguedad_anios",
    })
    
    # Descartar nulos en Antigüedad (1,722 registros) para paridad histórica
    filas_inicio = len(df)
    df = df[df["antiguedad_anios"].notna()].copy()
    
    # Filtrar únicamente distritos soportados (excluye distritos n <= 1)
    df = df[df["distrito"].isin(distrito_id_map.keys())].copy()
    df["distrito_id"] = df["distrito"].map(distrito_id_map).astype(int)
    
    # Limpieza numérica de atributos físicos
    cols_num = ["superficie_m2", "habitaciones", "banos", "garajes", "piso", "antiguedad_anios", "precio_soles_nominal", "ipc", "precio_soles_const"]
    for c in cols_num:
        if df[c].dtype == "object":
            df[c] = df[c].replace(["N/D", "ND", "n/d", "nd", " ", "", "None"], np.nan)
        df[c] = pd.to_numeric(df[c], errors="coerce")
        
    # Vista Exterior: moda distrital (1.0 = Exterior)
    df["vista_exterior"] = pd.to_numeric(df["vista_exterior"], errors="coerce").fillna(1.0).astype(int)
    df["vista_exterior"] = df["vista_exterior"].apply(lambda v: True if v >= 1 else False)
    
    # Imputar medianas condicionales en el resto de atributos físicos si existiesen vacíos
    for col in ["habitaciones", "banos", "garajes", "piso"]:
        if df[col].isna().any():
            medianas = df.groupby("distrito")[col].transform("median")
            df[col] = df[col].fillna(medianas).fillna(1)
            
    # Garantizar target consistente
    mask_calc_target = df["precio_soles_const"].isna() | (df["precio_soles_const"] <= 0)
    df.loc[mask_calc_target, "precio_soles_const"] = (df.loc[mask_calc_target, "precio_soles_nominal"] * 100.0) / df.loc[mask_calc_target, "ipc"]
    
    # Split temporal estándar
    df["split_dataset"] = df["anio"].apply(lambda a: "TRAIN" if a <= 2023 else "TEST")
    # Resetear índice para alineación consistente
    df = df.reset_index(drop=True)
    
    # Selección de columnas finales para la tabla dataset_inmuebles_venta
    cols_venta = [
        "distrito_id", "anio", "trimestre", "superficie_m2",
        "habitaciones", "banos", "garajes", "piso", "antiguedad_anios",
        "vista_exterior", "precio_soles_nominal", "ipc", "precio_soles_const", "split_dataset"
    ]
    df_venta = df[cols_venta].rename(columns={"ipc": "ipc_deflactador"}).copy()
    
    # Flags de auditoría para análisis de sensibilidad
    df_auditoria = pd.DataFrame({
        "tipo_operacion": "VENTA",
        "inmueble_id": df_venta.index + 1,
        "distrito_id": df_venta["distrito_id"].values,
        "anio": df_venta["anio"].values,
        "nse_imputado": df["distrito"].isin(["Lince", "Barranco", "Magdalena", "Magdalena del Mar"]).values,
        "tasas_criminalidad_imputada": df_venta["anio"].isin([2016, 2017, 2018]).values,
        "poblacion_imputada": df_venta["anio"].isin([2016, 2017]).values,
        "split_dataset": df_venta["split_dataset"].values
    })
    
    logger.info(f"Venta transformada: {len(df_venta)} registros listos para BD (descartados: {filas_inicio - len(df_venta)}).")
    return df_venta, df_auditoria

def transform_alquiler(df_raw: pd.DataFrame, distrito_id_map: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Transforma el Excel crudo de alquiler BCRP a la estructura de dataset_inmuebles_alquiler.
    Retorna: (df_alquiler_limpio, df_auditoria_flags)
    """
    df = df_raw.copy()
    
    # Renombrar columnas posicionales BCRP
    df = df.rename(columns={
        df.columns[1]: "anio",
        df.columns[2]: "trimestre",
        df.columns[3]: "alquiler_dolares",
        df.columns[4]: "tipo_cambio",
        df.columns[5]: "ipc",
        df.columns[6]: "alquiler_soles_nominal",
        df.columns[7]: "alquiler_soles_const",
        df.columns[8]: "distrito",
        df.columns[9]: "superficie_m2",
        df.columns[10]: "habitaciones",
        df.columns[11]: "banos",
        df.columns[12]: "garajes",
        df.columns[13]: "piso",
        df.columns[14]: "vista_exterior",
        df.columns[15]: "antiguedad_anios",
    })
    
    filas_inicio = len(df)
    # Filtrar únicamente distritos soportados (excluye distritos n <= 1)
    df = df[df["distrito"].isin(distrito_id_map.keys())].copy()
    df["distrito_id"] = df["distrito"].map(distrito_id_map).astype(int)
    
    # Limpieza numérica de atributos físicos
    cols_num = ["superficie_m2", "habitaciones", "banos", "garajes", "piso", "antiguedad_anios", "alquiler_soles_nominal", "ipc", "alquiler_soles_const"]
    for c in cols_num:
        if df[c].dtype == "object":
            df[c] = df[c].replace(["N/D", "ND", "n/d", "nd", " ", "", "None"], np.nan)
        df[c] = pd.to_numeric(df[c], errors="coerce")
        
    # Imputación de atributos físicos
    # Vista Exterior: moda distrital (1.0 = Exterior)
    df["vista_exterior"] = pd.to_numeric(df["vista_exterior"], errors="coerce").fillna(1.0).astype(int)
    df["vista_exterior"] = df["vista_exterior"].apply(lambda v: True if v >= 1 else False)
    
    # Imputar medianas distritales para nulos en alquiler
    for col in ["piso", "antiguedad_anios", "habitaciones", "banos", "garajes"]:
        if df[col].isna().any():
            medianas = df.groupby("distrito")[col].transform("median")
            df[col] = df[col].fillna(medianas).fillna(0)
            
    # Garantizar target consistente
    mask_calc_target = df["alquiler_soles_const"].isna() | (df["alquiler_soles_const"] <= 0)
    df.loc[mask_calc_target, "alquiler_soles_const"] = (df.loc[mask_calc_target, "alquiler_soles_nominal"] * 100.0) / df.loc[mask_calc_target, "ipc"]
    
    # Split temporal estándar
    df["split_dataset"] = df["anio"].apply(lambda a: "TRAIN" if a <= 2023 else "TEST")
    # Resetear índice para alineación consistente
    df = df.reset_index(drop=True)
    
    # Selección de columnas finales para dataset_inmuebles_alquiler
    cols_alquiler = [
        "distrito_id", "anio", "trimestre", "superficie_m2",
        "habitaciones", "banos", "garajes", "piso", "antiguedad_anios",
        "vista_exterior", "alquiler_soles_nominal", "ipc", "alquiler_soles_const", "split_dataset"
    ]
    df_alquiler = df[cols_alquiler].rename(columns={"ipc": "ipc_deflactador"}).copy()
    
    # Flags de auditoría para análisis de sensibilidad
    df_auditoria = pd.DataFrame({
        "tipo_operacion": "ALQUILER",
        "inmueble_id": df_alquiler.index + 1,
        "distrito_id": df_alquiler["distrito_id"].values,
        "anio": df_alquiler["anio"].values,
        "nse_imputado": df["distrito"].isin(["Lince", "Barranco", "Magdalena", "Magdalena del Mar"]).values,
        "tasas_criminalidad_imputada": df_alquiler["anio"].isin([2016, 2017, 2018]).values,
        "poblacion_imputada": df_alquiler["anio"].isin([2016, 2017]).values,
        "split_dataset": df_alquiler["split_dataset"].values
    })
    
    logger.info(f"Alquiler transformado: {len(df_alquiler)} registros listos para BD (descartados: {filas_inicio - len(df_alquiler)}).")
    return df_alquiler, df_auditoria
