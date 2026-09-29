import pandas as pd
import numpy as np 

def main():
    print("=== Pipeline de Extracción NSE - ENAHO 2025 ===\n")

    # 1. Carga de datos
    df = pd.read_csv("Sumaria-2025.csv", sep=";", encoding="latin-1", low_memory=False, decimal=",")    
    df['UBIGEO'] = df['UBIGEO'].astype(str).str.zfill(6)

    if 'factor07' in df.columns:
        df.rename(columns={'factor07': 'FACTOR07'}, inplace=True)

    # Filtrar Lima Metropolitana y Callao
    df_filtrado = df[(df['UBIGEO'].str.startswith('1501')) | (df['UBIGEO'].str.startswith('0701'))].copy()
    #print(f"Conteo de hogares antes de aplicar el filtro: {df_filtrado['HOGAR'].value_counts()}")  # antes de aplicar el filtro
    df_filtrado = df_filtrado[df_filtrado['HOGAR'] == 11].copy()

    # Total ponderado de Lima Metropolitana + Callao (todo junto, sin agrupar por distrito)
    total_ponderado = df_filtrado.groupby('ESTRSOCIAL')['FACTOR07'].sum()
    pct_total = (total_ponderado / total_ponderado.sum()) * 100
    print(pct_total)

    # =======================================================
    # CÁLCULO DE PORCENTAJES PONDERADOS POR ESTRSOCIAL
    # =======================================================
    agrupado = df_filtrado.groupby(['UBIGEO', 'ESTRSOCIAL'])['FACTOR07'].sum().reset_index()

    totales_distrito = agrupado.groupby('UBIGEO')['FACTOR07'].sum().reset_index()
    totales_distrito.rename(columns={'FACTOR07': 'TOTAL_FAMILIAS'}, inplace=True)

    nse_distrito = pd.merge(agrupado, totales_distrito, on='UBIGEO')
    nse_distrito['pct_NSE'] = (nse_distrito['FACTOR07'] / nse_distrito['TOTAL_FAMILIAS']) * 100

    df_final = nse_distrito.pivot(index='UBIGEO', columns='ESTRSOCIAL', values='pct_NSE').fillna(0).reset_index()

    mapeo_columnas = {
        1: 'pct_NSE_A', 2: 'pct_NSE_B', 3: 'pct_NSE_C', 
        4: 'pct_NSE_D', 5: 'pct_NSE_E'
    }
    df_final.rename(columns=mapeo_columnas, inplace=True)
    if 6 in df_final.columns:
        df_final.drop(columns=[6], inplace=True)

    # =======================================================
    # CONTEO DE OBSERVACIONES Y APLICACIÓN DEL UMBRAL (30 hogares)
    # =======================================================
    conteo_hogares = df_filtrado.groupby('UBIGEO').size().reset_index(name='n_hogares_muestra')
    df_final = df_final.merge(conteo_hogares, on='UBIGEO', how='left')

    UMBRAL_MINIMO = 30
    columnas_nse = ['pct_NSE_A', 'pct_NSE_B', 'pct_NSE_C', 'pct_NSE_D', 'pct_NSE_E']

    # Los distritos con muestra insuficiente quedan en NaN, sin inventar un "vecino" arbitrario
    distritos_insuficientes = df_final['n_hogares_muestra'] < UMBRAL_MINIMO
    df_final.loc[distritos_insuficientes, columnas_nse] = np.nan

    print(f"Distritos con muestra insuficiente (<{UMBRAL_MINIMO} hogares): {distritos_insuficientes.sum()}")

    # =======================================================
    # AGREGAR DISTRITOS SIN NINGUNA OBSERVACIÓN EN LA MUESTRA
    # =======================================================
    distritos_fantasma = pd.DataFrame({
        'UBIGEO': ['150127', '070105', '150124'],  # Punta Negra, La Punta, Pucusana
        'pct_NSE_A': [np.nan, np.nan, np.nan],
        'pct_NSE_B': [np.nan, np.nan, np.nan],
        'pct_NSE_C': [np.nan, np.nan, np.nan],
        'pct_NSE_D': [np.nan, np.nan, np.nan],
        'pct_NSE_E': [np.nan, np.nan, np.nan],
        'n_hogares_muestra': [0, 0, 0]
    })

    df_final = pd.concat([df_final, distritos_fantasma], ignore_index=True)
    df_final = df_final.sort_values(by='UBIGEO').reset_index(drop=True)

    print("\n=== DATASET FINAL GENERADO PARA EL MODELO ===")
    pd.set_option('display.max_columns', None)
    print(df_final.tail(15))

    #df_final.to_csv("Dataset_NSE_Distritos.csv", sep=";", index=False, decimal=",")

    df_final['suma_check'] = df_final[columnas_nse].sum(axis=1, skipna=False)
    print(df_final[['UBIGEO', 'n_hogares_muestra', 'suma_check']].sort_values('suma_check'))
    print("Distritos con NaN:", df_final['pct_NSE_A'].isnull().sum())
    print("Distritos con datos válidos:", df_final['pct_NSE_A'].notnull().sum())

if __name__ == "__main__":
    main()