# =============================================================================
# 01_EDA_ALQUILER.py
# =============================================================================
# Análisis Exploratorio de Datos (EDA) — Modelo Predictivo de Alquiler
# Valoración Inmobiliaria en Lima Metropolitana
#
# CRISP-DM: Fase 1 — Comprensión de los Datos
#
# Instrucciones:
#   Ejecutar localmente o en Google Colab.
# =============================================================================

# %% [markdown]
# # 📊 EDA — Análisis Exploratorio del Dataset de Alquiler (BCRP)
# ## Cobertura: 22 distritos representativos | Periodo: 2016–2025
# **Target**: `Alquiler mensual en soles constantes de 2009`

# %% Importaciones
import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")
plt.rcParams["figure.dpi"] = 120
plt.rcParams["font.size"] = 11

# %% 1. Configuración de Rutas
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, "dataset_entrenamineto_alquiler_2025.xlsx")
REPORTS_DIR = os.path.join(BASE_DIR, "reports", "figures", "alquiler")
os.makedirs(REPORTS_DIR, exist_ok=True)

# %% 2. Carga del Dataset
df = pd.read_excel(DATASET_PATH)
print("=" * 60)
print(f"Dataset Alquiler Crudo: {df.shape[0]:,} filas x {df.shape[1]} columnas")
print("=" * 60)
print("\nColumnas disponibles:")
for i, col in enumerate(df.columns, 1):
    print(f"  {i:2d}. {col}")

# %% 3. Normalización preliminar de nombres para análisis
df = df.rename(columns={
    df.columns[1]: "Anio",
    df.columns[2]: "Trimestre",
    df.columns[3]: "Alquiler_Dolares",
    df.columns[4]: "Tipo_Cambio",
    df.columns[5]: "IPC",
    df.columns[6]: "Alquiler_Soles",
    df.columns[7]: "Alquiler_Soles_Const",
    df.columns[8]: "Distrito",
    df.columns[9]: "Superficie",
    df.columns[10]: "Habitaciones",
    df.columns[11]: "Banios",
    df.columns[12]: "Garajes",
    df.columns[13]: "Piso",
    df.columns[14]: "Vista_Exterior",
    df.columns[15]: "Antiguedad",
})

# %% 4. Distribución por Distrito y Filtrado de Atípicos (1 dato)
print("\n" + "=" * 60)
print("DISTRIBUCIÓN POR DISTRITO")
print("=" * 60)
conteos_distrito = df["Distrito"].value_counts()
print(conteos_distrito)

distritos_excluir = conteos_distrito[conteos_distrito <= 1].index.tolist()
print(f"\nDistritos atípicos con <= 1 registro (a excluir): {distritos_excluir}")

df_filtrado = df[~df["Distrito"].isin(distritos_excluir)].copy()
print(f"Dataset tras filtrar atípicos: {df_filtrado.shape[0]:,} filas en {df_filtrado['Distrito'].nunique()} distritos")

# %% 5. Diagnóstico de Nulos y Valores Especiales
print("\n" + "=" * 60)
print("DIAGNÓSTICO DE VALORES NULOS Y COMPLEJOS")
print("=" * 60)
nulos = df_filtrado.isnull().sum()
pct_nulos = (nulos / len(df_filtrado) * 100).round(2)
rep_nulos = pd.DataFrame({"Nulos": nulos, "Porcentaje": pct_nulos})
print(rep_nulos[rep_nulos["Nulos"] > 0])

# Detalle de columnas críticas: Piso, Vista, Antiguedad
print("\nDistribución de 'Vista_Exterior':")
print(df_filtrado["Vista_Exterior"].value_counts(dropna=False))

print("\nResumen de 'Piso' (top 10 valores más frecuentes):")
print(df_filtrado["Piso"].value_counts(dropna=False).head(10))

print("\nResumen de 'Antiguedad' (top 10 valores más frecuentes):")
print(df_filtrado["Antiguedad"].value_counts(dropna=False).head(10))

# %% 6. Estadísticas Descriptivas del Target y Variables Físicas
print("\n" + "=" * 60)
print("ESTADÍSTICAS DESCRIPTIVAS")
print("=" * 60)
cols_num = ["Alquiler_Soles_Const", "Superficie", "Habitaciones", "Banios", "Garajes", "Antiguedad", "Piso"]
print(df_filtrado[cols_num].describe().T.round(2))

# %% 7. Distribución Temporal (Años y Trimestres)
print("\n" + "=" * 60)
print("DISTRIBUCIÓN TEMPORAL")
print("=" * 60)
print(pd.crosstab(df_filtrado["Anio"], df_filtrado["Trimestre"], margins=True))

# %% 8. Guardar Gráfico de Target (Histograma y Skewness)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.histplot(df_filtrado["Alquiler_Soles_Const"], kde=True, bins=50, ax=axes[0], color="royalblue")
axes[0].set_title(f"Target Original (Skewness: {df_filtrado['Alquiler_Soles_Const'].skew():.2f})")
axes[0].set_xlabel("Alquiler mensual (S/. constantes 2009)")

sns.histplot(np.log1p(df_filtrado["Alquiler_Soles_Const"]), kde=True, bins=50, ax=axes[1], color="forestgreen")
axes[1].set_title(f"Target Log1p (Skewness: {np.log1p(df_filtrado['Alquiler_Soles_Const']).skew():.2f})")
axes[1].set_xlabel("log1p(Alquiler_Soles_Const)")

plt.tight_layout()
target_plot_path = os.path.join(REPORTS_DIR, "distribucion_target_alquiler.png")
plt.savefig(target_plot_path)
plt.close()
print(f"\n[OK] Gráfico de distribución guardado en: {target_plot_path}")

print("\n" + "=" * 60)
print("[OK] EDA DE ALQUILER FINALIZADO CON ÉXITO")
print("=" * 60)
