# =============================================================================
# 01_EDA_EXPLORATORIO.py
# =============================================================================
# Análisis Exploratorio de Datos (EDA) — Modelo Predictivo XGBoost
# Valoración Inmobiliaria en Lima Metropolitana
#
# CRISP-DM: Fase 1 — Comprensión de los Datos
#
# Instrucciones para Google Colab:
#   1. Subir el archivo dataset_entrenamiento_final_imputado.xlsx a Colab
#      (o montar Google Drive y ajustar la ruta)
#   2. Ejecutar celda por celda
# =============================================================================

# %% [markdown]
# # 📊 EDA — Análisis Exploratorio de Datos
# ## Modelo Predictivo XGBoost para Valoración Inmobiliaria
# **Dataset**: BCRP + INEI + OSM | **Cobertura**: 26 distritos | **Target**: `Precio_Soles_Const`

# %% Instalación de dependencias (solo en Colab)
# !pip install -q openpyxl seaborn

# %% Importaciones
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")
plt.rcParams["figure.dpi"] = 120
plt.rcParams["font.size"] = 11

# %% [markdown]
# ## 1. Carga del Dataset

# %% Carga de datos
# =============================================
# En Colab: cambiar la ruta según tu ubicación
# =============================================
DATASET_PATH = "dataset_entrenamiento_final_imputado.xlsx"
# Si usas Google Drive:
# from google.colab import drive
# drive.mount('/content/drive')
# DATASET_PATH = "/content/drive/MyDrive/python_modelo_tesis/data/raw/dataset_entrenamiento_final_imputado.xlsx"

df = pd.read_excel(DATASET_PATH)
print(f"Shape: {df.shape[0]:,} filas × {df.shape[1]} columnas")
print(f"\nColumnas:\n{list(df.columns)}")

# %% [markdown]
# ## 2. Tipos de Datos y Estructura

# %% Tipos de datos
print("=" * 60)
print("TIPOS DE DATOS")
print("=" * 60)
print(df.dtypes.to_string())
print(f"\nVariables numéricas: {df.select_dtypes(include=[np.number]).shape[1]}")
print(f"Variables categóricas/texto: {df.select_dtypes(include=['object']).shape[1]}")

# %% Primeras filas
df.head(10)

# %% [markdown]
# ## 3. Análisis de Nulos y Completitud

# %% Análisis de nulos
nulos = df.isnull().sum()
pct_nulos = (nulos / len(df) * 100).round(2)
reporte_nulos = pd.DataFrame({
    "nulos": nulos,
    "% nulos": pct_nulos,
}).sort_values("nulos", ascending=False)

print("=" * 60)
print("REPORTE DE NULOS")
print("=" * 60)
print(reporte_nulos[reporte_nulos["nulos"] > 0].to_string() if reporte_nulos["nulos"].sum() > 0 else "✅ No hay valores nulos en el dataset")

# %% Mapa de calor de nulos (solo si hay nulos)
if df.isnull().sum().sum() > 0:
    fig, ax = plt.subplots(figsize=(16, 6))
    sns.heatmap(df.isnull(), cbar=True, yticklabels=False, ax=ax, cmap="YlOrRd")
    ax.set_title("Mapa de Valores Nulos")
    plt.tight_layout()
    plt.show()
else:
    print("✅ Dataset completamente limpio — sin nulos")

# %% [markdown]
# ## 4. Estadísticas Descriptivas

# %% Estadísticas descriptivas completas
print("=" * 60)
print("ESTADÍSTICAS DESCRIPTIVAS")
print("=" * 60)
df.describe().T.round(2)

# %% [markdown]
# ## 5. Distribución del Target: `Precio_Soles_Const`

# %% Distribución del target
TARGET = "Precio_Soles_Const"

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Histograma
axes[0].hist(df[TARGET], bins=80, edgecolor="black", alpha=0.7, color="#4A90D9")
axes[0].set_xlabel("Precio (S/. constantes)")
axes[0].set_ylabel("Frecuencia")
axes[0].set_title(f"Distribución de {TARGET}")
axes[0].axvline(df[TARGET].median(), color="red", linestyle="--", label=f"Mediana: S/. {df[TARGET].median():,.0f}")
axes[0].axvline(df[TARGET].mean(), color="orange", linestyle="--", label=f"Media: S/. {df[TARGET].mean():,.0f}")
axes[0].legend(fontsize=9)

# Boxplot
axes[1].boxplot(df[TARGET], vert=True)
axes[1].set_ylabel("Precio (S/. constantes)")
axes[1].set_title("Boxplot del Precio")

# Log del precio (para ver si log-transform ayudaría)
axes[2].hist(np.log1p(df[TARGET]), bins=80, edgecolor="black", alpha=0.7, color="#7CB342")
axes[2].set_xlabel("log(Precio + 1)")
axes[2].set_ylabel("Frecuencia")
axes[2].set_title("Distribución Log-Transformada")

plt.suptitle(f"Análisis del Target: {TARGET}", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.show()

# Estadísticas del target
print(f"\n📊 Estadísticas de {TARGET}:")
print(f"  Media:     S/. {df[TARGET].mean():>12,.2f}")
print(f"  Mediana:   S/. {df[TARGET].median():>12,.2f}")
print(f"  Std:       S/. {df[TARGET].std():>12,.2f}")
print(f"  Mínimo:    S/. {df[TARGET].min():>12,.2f}")
print(f"  Máximo:    S/. {df[TARGET].max():>12,.2f}")
print(f"  Skewness:  {df[TARGET].skew():>12.2f}")
print(f"  Kurtosis:  {df[TARGET].kurtosis():>12.2f}")

# %% [markdown]
# ## 6. Distribución Temporal

# %% Registros por año
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Por año
registros_anio = df["Anio"].value_counts().sort_index()
registros_anio.plot(kind="bar", ax=axes[0], color="#4A90D9", edgecolor="black")
axes[0].set_title("Registros por Año")
axes[0].set_xlabel("Año")
axes[0].set_ylabel("Número de registros")
axes[0].axvline(x=registros_anio.index.get_loc(2024) - 0.5 if 2024 in registros_anio.index else -1,
                color="red", linestyle="--", linewidth=2, label="Split: Train | Test")
axes[0].legend()
for i, v in enumerate(registros_anio.values):
    axes[0].text(i, v + 50, str(v), ha="center", fontsize=8)

# Por año-trimestre
df["anio_trim"] = df["Anio"].astype(str) + "-Q" + df["Trimestre"].astype(str)
registros_trim = df["anio_trim"].value_counts().sort_index()
registros_trim.plot(kind="line", ax=axes[1], marker="o", markersize=3, color="#4A90D9")
axes[1].set_title("Registros por Año-Trimestre")
axes[1].set_xlabel("Periodo")
axes[1].set_ylabel("Registros")
axes[1].tick_params(axis="x", rotation=90, labelsize=7)

plt.tight_layout()
plt.show()

# Estadísticas del split
n_train = df[df["Anio"] <= 2023].shape[0]
n_test = df[df["Anio"] >= 2024].shape[0]
print(f"\n📅 Split temporal:")
print(f"  Train (2016-2023): {n_train:,} filas ({n_train/(n_train+n_test)*100:.1f}%)")
print(f"  Test  (2024-2025): {n_test:,} filas ({n_test/(n_train+n_test)*100:.1f}%)")

# %% [markdown]
# ## 7. Distribución por Distrito

# %% Registros por distrito
fig, ax = plt.subplots(figsize=(14, 7))
registros_distrito = df["Distrito"].value_counts()
registros_distrito.plot(kind="barh", ax=ax, color="#4A90D9", edgecolor="black")
ax.set_xlabel("Número de registros")
ax.set_title("Registros por Distrito (desbalance)")

# Añadir conteo
for i, (v, d) in enumerate(zip(registros_distrito.values, registros_distrito.index)):
    ax.text(v + 20, i, f"{v:,}", va="center", fontsize=8)

plt.tight_layout()
plt.show()

print(f"\nDistrito con más datos: {registros_distrito.index[0]} ({registros_distrito.iloc[0]:,})")
print(f"Distrito con menos datos: {registros_distrito.index[-1]} ({registros_distrito.iloc[-1]:,})")
print(f"Ratio max/min: {registros_distrito.iloc[0]/registros_distrito.iloc[-1]:.1f}x")

# %% [markdown]
# ## 8. Precio Mediano por Distrito

# %% Precio por distrito
fig, ax = plt.subplots(figsize=(14, 7))
precio_distrito = df.groupby("Distrito")[TARGET].median().sort_values(ascending=True)
precio_distrito.plot(kind="barh", ax=ax, color="#7CB342", edgecolor="black")
ax.set_xlabel(f"Mediana de {TARGET} (S/.)")
ax.set_title("Precio Mediano por Distrito")

for i, (v, d) in enumerate(zip(precio_distrito.values, precio_distrito.index)):
    ax.text(v + 5000, i, f"S/. {v:,.0f}", va="center", fontsize=8)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 9. Correlaciones

# %% Heatmap de correlaciones
cols_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
# Excluir ID si existe
cols_numericas = [c for c in cols_numericas if c != "ID"]

corr_matrix = df[cols_numericas].corr()

# Correlaciones con el target
corr_target = corr_matrix[TARGET].drop(TARGET).sort_values(ascending=False)
print(f"\n🔗 Correlaciones con {TARGET}:")
print(corr_target.to_string())

# Heatmap completo
fig, ax = plt.subplots(figsize=(18, 14))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(
    corr_matrix, mask=mask, annot=True, fmt=".2f",
    cmap="RdBu_r", center=0, square=True,
    linewidths=0.5, ax=ax,
    annot_kws={"size": 7},
)
ax.set_title(f"Matriz de Correlaciones", fontsize=14)
plt.tight_layout()
plt.show()

# %% Correlaciones altas (multicolinealidad potencial)
print("\n⚠️ Pares con correlación |r| > 0.90 (posible multicolinealidad):")
for col in corr_matrix.columns:
    for idx in corr_matrix.index:
        if col != idx and abs(corr_matrix.loc[idx, col]) > 0.90:
            if corr_matrix.columns.get_loc(col) > corr_matrix.columns.get_loc(idx):
                print(f"  {idx} ↔ {col}: {corr_matrix.loc[idx, col]:.3f}")

# %% [markdown]
# ## 10. Detección de Outliers

# %% Outliers en variables clave
variables_outlier = ["Precio_Soles_Const", "Superficie", "Habitaciones", "Banios", "Garajes", "Piso", "Antiguedad"]
variables_outlier = [v for v in variables_outlier if v in df.columns]

fig, axes = plt.subplots(2, 4, figsize=(18, 8))
axes = axes.flatten()

for i, var in enumerate(variables_outlier):
    if i < len(axes):
        axes[i].boxplot(df[var].dropna())
        axes[i].set_title(var, fontsize=10)

# Ocultar ejes vacíos
for j in range(len(variables_outlier), len(axes)):
    axes[j].set_visible(False)

plt.suptitle("Boxplots — Detección de Outliers", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()

# %% Conteo de outliers por IQR
print("\n📋 Outliers detectados (método IQR × 1.5):")
for var in variables_outlier:
    Q1 = df[var].quantile(0.25)
    Q3 = df[var].quantile(0.75)
    IQR = Q3 - Q1
    n_outliers = ((df[var] < Q1 - 1.5 * IQR) | (df[var] > Q3 + 1.5 * IQR)).sum()
    print(f"  {var:30s}: {n_outliers:>5,} outliers ({n_outliers/len(df)*100:.1f}%)")

# %% [markdown]
# ## 11. Variables Imputadas

# %% Análisis de flags de imputación
flags = ["nse_imputado", "tasas_criminalidad_imputada", "poblacion_imputada"]
flags_existentes = [f for f in flags if f in df.columns]

if flags_existentes:
    print("\n🏷️ Variables con imputación:")
    for flag in flags_existentes:
        n_imputados = df[flag].sum() if df[flag].dtype in [int, float, bool] else (df[flag] == 1).sum()
        print(f"  {flag}: {n_imputados:,} filas imputadas ({n_imputados/len(df)*100:.1f}%)")

    # Comparar precios: datos imputados vs no imputados
    if "nse_imputado" in df.columns:
        fig, ax = plt.subplots(figsize=(10, 5))
        df.boxplot(column=TARGET, by="nse_imputado", ax=ax)
        ax.set_title(f"Precio según NSE imputado (0=original, 1=imputado)")
        ax.set_xlabel("NSE Imputado")
        ax.set_ylabel(f"{TARGET} (S/.)")
        plt.suptitle("")
        plt.tight_layout()
        plt.show()

# %% [markdown]
# ## 12. Relaciones Bivariadas Clave

# %% Superficie vs Precio
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Scatter: Superficie vs Precio
axes[0].scatter(df["Superficie"], df[TARGET], alpha=0.15, s=5, color="#4A90D9")
axes[0].set_xlabel("Superficie (m²)")
axes[0].set_ylabel(f"{TARGET} (S/.)")
axes[0].set_title("Superficie vs Precio")

# NSE_A vs Precio
if "pct_NSE_A" in df.columns:
    axes[1].scatter(df["pct_NSE_A"], df[TARGET], alpha=0.15, s=5, color="#E57373")
    axes[1].set_xlabel("% NSE A")
    axes[1].set_ylabel(f"{TARGET} (S/.)")
    axes[1].set_title("NSE A (%) vs Precio")

plt.tight_layout()
plt.show()

# %% Antigüedad vs Precio
if "Antiguedad" in df.columns:
    fig, ax = plt.subplots(figsize=(10, 5))
    # Agrupar por rangos de antigüedad
    df["rango_antiguedad"] = pd.cut(df["Antiguedad"], bins=[0, 2, 5, 10, 20, 50, 100], right=True)
    df.boxplot(column=TARGET, by="rango_antiguedad", ax=ax)
    ax.set_title("Precio por Rango de Antigüedad")
    ax.set_xlabel("Años de Antigüedad")
    ax.set_ylabel(f"{TARGET} (S/.)")
    plt.suptitle("")
    plt.tight_layout()
    plt.show()
    df.drop(columns=["rango_antiguedad"], inplace=True)

# %% Evolución temporal del precio por distrito (top 5)
top5_distritos = df["Distrito"].value_counts().head(5).index.tolist()
df_top5 = df[df["Distrito"].isin(top5_distritos)]

fig, ax = plt.subplots(figsize=(14, 6))
for distrito in top5_distritos:
    subset = df_top5[df_top5["Distrito"] == distrito]
    evol = subset.groupby("Anio")[TARGET].median()
    ax.plot(evol.index, evol.values, marker="o", label=distrito, linewidth=2)

ax.set_xlabel("Año")
ax.set_ylabel(f"Mediana {TARGET} (S/.)")
ax.set_title("Evolución del Precio Mediano — Top 5 Distritos")
ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
ax.axvline(x=2023.5, color="red", linestyle="--", alpha=0.7, label="Split Train|Test")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 13. Resumen y Hallazgos Clave

# %% Resumen automático
print("=" * 60)
print("📋 RESUMEN DEL EDA")
print("=" * 60)
print(f"\n1. DIMENSIONES: {df.shape[0]:,} filas × {df.shape[1]} columnas")
print(f"2. NULOS TOTALES: {df.isnull().sum().sum():,}")
print(f"3. TARGET ({TARGET}):")
print(f"   - Media: S/. {df[TARGET].mean():,.0f}")
print(f"   - Mediana: S/. {df[TARGET].median():,.0f}")
print(f"   - Skewness: {df[TARGET].skew():.2f} ({'sesgada positivamente' if df[TARGET].skew() > 1 else 'moderadamente sesgada' if df[TARGET].skew() > 0.5 else 'simétrica'})")
print(f"4. DISTRITOS: {df['Distrito'].nunique()}")
print(f"5. PERIODO: {df['Anio'].min()} – {df['Anio'].max()}")
print(f"6. SPLIT:")
print(f"   - Train (≤2023): {df[df['Anio'] <= 2023].shape[0]:,}")
print(f"   - Test  (≥2024): {df[df['Anio'] >= 2024].shape[0]:,}")

# Limpieza de variable temporal auxiliar
if "anio_trim" in df.columns:
    df.drop(columns=["anio_trim"], inplace=True)

print("\n✅ EDA completado. Siguiente paso: 02_preprocesamiento.py")
