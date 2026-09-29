import sys
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except AttributeError:
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace", line_buffering=True)

import pandas as pd
import unicodedata
import os

def normalizar_texto(texto):
    """Limpia tildes, espacios extra y pasa a mayusculas."""
    if pd.isna(texto):
        return ""
    texto = str(texto).strip().upper()
    texto = ''.join(c for c in unicodedata.normalize('NFD', texto) if unicodedata.category(c) != 'Mn')
    return texto

def parse_ubigeo(val):
    """Convierte el UBIGEO a un string estandar de 6 digitos."""
    if pd.isna(val):
        return None
    try:
        val_float = float(str(val).strip())
        return f"{int(val_float):06d}"
    except (ValueError, TypeError):
        return None

def main():
    # 1. Ruta del archivo Excel de poblacion proyectada
    ruta_excel = r"C:\Users\Joaquin\Downloads\poblacion_proyectada.xlsx"
    if not os.path.exists(ruta_excel):
        ruta_excel = "poblacion_proyectada.xlsx"
    
    print(f"Cargando archivo: {ruta_excel}...")
    df = pd.read_excel(ruta_excel)
    print(f"Filas totales en el archivo: {len(df):,}")

    # 2. Diccionario de distritos permitidos (Lima Metropolitana y Callao)
    mapeo_distrito_ubigeo = {
        # Lima Metropolitana (1501xx)
        "LIMA": "150101",
        "CERCADO DE LIMA": "150101",
        "ANCON": "150102",
        "ATE": "150103",
        "ATE VITARTE": "150103",
        "BARRANCO": "150104",
        "BRENA": "150105",
        "CARABAYLLO": "150106",
        "CHACLACAYO": "150107",
        "CHORRILLOS": "150108",
        "CIENEGUILLA": "150109",
        "COMAS": "150110",
        "EL AGUSTINO": "150111",
        "INDEPENDENCIA": "150112",
        "JESUS MARIA": "150113",
        "LA MOLINA": "150114",
        "LA VICTORIA": "150115",
        "LINCE": "150116",
        "LOS OLIVOS": "150117",
        "LURIGANCHO": "150118",
        "LURIGANCHO CHOSICA": "150118",
        "CHOSICA": "150118",
        "LURIN": "150119",
        "MAGDALENA DEL MAR": "150120",
        "MAGDALENA": "150120",
        "PUEBLO LIBRE": "150121",
        "MIRAFLORES": "150122",
        "PACHACAMAC": "150123",
        "PUCUSANA": "150124",
        "PUENTE PIEDRA": "150125",
        "PUNTA HERMOSA": "150126",
        "PUNTA NEGRA": "150127",
        "RIMAC": "150128",
        "SAN BARTOLO": "150129",
        "SAN BORJA": "150130",
        "SAN ISIDRO": "150131",
        "SAN JUAN DE LURIGANCHO": "150132",
        "SAN JUAN DE MIRAFLORES": "150133",
        "SAN LUIS": "150134",
        "SAN MARTIN DE PORRES": "150135",
        "SAN MIGUEL": "150136",
        "SANTA ANITA": "150137",
        "SANTA MARIA DEL MAR": "150138",
        "SANTA ROSA": "150139",
        "SANTIAGO DE SURCO": "150140",
        "SURCO": "150140",
        "SURQUILLO": "150141",
        "VILLA EL SALVADOR": "150142",
        "VILLA MARIA DEL TRIUNFO": "150143",

        # Callao (0701xx)
        "CALLAO": "070101",
        "BELLAVISTA": "070102",
        "CARMEN DE LA LEGUA": "070103",
        "CARMEN DE LA LEGUA REYNOSO": "070103",
        "LA PERLA": "070104",
        "LA PUNTA": "070105",
        "VENTANILLA": "070106",
        "MI PERU": "070107",
    }

    ubigeos_validos = set(mapeo_distrito_ubigeo.values())

    # 3. Limpiar y estandarizar la columna UBIGEO del Excel
    col_ubigeo = [c for c in df.columns if "ubigeo" in str(c).lower()][0]
    col_distrito = [c for c in df.columns if "distrito" in str(c).lower() or "departamento" in str(c).lower()][0]

    df["ubigeo"] = df[col_ubigeo].apply(parse_ubigeo)
    df["nombdist"] = df[col_distrito].apply(normalizar_texto)

    # 4. Filtrar solo los 50 distritos de Lima y Callao
    df_filtrado = df[df["ubigeo"].isin(ubigeos_validos)].copy()
    print(f"Distritos encontrados: {df_filtrado['ubigeo'].nunique()} de {len(ubigeos_validos)}")

    # 5. Identificar las columnas de anio dinamicamente (ej. 2018, 2019, ..., 2026)
    columnas_anios = []
    for col in df.columns:
        try:
            val = int(str(col).strip())
            if 2000 <= val <= 2100:
                columnas_anios.append(col)
        except ValueError:
            pass

    print(f"Años detectados: {columnas_anios}")

    # 6. Transformar de formato ancho a formato largo (melt)
    df_largo = df_filtrado.melt(
        id_vars=["ubigeo", "nombdist"],
        value_vars=columnas_anios,
        var_name="anio",
        value_name="poblacion_proyectada"
    )

    # 7. Convertir tipos de datos y ordenar
    df_largo["anio"] = df_largo["anio"].astype(int)
    df_largo["poblacion_proyectada"] = pd.to_numeric(df_largo["poblacion_proyectada"], errors="coerce").fillna(0).astype(int)
    
    # Reordenar columnas exactamente como se solicitó: ubigeo | nombdist | anio | poblacion_proyectada
    df_largo = df_largo[["ubigeo", "nombdist", "anio", "poblacion_proyectada"]]
    df_largo = df_largo.sort_values(by=["ubigeo", "anio"]).reset_index(drop=True)

    # 8. Guardar resultado
    salida_excel = "poblacion_proyectada_limpia.xlsx"
    salida_csv = "poblacion_proyectada_limpia.csv"
    
    df_largo.to_excel(salida_excel, index=False)
    df_largo.to_csv(salida_csv, index=False, encoding="utf-8")
    
    print(f"\n[OK] Datos procesados con exito!")
    print(f"Total registros generados: {len(df_largo):,} ({df_largo['ubigeo'].nunique()} distritos x {len(columnas_anios)} anios)")
    print(f"Guardado en Excel: {salida_excel}")
    print(f"Guardado en CSV: {salida_csv}")
    
    print("\nEjemplo de salida (primeras 10 filas):")
    print(df_largo.head(10).to_string(index=False))

    print("\nEjemplo de salida para Miraflores (150122):")
    print(df_largo[df_largo["ubigeo"] == "150122"].to_string(index=False))

if __name__ == "__main__":
    main()