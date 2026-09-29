# -*- coding: utf-8 -*-
import sys, io
# Forzar stdout a UTF-8 para que los caracteres especiales se impriman en Windows
# y desactivar el buffering para que los prints aparezcan de inmediato
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except AttributeError:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace", line_buffering=True)

import geopandas as gpd
import pandas as pd
import requests
import time
import os
import pickle

from shapely.geometry import Point, Polygon, MultiPolygon

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN: endpoints Overpass públicos (mirrors)
# Nota: se usan con requests directamente, SIN osmnx, para tener control
# total sobre el endpoint en cada intento.
# ─────────────────────────────────────────────────────────────────────────────
OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
    "https://overpass.openstreetmap.ru/api/interpreter",
]

CACHE_DIR = "pois_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# Área de búsqueda: bounding box de Lima Metropolitana + Callao (WGS84)
# Sur, Oeste, Norte, Este
LIMA_BBOX = (-12.53, -77.22, -11.55, -76.58)


def tags_to_overpass_filters(tags: dict) -> list[str]:
    """
    Convierte un dict de tags OSM en una lista de filtros Overpass QL.
    Cada clave genera un filtro separado para hacer OR entre ellos.

    Ejemplo:
      {"railway": "station", "highway": "bus_stop"}
      -> ['[railway=station]', '[highway=bus_stop]']

      {"amenity": ["hospital", "clinic"]}
      -> ['[amenity~hospital|clinic]']
    """
    filtros = []
    for key, value in tags.items():
        if isinstance(value, list):
            regex = "|".join(value)
            filtros.append(f'["{key}"~"{regex}"]')
        else:
            filtros.append(f'["{key}"="{value}"]')
    return filtros


def overpass_json_to_geodataframe(data: dict) -> gpd.GeoDataFrame:
    """
    Convierte la respuesta JSON de Overpass en un GeoDataFrame de puntos.
    - Nodos (node) → Point directo
    - Ways / Relations → centroide del bounding box de sus nodos
    """
    registros = []
    # Índice de nodos para reconstruir geometrías de ways
    nodos_idx = {el["id"]: el for el in data.get("elements", []) if el["type"] == "node"}

    for el in data.get("elements", []):
        tags = el.get("tags", {})
        geom = None

        if el["type"] == "node" and "lat" in el:
            geom = Point(el["lon"], el["lat"])

        elif el["type"] == "way":
            coords = []
            for nid in el.get("nodes", []):
                if nid in nodos_idx:
                    n = nodos_idx[nid]
                    coords.append((n["lon"], n["lat"]))
            if len(coords) >= 3:
                try:
                    geom = Polygon(coords).centroid
                except Exception:
                    pass
            elif len(coords) > 0:
                geom = Point(coords[0])

        elif el["type"] == "relation":
            # Usar el centro si Overpass lo devuelve
            center = el.get("center")
            if center:
                geom = Point(center["lon"], center["lat"])

        if geom is not None:
            registros.append({"geometry": geom, **tags})

    if not registros:
        return None

    gdf = gpd.GeoDataFrame(registros, geometry="geometry", crs="EPSG:4326")
    return gdf.to_crs(epsg=32718)


def fetch_pois_con_reintentos(nombre: str, tags: dict,
                               bbox: tuple = LIMA_BBOX,
                               max_intentos: int = 4,
                               pausa_entre_intentos: float = 20,
                               timeout_connect: int = 30,
                               timeout_read: int = 120) -> gpd.GeoDataFrame:
    """
    Descarga POIs de Overpass DIRECTAMENTE con requests (sin osmnx),
    lo que permite rotar el endpoint real en cada intento.

    Estrategias:
      - Cache en disco (.pkl): si ya se descargo, no toca la red.
      - Rotacion de 4 endpoints publicos.
      - Backoff exponencial entre intentos (20s, 40s, 80s).
      - timeout_connect=30s: falla rapido si el servidor no responde;
        timeout_read=120s:   espera suficiente para queries grandes.

    Returns GeoDataFrame o None si todos los intentos fallaron.
    """
    cache_path = os.path.join(CACHE_DIR, f"{nombre}.pkl")

    # ── 1. Caché ──────────────────────────────────────────────────────────────
    if os.path.exists(cache_path):
        print(f"  [{nombre}] Cargando desde caché local…")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    # -- 2. Construir query Overpass QL -----------------------------------
    s, w, n, e = bbox
    lista_filtros = tags_to_overpass_filters(tags)

    # Cada filtro genera sus propias lineas node/way/relation (OR entre filtros)
    lineas_union = []
    for f in lista_filtros:
        lineas_union.append(f"      node{f}({s},{w},{n},{e});")
        lineas_union.append(f"      way{f}({s},{w},{n},{e});")
        lineas_union.append(f"      relation{f}({s},{w},{n},{e});")
    union_block = "\n".join(lineas_union)

    # out center: devuelve el centroide de ways/relations directamente
    query = f"""[out:json][timeout:{timeout_read}];
(\n{union_block}\n);
out center;
"""

    # ── 3. Intentos con rotacion de endpoints ────────────────────────────────
    for intento in range(max_intentos):
        endpoint = OVERPASS_ENDPOINTS[intento % len(OVERPASS_ENDPOINTS)]
        print(f"  [{nombre}] Intento {intento + 1}/{max_intentos} -> {endpoint}", flush=True)
        try:
            resp = requests.post(
                endpoint,
                data={"data": query},
                timeout=(timeout_connect, timeout_read),  # (connect, read)
                headers={"User-Agent": "tesis-inmobiliaria-lima/1.0"},
            )
            resp.raise_for_status()
            data = resp.json()

            gdf = overpass_json_to_geodataframe(data)
            if gdf is None or len(gdf) == 0:
                print(f"  [{nombre}] Respuesta vacia en {endpoint}", flush=True)
                continue

            # Guardar en cache
            with open(cache_path, "wb") as f:
                pickle.dump(gdf, f)

            print(f"  [{nombre}] OK: {len(gdf)} elementos encontrados", flush=True)
            return gdf

        except Exception as e:
            print(f"  [{nombre}] Error intento {intento + 1}: {e}", flush=True)
            if intento < max_intentos - 1:
                espera = pausa_entre_intentos * (2 ** intento)
                print(f"  [{nombre}] Esperando {espera:.0f}s...", flush=True)
                time.sleep(espera)

    print(f"  [{nombre}] Todos los intentos fallaron. Se omitira.", flush=True)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# CARGA Y PREPARACIÓN DE DISTRITOS
# ─────────────────────────────────────────────────────────────────────────────
distritos = gpd.read_file("DISTRITO.gpkg")

# Reproyectar a UTM 18S antes de calcular centroide (para que quede en metros)
distritos = distritos.to_crs(epsg=32718)

# Calcular área en km² (UTM está en metros → dividir entre 1_000_000)
distritos["area_km2"] = distritos.geometry.area / 1_000_000

# Calcular centroide de cada polígono
distritos["centroide"] = distritos.geometry.centroid

# Extraer lat/long del centroide
distritos["lon_centroide"] = distritos["centroide"].x
distritos["lat_centroide"] = distritos["centroide"].y

lima_metropolitana = distritos[
    (distritos["nombdep"] == "LIMA") & (distritos["nombprov"] == "LIMA")
]
callao = distritos[distritos["nombdep"] == "CALLAO"]

lima_callao_final = pd.concat([lima_metropolitana, callao])

print(f"Distritos cargados: {len(lima_callao_final)}")
print(lima_callao_final["nombdist"].tolist())

# Tomar el centroide (se cambia a representative_point ya que el punto no cae exactamente en el centro,
# y al ser el punto de referencia eso generaría un problema)
punto_referencia = lima_callao_final[
    lima_callao_final["nombdist"] == "LIMA"
]["geometry"].values[0].representative_point()

# --- NUEVO: reemplazar el centroide de LIMA por el punto representativo ---
idx_lima = lima_callao_final[lima_callao_final["nombdist"] == "LIMA"].index

lima_callao_final.loc[idx_lima, "centroide"] = lima_callao_final.loc[idx_lima, "geometry"].apply(
    lambda geom: geom.representative_point()
)

# Recalcular lon/lat solo para que quede consistente con el nuevo centroide
lima_callao_final["lon_centroide"] = lima_callao_final["centroide"].x
lima_callao_final["lat_centroide"] = lima_callao_final["centroide"].y
# --- FIN NUEVO ---

# Calcular distancia de cada distrito al punto de referencia (en metros, porque estás en UTM)
lima_callao_final["distancia_centro_km"] = lima_callao_final["centroide"].distance(punto_referencia) / 1000

print(f"\nDuplicados en nombdist: {lima_callao_final['nombdist'].duplicated().sum()}")

tabla_distritos = lima_callao_final[["nombdist", "ubigeo", "distancia_centro_km"]]
print(tabla_distritos)


# ─────────────────────────────────────────────────────────────────────────────
# DESCARGA DE PUNTOS DE INTERÉS (POIs) DESDE OSM
# Búsqueda directa con requests sobre el bbox de Lima+Callao (sin osmnx)
# ─────────────────────────────────────────────────────────────────────────────
categorias = {
    "colegio":             {"amenity": "school"},
    "hospital":            {"amenity": ["hospital", "clinic"]},
    "estacion_transporte": {"railway": "station", "highway": "bus_stop"},
    "centro_comercial":    {"shop": "mall"},
    "parque":              {"leisure": "park"},
    "universidad":         {"amenity": "university"},
}

print("\n── Descargando POIs desde Overpass / OpenStreetMap ──")
pois = {}
for nombre, tags in categorias.items():
    print(f"\n[{nombre}]")
    gdf = fetch_pois_con_reintentos(nombre=nombre, tags=tags,
                                     max_intentos=4,
                                     pausa_entre_intentos=30)
    if gdf is not None:
        pois[nombre] = gdf

print(f"\n── POIs descargados correctamente: {list(pois.keys())} ──")


# ─────────────────────────────────────────────────────────────────────────────
# CÁLCULO DE DISTANCIAS MÍNIMAS POR DISTRITO
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Calculando distancias mínimas por distrito ──")
for nombre, gdf_poi in pois.items():
    # Las geometrías ya son puntos (centroide de ways/relations calculado en el parser)
    # Pero por si alguna no lo es, aplicamos centroid de forma segura
    gdf_poi = gdf_poi.copy()
    gdf_poi["geometry"] = gdf_poi.geometry.centroid

    distancias_min = []
    for centroide in lima_callao_final["centroide"]:
        dist_min = gdf_poi.geometry.distance(centroide).min() / 1000  # km
        distancias_min.append(dist_min)

    lima_callao_final[f"dist_{nombre}_km"] = distancias_min
    print(f"  dist_{nombre}_km → OK")

# ─────────────────────────────────────────────────────────────────────────────
# RESULTADO FINAL
# ─────────────────────────────────────────────────────────────────────────────
cols_resultado = ["nombdist", "ubigeo", "distancia_centro_km"] + \
                 [f"dist_{n}_km" for n in pois.keys()]

print("\n── Resultado final ──")
print(lima_callao_final[cols_resultado])

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 8))
lima_callao_final.to_crs(epsg=4326).plot(ax=ax, color="lightgray", edgecolor="black")

# Dibujar el bbox usado
sur, oeste, norte, este = LIMA_BBOX  # usa la constante definida arriba
ax.plot([oeste, este, este, oeste, oeste], [sur, sur, norte, norte, sur], color="red", linewidth=2)

plt.show()

# Guardar a Excel para el dataset de tesis
orden_columnas = [
    "ubigeo", "nombdist", "area_km2", "distancia_centro_km",
    "dist_colegio_km", "dist_hospital_km", "dist_estacion_transporte_km",
    "dist_centro_comercial_km", "dist_parque_km", "dist_universidad_km"
]

df_resultado = lima_callao_final[orden_columnas].copy()

with pd.ExcelWriter("distancias_pois.xlsx", engine="openpyxl") as writer:
    df_resultado.to_excel(writer, index=False, sheet_name="Distritos")

    # Formato: autoajuste de ancho de columnas + congelar cabecera
    ws = writer.sheets["Distritos"]
    for col in ws.columns:
        max_len = max(len(str(cell.value)) if cell.value is not None else 0 for cell in col)
        ws.column_dimensions[col[0].column_letter].width = max_len + 4
    ws.freeze_panes = "A2"  # Congela la fila de encabezados

print("\n✓ Guardado en distancias_pois.xlsx")