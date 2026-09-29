"""
Orquestador Principal del Pipeline ETL (CLI de Producción).
Ejecuta la ingesta, limpieza y carga a la Base de Datos de Entrenamiento (PostgreSQL).

Uso:
    python -m src.etl.run_etl --init-db
    python -m src.etl.run_etl --run-all --truncate
    python -m src.etl.run_etl --target venta
    python -m src.etl.run_etl --target alquiler
"""
import argparse
import logging
import sys
from pathlib import Path

# Asegurar que la raíz del proyecto esté en el PYTHONPATH
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.db.connection import init_db, check_connection
from src.etl.extract import extract_contexto_distrital, extract_raw_venta, extract_raw_alquiler
from src.etl.transform import (
    transform_catalogo_distritos,
    transform_contexto_distrital,
    transform_venta,
    transform_alquiler
)
from src.etl.load import (
    load_distritos,
    load_contexto_distrital,
    load_inmuebles_venta,
    load_inmuebles_alquiler,
    load_auditoria_flags
)
from src.etl.tracker import PipelineTracker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("ETL_PIPELINE")

def run_pipeline(target: str = "all", truncate: bool = False, ejecutado_por: str = "CLI_LOCAL"):
    """Ejecuta el pipeline ETL completo con auditoría en pipeline_ejecuciones."""
    tipo_pipeline = f"ETL_INGESTA_{target.upper()}"
    
    with PipelineTracker(tipo_ejecucion=tipo_pipeline, ejecutado_por=ejecutado_por) as tracker:
        logger.info(f"=== INICIANDO PIPELINE ETL [{target.upper()}] ===")
        
        # 1. Extracción y Carga del Contexto Distrital
        df_raw_contexto = extract_contexto_distrital()
        df_distritos = transform_catalogo_distritos(df_raw_contexto)
        distrito_id_map = load_distritos(df_distritos)
        
        df_contexto_db = transform_contexto_distrital(df_raw_contexto, distrito_id_map)
        load_contexto_distrital(df_contexto_db)
        
        total_leidos = 0
        total_guardados = 0
        total_filtrados = 0
        resumen_operaciones = []
        
        # 2. Pipeline de Venta
        if target in ["all", "venta"]:
            logger.info("--- Procesando Inmuebles en Venta ---")
            df_raw_venta = extract_raw_venta()
            n_leidos = len(df_raw_venta)
            df_venta_db, df_flags_venta = transform_venta(df_raw_venta, distrito_id_map)
            n_guardados = len(df_venta_db)
            n_filtrados = n_leidos - n_guardados
            
            load_inmuebles_venta(df_venta_db, truncate=truncate)
            load_auditoria_flags(df_flags_venta, truncate=truncate)
            
            total_leidos += n_leidos
            total_guardados += n_guardados
            total_filtrados += n_filtrados
            resumen_operaciones.append(f"Venta: {n_guardados:,} guardados ({n_filtrados:,} descartados)")
            
        # 3. Pipeline de Alquiler
        if target in ["all", "alquiler"]:
            logger.info("--- Procesando Inmuebles en Alquiler ---")
            df_raw_alquiler = extract_raw_alquiler()
            n_leidos = len(df_raw_alquiler)
            df_alquiler_db, df_flags_alquiler = transform_alquiler(df_raw_alquiler, distrito_id_map)
            n_guardados = len(df_alquiler_db)
            n_filtrados = n_leidos - n_guardados
            
            load_inmuebles_alquiler(df_alquiler_db, truncate=truncate)
            load_auditoria_flags(df_flags_alquiler, truncate=False) # Append
            
            total_leidos += n_leidos
            total_guardados += n_guardados
            total_filtrados += n_filtrados
            resumen_operaciones.append(f"Alquiler: {n_guardados:,} guardados ({n_filtrados:,} descartados)")
            
        # 4. Registrar éxito y métricas
        observaciones_txt = " | ".join(resumen_operaciones)
        tracker.completar_exito(
            leidos=total_leidos,
            guardados=total_guardados,
            filtrados=total_filtrados,
            metricas={"distritos_cubiertos": len(distrito_id_map), "target": target},
            observaciones=observaciones_txt
        )
        logger.info(f"=== PIPELINE FINALIZADO CON ÉXITO: {observaciones_txt} ===")

def main():
    parser = argparse.ArgumentParser(description="Orquestador ETL para Base de Datos de Entrenamiento (PostgreSQL)")
    parser.add_argument("--init-db", action="store_true", help="Crea la base de datos y ejecuta el script DDL de tablas")
    parser.add_argument("--run-all", action="store_true", help="Ejecuta la ingesta completa (Venta + Alquiler + Contexto)")
    parser.add_argument("--target", choices=["all", "venta", "alquiler"], default="all", help="Dataset específico a procesar")
    parser.add_argument("--truncate", action="store_true", help="Limpia las tablas antes de insertar los registros")
    parser.add_argument("--check-db", action="store_true", help="Verifica si PostgreSQL está accesible")
    parser.add_argument("--ejecutado-por", default="CLI_LOCAL", help="Identificador del autor o job que dispara el pipeline")
    
    args = parser.parse_args()
    
    if args.check_db:
        ok = check_connection()
        if ok:
            print("Conexión con PostgreSQL: EXITOSA.")
            sys.exit(0)
        else:
            print("Conexión con PostgreSQL: FALLÓ. Revisa tus credenciales en .env")
            sys.exit(1)
            
    if args.init_db:
        logger.info("Inicializando esquema de base de datos...")
        init_db()
        logger.info("Esquema inicializado correctamente.")
        
    if args.run_all or (not args.init_db and not args.check_db):
        run_pipeline(target=args.target, truncate=args.truncate, ejecutado_por=args.ejecutado_por)

if __name__ == "__main__":
    main()
