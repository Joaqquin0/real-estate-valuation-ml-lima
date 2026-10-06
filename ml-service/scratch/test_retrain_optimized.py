"""
scratch/test_retrain_optimized.py
Lanza y monitorea el reentrenamiento optimizado de Venta y Alquiler con las 4 palancas.
"""
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from dotenv import load_dotenv
load_dotenv()

from app.core.model_loader import model_state
from app.services.entrenamiento_service import (
    iniciar_entrenamiento_venta,
    iniciar_entrenamiento_alquiler,
    get_job_status,
)


def ejecutar_job(tipo: str, func_iniciar, nombre_modelo: str):
    print("=" * 60)
    print(f"INICIANDO REENTRENAMIENTO OPTIMIZADO: {tipo.upper()}")
    print("=" * 60)
    
    job_id = func_iniciar(
        nombre_modelo=nombre_modelo,
        guardar_como_activo=True,
        state=model_state,
    )
    print(f"Job ID: {job_id}")

    while True:
        status = get_job_status(job_id)
        estado = status.get("estado")
        progreso = status.get("progreso")
        print(f"[{estado}] {progreso}")
        
        if estado in ("completado", "fallido"):
            break
        time.sleep(3)

    if estado == "completado":
        metricas = status.get("metricas")
        print("\n" + "-" * 50)
        print(f"RESULTADOS {tipo.upper()}:")
        print(f"  MAPE (%): {metricas.mape_pct:.4f}%")
        print(f"  R²:       {metricas.r2:.4f}")
        print(f"  MAE:      S/ {metricas.mae:,.2f}")
        print(f"  RMSE:     S/ {metricas.rmse:,.2f}")
        print(f"  Train N:  {metricas.n_train:,}")
        print(f"  Test N:   {metricas.n_test:,}")
        print(f"  Benchmark Superado: {metricas.supera_benchmark}")
        print(f"  MLflow Run ID: {status.get('mlflow_run_id')}")
        print("-" * 50 + "\n")
        return metricas
    else:
        print(f"\nERROR en {tipo}: {status.get('error')}")
        sys.exit(1)


if __name__ == "__main__":
    print("\n>>> INICIANDO PRUEBAS DE LAS 4 PALANCAS EN VENTA Y ALQUILER <<<\n")
    metricas_venta = ejecutar_job("venta", iniciar_entrenamiento_venta, "xgboost_venta_v2")
    metricas_alquiler = ejecutar_job("alquiler", iniciar_entrenamiento_alquiler, "xgboost_alquiler_v1")

    print("\n" + "=" * 60)
    print("RESUMEN FINAL DE METRICAS TRAS LAS 4 PALANCAS")
    print("=" * 60)
    print(f"VENTA:    MAPE = {metricas_venta.mape_pct:.2f}% | R² = {metricas_venta.r2:.4f} | MAE = S/ {metricas_venta.mae:,.2f}")
    print(f"ALQUILER: MAPE = {metricas_alquiler.mape_pct:.2f}% | R² = {metricas_alquiler.r2:.4f} | MAE = S/ {metricas_alquiler.mae:,.2f}")
    print("=" * 60)
