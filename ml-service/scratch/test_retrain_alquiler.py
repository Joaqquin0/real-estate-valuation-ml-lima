"""
scratch/test_retrain_alquiler.py
Lanza y monitorea el reentrenamiento del modelo de alquiler con Target Encoding sobre canon por m2
directamente desde PostgreSQL.
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
    iniciar_entrenamiento_alquiler,
    get_job_status,
)

print("Iniciando reentrenamiento del modelo de alquiler con target encoding por m2...")
job_id = iniciar_entrenamiento_alquiler(
    nombre_modelo="xgboost_alquiler_v1",
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
    print("\n" + "=" * 50)
    print("REENTRENAMIENTO COMPLETADO EXITOSAMENTE")
    print("=" * 50)
    print(f"MAPE (%): {metricas.get('mape_pct'):.4f}%")
    print(f"R²:       {metricas.get('r2'):.4f}")
    print(f"MAE:      S/ {metricas.get('mae'):.2f}")
    print(f"RMSE:     S/ {metricas.get('rmse'):.2f}")
    print(f"Train N:  {metricas.get('n_train')}")
    print(f"Test N:   {metricas.get('n_test')}")
    print(f"Supera benchmark: {metricas.get('supera_benchmark')}")
    print("=" * 50)
else:
    print(f"\nERROR: {status.get('error')}")
    sys.exit(1)
