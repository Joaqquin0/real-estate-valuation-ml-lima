"""
app/routers/prediccion.py
─────────────────────────
Router principal del servicio: predicción de precio de venta.

POST /api/v1/prediccion/venta
  → Recibe datos del inmueble
  → Devuelve precio predicho + intervalo de confianza + valores SHAP

Fase posterior: POST /api/v1/prediccion/alquiler (mismo patrón)
"""

from __future__ import annotations

from fastapi import APIRouter

from app.core.model_loader import model_state
from app.schemas.prediccion import PrediccionVentaRequest, PrediccionVentaResponse
from app.services.prediccion_service import predecir_venta

router = APIRouter(prefix="/api/v1/prediccion", tags=["Predicción"])


@router.post(
    "/venta",
    response_model=PrediccionVentaResponse,
    summary="Predecir precio de venta",
    description=(
        "Predice el precio de venta de un inmueble en Lima Metropolitana.\n\n"
        "**Variables del inmueble requeridas**: distrito, superficie, habitaciones, "
        "baños, piso y antigüedad.\n\n"
        "Las variables contextuales del distrito (NSE, criminalidad, distancias geoespaciales "
        "y datos demográficos) son resueltas automáticamente por el servicio.\n\n"
        "**Respuesta incluye**:\n"
        "- Precio en Soles Constantes (base dic 2009) y Soles Nominales (IPC vigente)\n"
        "- Precio por m² en ambas unidades\n"
        "- Intervalo de confianza ± MAPE del modelo\n"
        "- Top 10 variables SHAP con su contribución individual al precio predicho "
        "(para mostrar explicabilidad al usuario final)\n"
        "- Metadata del modelo activo (versión, IPC, métricas)\n\n"
        "Si el distrito no está disponible, retorna **HTTP 422** con la lista de "
        "distritos válidos."
    ),
    responses={
        200: {"description": "Predicción exitosa con valores SHAP."},
        422: {
            "description": "Datos de entrada inválidos o distrito no disponible.",
            "content": {
                "application/json": {
                    "example": {
                        "error": "distrito_no_disponible",
                        "message": "El distrito 'Baños de Chosica' no está disponible en el modelo actual.",
                        "distritos_disponibles": ["Ate Vitarte", "Barranco", "..."],
                        "total_distritos": 22,
                        "sugerencia": "Usa GET /api/v1/distritos para ver la lista completa.",
                    }
                }
            },
        },
        500: {"description": "Error interno durante inferencia o cálculo SHAP."},
    },
)
def predecir_precio_venta(request: PrediccionVentaRequest) -> PrediccionVentaResponse:
    """
    Endpoint de predicción de precio de venta.
    Toda la lógica está en prediccion_service.predecir_venta().
    """
    return predecir_venta(req=request, state=model_state)
