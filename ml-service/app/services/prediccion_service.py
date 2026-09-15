"""
app/services/prediccion_service.py
───────────────────────────────────
Lógica pura de predicción de precio de venta.

Extrae y formaliza la función `predecir()` del notebook
07_validacion_prediccion.py como función sin side effects:
  - Sin prints ni gráficos
  - Sin dependencia directa de CSV o modelo (solo recibe ModelState)
  - Retorna PrediccionVentaResponse listo para serializar

Flujo por request (< 50ms):
  1. Validar distrito → HTTPException 422 si no está en encoding_map
  2. Obtener contexto distrital via data_provider (IContextDataProvider)
  3. Construir X_fila con las 33 features
  4. modelo.predict(X_fila)  →  pred_const = exp(pred_log) - 1
  5. explainer.shap_values(X_fila)  →  array[33]
  6. Seleccionar top N por |shap_val|  →  lista de ContribucionFeature
  7. Convertir a nominales con IPC
  8. Calcular intervalo de confianza (± MAPE%)
  9. Armar y retornar PrediccionVentaResponse
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from fastapi import HTTPException

from app.schemas.prediccion import (
    ContribucionFeature,
    Explicabilidad,
    InfoModelo,
    IntervaloConfianza,
    PrediccionVentaRequest,
    PrediccionVentaResponse,
    ValoresPrediccion,
)

if TYPE_CHECKING:
    from app.core.model_loader import ModelState


# ─── Etiquetas legibles para las features ─────────────────────────────────────
# Permite que el frontend muestre nombres amigables en lugar de nombres técnicos.

_FEATURE_LABELS: dict[str, str] = {
    "Superficie":                  "Superficie ({v} m²)",
    "Habitaciones":                "Habitaciones ({v})",
    "Banios":                      "Baños ({v})",
    "Garajes":                     "Cocheras ({v})",
    "Piso":                        "Piso {v}",
    "Vista_Exterior":              "Vista exterior",
    "Antiguedad":                  "Antigüedad ({v} años)",
    "Anio":                        "Año {v}",
    "Trimestre":                   "Trimestre {v}",
    "pct_NSE_A":                   "NSE A en zona ({v:.1%})",
    "pct_NSE_B":                   "NSE B en zona ({v:.1%})",
    "pct_NSE_C":                   "NSE C en zona ({v:.1%})",
    "pct_NSE_D":                   "NSE D en zona ({v:.1%})",
    "pct_NSE_E":                   "NSE E en zona ({v:.1%})",
    "tasa_robo":                   "Tasa robos zona ({v:.2f})",
    "tasa_hurto":                  "Tasa hurtos zona ({v:.2f})",
    "poblacion_proyectada":        "Población distrito ({v:,.0f} hab)",
    "area_distrito_km2":           "Área distrito ({v:.1f} km²)",
    "densidad_hab_km2":            "Densidad poblacional ({v:.0f} hab/km²)",
    "distancia_centro_km":         "Dist. al centro ({v:.1f} km)",
    "dist_colegio_km":             "Dist. al colegio más cercano ({v:.1f} km)",
    "dist_hospital_km":            "Dist. al hospital más cercano ({v:.1f} km)",
    "dist_estacion_transporte_km": "Dist. a estación de transporte ({v:.1f} km)",
    "dist_centro_comercial_km":    "Dist. a centro comercial ({v:.1f} km)",
    "dist_parque_km":              "Dist. al parque más cercano ({v:.1f} km)",
    "dist_universidad_km":         "Dist. a universidad ({v:.1f} km)",
    "periodo_numerico":            "Periodo temporal ({v})",
    "m2_por_habitacion":           "m² por habitación ({v:.1f})",
    "ratio_banios_hab":            "Ratio baños/habitaciones ({v:.2f})",
    "tiene_garaje":                "Tiene cochera",
    "es_piso_alto":                "Piso alto (≥8)",
    "superficie_cuadrado":         "Superficie² ({v:.2f})",
    "distrito_encoded":            "Distrito ({distrito})",
}


def _build_label(feature: str, valor: float, distrito: str) -> str:
    """Genera la etiqueta legible de una feature para el response SHAP."""
    template = _FEATURE_LABELS.get(feature, feature)
    try:
        if "{v" in template:
            return template.format(v=valor, distrito=distrito)
        elif "{distrito}" in template:
            return template.format(distrito=distrito)
        return template
    except (ValueError, KeyError):
        return template


# ─── Función principal ────────────────────────────────────────────────────────

def predecir_venta(
    req: PrediccionVentaRequest,
    state: "ModelState",
) -> PrediccionVentaResponse:
    """
    Predice el precio de venta de un inmueble y retorna la explicabilidad SHAP.

    Args:
        req:   Request validado por Pydantic.
        state: Singleton con modelo, explainer, encoding_map y data_provider.

    Returns:
        PrediccionVentaResponse con predicción + SHAP + metadata.

    Raises:
        HTTPException 422: Si el distrito no está disponible en el modelo.
        HTTPException 500: Si ocurre un error inesperado durante la inferencia.
    """

    # ── 1. Validar distrito ───────────────────────────────────────────────────
    if req.distrito not in state.encoding_map:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "distrito_no_disponible",
                "message": (
                    f"El distrito '{req.distrito}' no está disponible en el modelo actual. "
                    f"El modelo cubre {len(state.encoding_map)} distritos de Lima Metropolitana."
                ),
                "distritos_disponibles": sorted(state.encoding_map.keys()),
                "total_distritos": len(state.encoding_map),
                "sugerencia": (
                    "Usa GET /api/v1/distritos para ver la lista completa, "
                    "o GET /api/v1/modelo/info para más detalles."
                ),
            },
        )

    # ── 2. Contexto distrital ─────────────────────────────────────────────────
    ctx = state.data_provider.get_distrito_context(req.distrito)  # type: ignore[union-attr]

    # ── 3. Construir fila de features ─────────────────────────────────────────
    from datetime import datetime
    ahora = datetime.now()
    anio: int = req.anio if req.anio is not None else ahora.year
    trimestre: int = req.trimestre if req.trimestre is not None else ((ahora.month - 1) // 3 + 1)

    sup = req.superficie
    hab = req.habitaciones
    ban = req.banios
    gar = req.garajes

    fila: dict[str, float] = {
        "Anio":                        float(anio),
        "Trimestre":                   float(trimestre),
        "Superficie":                  sup,
        "Habitaciones":                float(hab),
        "Banios":                      float(ban),
        "Garajes":                     float(gar),
        "Piso":                        float(req.piso),
        "Vista_Exterior":              1.0 if req.vista_exterior else 0.0,
        "Antiguedad":                  float(req.antiguedad),
        # NSE y contexto distrital (del data_provider)
        "pct_NSE_A":                   float(ctx.get("pct_NSE_A", 0)),
        "pct_NSE_B":                   float(ctx.get("pct_NSE_B", 0)),
        "pct_NSE_C":                   float(ctx.get("pct_NSE_C", 0)),
        "pct_NSE_D":                   float(ctx.get("pct_NSE_D", 0)),
        "pct_NSE_E":                   float(ctx.get("pct_NSE_E", 0)),
        "tasa_robo":                   float(ctx.get("tasa_robo", 0)),
        "tasa_hurto":                  float(ctx.get("tasa_hurto", 0)),
        "poblacion_proyectada":        float(ctx.get("poblacion_proyectada", 0)),
        "area_distrito_km2":           float(ctx.get("area_distrito_km2", 0)),
        "distancia_centro_km":         float(ctx.get("distancia_centro_km", 0)),
        "dist_colegio_km":             float(ctx.get("dist_colegio_km", 0)),
        "dist_hospital_km":            float(ctx.get("dist_hospital_km", 0)),
        "dist_estacion_transporte_km": float(ctx.get("dist_estacion_transporte_km", 0)),
        "dist_centro_comercial_km":    float(ctx.get("dist_centro_comercial_km", 0)),
        "dist_parque_km":              float(ctx.get("dist_parque_km", 0)),
        "dist_universidad_km":         float(ctx.get("dist_universidad_km", 0)),
        "densidad_hab_km2":            float(ctx.get("densidad_hab_km2", 0)),
        # Features engineered (replicadas de 02_preprocesamiento.py)
        "periodo_numerico":            float(anio * 4 + trimestre),
        "m2_por_habitacion":           sup / (hab + 1),
        "ratio_banios_hab":            ban / (hab + 0.1),
        "tiene_garaje":                1.0 if gar > 0 else 0.0,
        "es_piso_alto":                1.0 if req.piso >= 8 else 0.0,
        "superficie_cuadrado":         (sup / 100) ** 2,
        "distrito_encoded":            state.encoding_map[req.distrito],
    }

    X = pd.DataFrame([fila])[state.feature_cols]

    # ── 4. Predicción (escala log → escala original) ──────────────────────────
    try:
        pred_log = state.modelo.predict(X)[0]  # type: ignore[union-attr]
        pred_const = np.expm1(pred_log)         # Soles Constantes (Base Dic 2009)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Error durante la inferencia del modelo: {exc}",
        ) from exc

    # ── 5. SHAP values (escala log) ───────────────────────────────────────────
    try:
        shap_vals_raw = state.explainer.shap_values(X)  # type: ignore[union-attr]
        # shap_vals_raw puede ser array 2D (1, n_features) o 1D según versión SHAP
        shap_arr: np.ndarray = (
            shap_vals_raw[0] if shap_vals_raw.ndim == 2 else shap_vals_raw
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Error calculando SHAP values: {exc}",
        ) from exc

    # ── 6. Top N contribuciones SHAP ─────────────────────────────────────────
    n_top: int = state.config.get("shap_top_n", 10)
    feature_names = state.feature_cols
    fila_vals = X.iloc[0].to_dict()

    # Ordenar por |shap_value| descendente
    indexed = sorted(
        enumerate(shap_arr),
        key=lambda t: abs(t[1]),
        reverse=True,
    )

    contribuciones: list[ContribucionFeature] = []
    for idx, sv in indexed[:n_top]:
        fname = feature_names[idx]
        fval  = float(fila_vals.get(fname, 0.0))
        contribuciones.append(
            ContribucionFeature(
                feature=fname,
                label=_build_label(fname, fval, req.distrito),
                shap_value=round(float(sv), 6),
                valor_feature=round(fval, 4),
                impacto=(
                    "positivo" if sv > 1e-6
                    else "negativo" if sv < -1e-6
                    else "neutro"
                ),
            )
        )

    # ── 7. Conversión a soles nominales ───────────────────────────────────────
    ipc_cfg  = state.config["ipc_actual"]
    ipc_val  = float(ipc_cfg["valor"])
    ipc_factor = ipc_val / 100.0

    pred_nominal    = pred_const * ipc_factor
    pred_m2_const   = pred_const  / sup
    pred_m2_nominal = pred_nominal / sup

    # ── 8. Intervalo de confianza ─────────────────────────────────────────────
    mape = float(state.config["mape_test"])
    ic_factor = mape / 100.0

    ic_inf_const   = pred_const   * (1 - ic_factor)
    ic_sup_const   = pred_const   * (1 + ic_factor)
    ic_inf_nominal = pred_nominal * (1 - ic_factor)
    ic_sup_nominal = pred_nominal * (1 + ic_factor)

    # ── 9. Valor base SHAP (media global del entrenamiento) ───────────────────
    base_log   = float(state.explainer.expected_value)  # type: ignore[union-attr]
    base_const = math.expm1(base_log) if not math.isnan(base_log) else state.media_global

    # ── 10. Armar response ────────────────────────────────────────────────────
    return PrediccionVentaResponse(
        distrito=req.distrito,
        superficie_m2=sup,
        prediccion=ValoresPrediccion(
            soles_constantes=round(pred_const, 2),
            soles_nominales=round(pred_nominal, 2),
            precio_m2_constantes=round(pred_m2_const, 2),
            precio_m2_nominales=round(pred_m2_nominal, 2),
        ),
        intervalo_confianza=IntervaloConfianza(
            inferior_nominales=round(ic_inf_nominal, 2),
            superior_nominales=round(ic_sup_nominal, 2),
            inferior_constantes=round(ic_inf_const, 2),
            superior_constantes=round(ic_sup_const, 2),
            mape_pct=mape,
        ),
        explicabilidad=Explicabilidad(
            valor_base_constantes=round(base_const, 2),
            contribuciones=contribuciones,
            n_features_mostradas=len(contribuciones),
            n_features_total=len(feature_names),
        ),
        modelo=InfoModelo(
            version=state.config.get("modelo_version", "v2"),
            mape_test=mape,
            r2_test=float(state.config.get("r2_test", 0.0)),
            ipc_factor=ipc_val,
            periodo_ipc=ipc_cfg.get("periodo", "2026-Q1"),
            train_periodo=state.config.get("train_periodo", "2016-2023"),
            test_periodo=state.config.get("test_periodo", "2024-2025"),
        ),
    )
