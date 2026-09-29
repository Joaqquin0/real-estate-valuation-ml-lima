-- =============================================================================
-- ESQUEMA DDL: BASE DE DATOS DE ENTRENAMIENTO (ML / OLAP)
-- Proyecto: Sistema Web de Valoración Inmobiliaria y Análisis Financiero - Lima
-- Base de datos objetivo: inmobiliaria_ml_db
-- Motor: PostgreSQL 14+
-- =============================================================================

-- 1. EXTENSIONES
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- 2. TABLA DIMENSIONAL: DISTRITOS (22 distritos representativos de Lima Metropolitana)
CREATE TABLE IF NOT EXISTS distritos (
    id SERIAL PRIMARY KEY,
    nombre VARCHAR(100) UNIQUE NOT NULL,
    ubigeo VARCHAR(6) UNIQUE NOT NULL,
    area_km2 DECIMAL(10, 4) NOT NULL CHECK (area_km2 > 0),
    creado_en TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

COMMENT ON TABLE distritos IS 'Catálogo maestro de los 22 distritos modelados (excluye distritos con n<=1 en transacciones).';

-- 3. TABLA CONTEXTUAL ANUAL: DISTRITO_ANIO_CONTEXTO (Single Source of Truth)
CREATE TABLE IF NOT EXISTS distrito_anio_contexto (
    id SERIAL PRIMARY KEY,
    distrito_id INTEGER NOT NULL REFERENCES distritos(id) ON DELETE RESTRICT,
    anio SMALLINT NOT NULL CHECK (anio BETWEEN 2016 AND 2030),
    
    -- Variables Socioeconómicas (INEI / ENAHO 2025 - nulables para años tempranos sin muestra)
    pct_nse_a DECIMAL(6, 3) CHECK (pct_nse_a IS NULL OR (pct_nse_a >= 0 AND pct_nse_a <= 100)),
    pct_nse_b DECIMAL(6, 3) CHECK (pct_nse_b IS NULL OR (pct_nse_b >= 0 AND pct_nse_b <= 100)),
    pct_nse_c DECIMAL(6, 3) CHECK (pct_nse_c IS NULL OR (pct_nse_c >= 0 AND pct_nse_c <= 100)),
    pct_nse_d DECIMAL(6, 3) CHECK (pct_nse_d IS NULL OR (pct_nse_d >= 0 AND pct_nse_d <= 100)),
    pct_nse_e DECIMAL(6, 3) CHECK (pct_nse_e IS NULL OR (pct_nse_e >= 0 AND pct_nse_e <= 100)),
    
    -- Seguridad Ciudadana: Denuncias por cada 10,000 hab. (PNP 2019-2025; 2016-2018 imputado)
    tasa_robo DECIMAL(12, 4) NOT NULL CHECK (tasa_robo >= 0),
    tasa_hurto DECIMAL(12, 4) NOT NULL CHECK (tasa_hurto >= 0),
    tasa_denuncias DECIMAL(12, 4) NOT NULL CHECK (tasa_denuncias >= 0),
    
    -- Demografía y Densidad (INEI 2018-2025; 2016-2017 imputado)
    poblacion_proyectada INTEGER NOT NULL CHECK (poblacion_proyectada > 0),
    densidad_hab_km2 DECIMAL(12, 4) NOT NULL CHECK (densidad_hab_km2 > 0),
    
    -- Distancias Cartográficas y Equipamiento Urbano (OpenStreetMap / INEI)
    distancia_centro_km DECIMAL(10, 4) NOT NULL CHECK (distancia_centro_km >= 0),
    dist_colegio_km DECIMAL(10, 4) NOT NULL CHECK (dist_colegio_km >= 0),
    dist_hospital_km DECIMAL(10, 4) NOT NULL CHECK (dist_hospital_km >= 0),
    dist_estacion_transporte_km DECIMAL(10, 4) NOT NULL CHECK (dist_estacion_transporte_km >= 0),
    dist_centro_comercial_km DECIMAL(10, 4) NOT NULL CHECK (dist_centro_comercial_km >= 0),
    dist_parque_km DECIMAL(10, 4) NOT NULL CHECK (dist_parque_km >= 0),
    dist_universidad_km DECIMAL(10, 4) NOT NULL CHECK (dist_universidad_km >= 0),
    
    creado_en TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_distrito_anio UNIQUE (distrito_id, anio)
);

COMMENT ON TABLE distrito_anio_contexto IS 'Tabla analítica de 202 observaciones distrital-anual que alimenta el pipeline de ML.';

-- 4. TABLA DE ENTRENAMIENTO: INMUEBLES EN VENTA (BCRP / Portales)
CREATE TABLE IF NOT EXISTS dataset_inmuebles_venta (
    id BIGSERIAL PRIMARY KEY,
    distrito_id INTEGER NOT NULL REFERENCES distritos(id) ON DELETE RESTRICT,
    anio SMALLINT NOT NULL CHECK (anio BETWEEN 2016 AND 2030),
    trimestre SMALLINT NOT NULL CHECK (trimestre BETWEEN 1 AND 4),
    
    -- Características Físicas del Predio
    superficie_m2 DECIMAL(10, 2) NOT NULL CHECK (superficie_m2 > 0),
    habitaciones SMALLINT NOT NULL CHECK (habitaciones >= 0),
    banos SMALLINT NOT NULL CHECK (banos >= 0),
    garajes SMALLINT NOT NULL DEFAULT 0 CHECK (garajes >= 0),
    piso SMALLINT NOT NULL DEFAULT 1 CHECK (piso >= 0),
    antiguedad_anios SMALLINT NOT NULL DEFAULT 0 CHECK (antiguedad_anios >= 0),
    vista_exterior BOOLEAN NOT NULL DEFAULT TRUE,
    
    -- Indicadores Monetarios
    precio_soles_nominal DECIMAL(14, 2) NOT NULL CHECK (precio_soles_nominal > 0),
    ipc_deflactador DECIMAL(8, 4) NOT NULL CHECK (ipc_deflactador > 0),
    precio_soles_const DECIMAL(14, 2) NOT NULL CHECK (precio_soles_const > 0), -- Target
    
    -- Partición de Modelado
    split_dataset VARCHAR(10) NOT NULL CHECK (split_dataset IN ('TRAIN', 'TEST')),
    creado_en TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

COMMENT ON TABLE dataset_inmuebles_venta IS 'Observaciones depuradas de departamentos residenciales en venta (BCRP 2016-2025).';

-- 5. TABLA DE ENTRENAMIENTO: INMUEBLES EN ALQUILER (BCRP / Portales)
CREATE TABLE IF NOT EXISTS dataset_inmuebles_alquiler (
    id BIGSERIAL PRIMARY KEY,
    distrito_id INTEGER NOT NULL REFERENCES distritos(id) ON DELETE RESTRICT,
    anio SMALLINT NOT NULL CHECK (anio BETWEEN 2016 AND 2030),
    trimestre SMALLINT NOT NULL CHECK (trimestre BETWEEN 1 AND 4),
    
    -- Características Físicas del Predio
    superficie_m2 DECIMAL(10, 2) NOT NULL CHECK (superficie_m2 > 0),
    habitaciones SMALLINT NOT NULL CHECK (habitaciones >= 0),
    banos SMALLINT NOT NULL CHECK (banos >= 0),
    garajes SMALLINT NOT NULL DEFAULT 0 CHECK (garajes >= 0),
    piso SMALLINT NOT NULL DEFAULT 1 CHECK (piso >= 0),
    antiguedad_anios SMALLINT NOT NULL DEFAULT 0 CHECK (antiguedad_anios >= 0),
    vista_exterior BOOLEAN NOT NULL DEFAULT TRUE,
    
    -- Indicadores Monetarios
    alquiler_soles_nominal DECIMAL(14, 2) NOT NULL CHECK (alquiler_soles_nominal > 0),
    ipc_deflactador DECIMAL(8, 4) NOT NULL CHECK (ipc_deflactador > 0),
    alquiler_soles_const DECIMAL(14, 2) NOT NULL CHECK (alquiler_soles_const > 0), -- Target
    
    -- Partición de Modelado
    split_dataset VARCHAR(10) NOT NULL CHECK (split_dataset IN ('TRAIN', 'TEST')),
    creado_en TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

COMMENT ON TABLE dataset_inmuebles_alquiler IS 'Observaciones depuradas de alquiler de departamentos (BCRP 2016-2025).';

-- 6. TABLA DE AUDITORÍA DE FLAGS DE IMPUTACIÓN (Para Análisis de Sensibilidad sin Data Leakage)
CREATE TABLE IF NOT EXISTS auditoria_flags_imputacion (
    id BIGSERIAL PRIMARY KEY,
    tipo_operacion VARCHAR(10) NOT NULL CHECK (tipo_operacion IN ('VENTA', 'ALQUILER')),
    inmueble_id BIGINT NOT NULL,
    distrito_id INTEGER NOT NULL REFERENCES distritos(id),
    anio SMALLINT NOT NULL,
    nse_imputado BOOLEAN NOT NULL DEFAULT FALSE,
    tasas_criminalidad_imputada BOOLEAN NOT NULL DEFAULT FALSE,
    poblacion_imputada BOOLEAN NOT NULL DEFAULT FALSE,
    split_dataset VARCHAR(10) NOT NULL CHECK (split_dataset IN ('TRAIN', 'TEST')),
    creado_en TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

COMMENT ON TABLE auditoria_flags_imputacion IS 'Persistencia desacoplada de flags de imputación para auditoría científica y test de Chow/sensibilidad.';

-- 7. TABLA DE CONTROL Y AUDITORÍA DE PIPELINE (MLOps / Pipeline Tracker)
CREATE TABLE IF NOT EXISTS pipeline_ejecuciones (
    id BIGSERIAL PRIMARY KEY,
    tipo_ejecucion VARCHAR(60) NOT NULL, -- 'ETL_INGESTA_VENTA', 'ETL_INGESTA_ALQUILER', 'ENTRENAMIENTO_XGBOOST_VENTA', etc.
    ejecutado_por VARCHAR(50) NOT NULL DEFAULT 'CLI_LOCAL',
    iniciado_en TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    finalizado_en TIMESTAMPTZ,
    duracion_segundos INTEGER,
    registros_leidos INTEGER NOT NULL DEFAULT 0,
    registros_guardados INTEGER NOT NULL DEFAULT 0,
    registros_filtrados INTEGER NOT NULL DEFAULT 0,
    errores INTEGER NOT NULL DEFAULT 0,
    metricas_salida JSONB, -- {'mape_test': 15.01, 'r2_test': 0.7469, 'cod_iaao': 14.31}
    observaciones TEXT,
    estado VARCHAR(20) NOT NULL DEFAULT 'EN_PROCESO' CHECK (estado IN ('EN_PROCESO', 'COMPLETADO', 'FALLIDO', 'ADVERTENCIA'))
);

COMMENT ON TABLE pipeline_ejecuciones IS 'Trazabilidad de corridas de ETL y entrenamiento de modelos para auditoría de MLOps.';

-- =============================================================================
-- ÍNDICES DE RENDIMIENTO PARA ALIMENTAR EL SERVICIO DE MACHINE LEARNING
-- =============================================================================

CREATE INDEX IF NOT EXISTS idx_contexto_distrito_anio ON distrito_anio_contexto (distrito_id, anio);
CREATE INDEX IF NOT EXISTS idx_venta_split_anio ON dataset_inmuebles_venta (split_dataset, anio);
CREATE INDEX IF NOT EXISTS idx_venta_distrito_anio ON dataset_inmuebles_venta (distrito_id, anio);
CREATE INDEX IF NOT EXISTS idx_alquiler_split_anio ON dataset_inmuebles_alquiler (split_dataset, anio);
CREATE INDEX IF NOT EXISTS idx_alquiler_distrito_anio ON dataset_inmuebles_alquiler (distrito_id, anio);
CREATE INDEX IF NOT EXISTS idx_pipeline_tipo_fecha ON pipeline_ejecuciones (tipo_ejecucion, iniciado_en DESC);
