# Propuesta de Diseño de Base de Datos - Sistema Inmobiliario con Tasación ML

* **Proyecto:** Sistema web con Machine Learning para optimización de valoración inmobiliaria en Lima Metropolitana.
* **Carrera:** Ingeniería de Software - UPC
* **Motor sugerido:** PostgreSQL 15+ (con extensión `pgcrypto` o `uuid-ossp` para UUIDs).

---

## 1. Módulo / Esquema: `schema_inmuebles`

### 1.1. Tabla: `inmuebles`
* **Descripción:** Propiedades inmobiliarias registradas por agentes para comercialización y valoración.
* **Naturaleza:** Transaccional (OLTP).
* **Alcance del activo:** Departamentos residenciales (alcance validado con el modelo V2). Para otras tipologías (casas/terrenos), el registro se permite comercialmente pero con flag de incompatibilidad de tasación automática.

| Atributo | Tipo de Dato | Nulable | PK / FK / Default | Descripción y Propósito |
|---|---|---|---|---|
| `id` | `UUID` | No | **PK** (`gen_random_uuid()`) | Identificador único del inmueble. |
| `agente_id` | `UUID` | No | Ref. Lógica | ID del agente propietario (servicio de usuarios/auth). Sin FK física por desacoplamiento de microservicios. |
| `distrito_id` | `INTEGER` | No | **FK** $\rightarrow$ `distritos(id)` | Identificador del distrito (relacionado con el catálogo de los 22 distritos modelados). |
| `titulo` | `VARCHAR(255)` | No | — | Título comercial (ej: *"Dpto. 3 dorm. frente a parque - Miraflores"*). |
| `descripcion` | `TEXT` | Sí | `NULL` | Memoria descriptiva y acabados del inmueble. |
| `tipo_inmueble` | `VARCHAR(30)` | No | `'DEPARTAMENTO'` | Tipo de inmueble (`DEPARTAMENTO`, `CASA`, `TERRENO`, etc.). |
| `superficie_total_m2` | `DECIMAL(10,2)` | No | `CHECK (> 0)` | Área total en metros cuadrados. Predictor numérico de mayor impacto en el modelo. |
| `superficie_techada_m2` | `DECIMAL(10,2)` | Sí | `CHECK (> 0)` | Área construida/techada útil. |
| `habitaciones` | `SMALLINT` | No | `CHECK (>= 0)` | Cantidad de dormitorios. |
| `banos` | `SMALLINT` | No | `CHECK (>= 0)` | Cantidad de cuartos de baño completos. |
| `estacionamientos` | `SMALLINT` | No | `DEFAULT 0` | Cantidad de cocheras incluidas. |
| `piso` | `SMALLINT` | Sí | `NULL` | Número de piso de la unidad (en casas/terrenos es `NULL`). |
| `antiguedad_anios` | `SMALLINT` | Sí | `CHECK (>= 0)` | Años de antigüedad (0 para inmuebles de estreno). |
| `es_estreno` | `BOOLEAN` | No | `DEFAULT FALSE` | Flag rápido comercial para inmuebles a estrenar. |
| `vista_exterior` | `BOOLEAN` | No | `DEFAULT TRUE` | **Requerido por ML:** `TRUE` si tiene vista a calle/parque, `FALSE` si es interior/pozo de luz. |
| `estado_comercial` | `VARCHAR(20)` | No | `'DISPONIBLE'` | Estado de comercialización: `'DISPONIBLE'`, `'RESERVADO'`, `'VENDIDO'`, `'INACTIVO'`. |
| `moneda` | `VARCHAR(3)` | No | `'USD'` | Código ISO de la moneda fijada por el agente (`'USD'` o `'PEN'`). |
| `precio_publicado` | `DECIMAL(12,2)` | Sí | `CHECK (> 0)` | Precio de lista fijado por el agente en la moneda original (`moneda`). |
| `tipo_cambio_ref` | `DECIMAL(6,4)` | Sí | `NULL` | Tipo de cambio (USD $\rightarrow$ PEN) vigente al momento de publicar/actualizar el precio. |
| `precio_pen_equiv` | `DECIMAL(12,2)` | Sí | `NULL` | **Precio normalizado en Soles:** Facilita filtros, ordenamientos y comparación directa con la tasación de la IA. |
| `latitud` | `DECIMAL(10,8)` | Sí | `NULL` | Coordenada GPS latitud. Usado por el backend para calcular distancias espaciales (km) a equipamiento urbano. |
| `longitud` | `DECIMAL(11,8)` | Sí | `NULL` | Coordenada GPS longitud. Usado por el backend para calcular distancias espaciales (km) a equipamiento urbano. |
| `direccion` | `VARCHAR(500)` | Sí | `NULL` | Dirección de referencia o texto de ubicación. |
| `created_at` | `TIMESTAMPTZ` | No | `CURRENT_TIMESTAMP` | Fecha de creación del registro. |
| `updated_at` | `TIMESTAMPTZ` | No | `CURRENT_TIMESTAMP` | Fecha de última actualización. |
| `deleted_at` | `TIMESTAMPTZ` | Sí | `NULL` | Soft delete para conservar integridad analítica e histórica. |

---

### 1.2. Índices recomendados para optimización

```sql
-- Índice para filtros de búsqueda comercial por distrito, estado y rango de precio normalizado
CREATE INDEX idx_inmuebles_busqueda 
ON inmuebles (distrito_id, estado_comercial, precio_pen_equiv)
WHERE deleted_at IS NULL;

-- Índice para consultas del panel del agente
CREATE INDEX idx_inmuebles_agente 
ON inmuebles (agente_id)
WHERE deleted_at IS NULL;
```

---

## 2. Módulo / Esquema: `schema_predicciones`

### 2.1. Tabla: `predicciones`
* **Descripción:** Historial inmutable de las inferencias realizadas por el modelo de Machine Learning (XGBoost). Implementa un patrón **Snapshot / Trazabilidad inmutable** que registra las variables de entrada enviadas, las variables de entorno inyectadas, el resultado monetario en ambas divisas, los factores macroeconómicos aplicados (IPC y Tipo de Cambio) y los valores SHAP para interpretabilidad gráfica.
* **Naturaleza:** Transaccional con patrón Snapshot / Auditoría inmutable.

| Atributo | Tipo de Dato | Nulable | PK / FK / Default | Descripción y Propósito |
|---|---|---|---|---|
| `id` | `UUID` | No | **PK** (`gen_random_uuid()`) | Identificador único de la tasación / predicción. |
| `agente_id` | `UUID` | No | Ref. Lógica | Agente que solicitó la tasación (`schema_usuarios.usuarios.id`). |
| `inmueble_id` | `UUID` | Sí | Ref. Lógica | Inmueble asociado (`schema_inmuebles.inmuebles.id`). Si es cotización rápida sin guardar es `NULL`. |
| `tipo_prediccion` | `VARCHAR(20)` | No | `'VENTA'` | Modalidad tasada: `'VENTA'` (vigente en modelo M3) o `'ALQUILER'` (futura expansión). |
| `distrito_nombre` | `VARCHAR(100)` | No | — | Snapshot del nombre del distrito evaluado al momento de la tasación. |
| `superficie_m2` | `DECIMAL(10,2)` | No | — | Superficie en m² evaluada. |
| `habitaciones` | `SMALLINT` | No | — | Dormitorios evaluados. |
| `banos` | `SMALLINT` | No | — | Baños evaluados. |
| `estacionamientos` | `SMALLINT` | No | `DEFAULT 0` | Cocheras evaluadas. |
| `piso` | `SMALLINT` | Sí | `NULL` | Piso evaluado. |
| `antiguedad_anios` | `SMALLINT` | Sí | `NULL` | Antigüedad evaluada (0 para estreno). |
| `vista_exterior` | `BOOLEAN` | No | `DEFAULT TRUE` | **Requerido por ML:** Snapshot de la orientación (exterior / interior). |
| `features_contexto` | `JSONB` | Sí | `NULL` | Snapshot de variables de entorno inyectadas automáticamente (NSE, delincuencia PNP, distancias OSM). |
| `valor_estimado_pen` | `DECIMAL(12,2)` | No | — | **Monto predicho en Soles nominales:** Salida nativa del modelo tras reconversión con IPC. |
| `valor_estimado_usd` | `DECIMAL(12,2)` | No | — | **Monto predicho en Dólares:** Conversión al tipo de cambio para fines comerciales del agente. |
| `rango_min_usd` | `DECIMAL(12,2)` | No | — | Límite inferior sugerido (intervalo de negociación según MAPE). |
| `rango_max_usd` | `DECIMAL(12,2)` | No | — | Límite superior sugerido (intervalo de negociación según MAPE). |
| `factor_ipc_usado` | `DECIMAL(8,4)` | No | `1.6918` | Factor de reconversión IPC aplicado (Q1-2026 = 1.6918). Garantiza auditoría y reproducibilidad científica. |
| `tipo_cambio_usado` | `DECIMAL(6,4)` | No | — | Tipo de cambio USD/PEN aplicado en el momento de la inferencia. |
| `version_modelo` | `VARCHAR(50)` | No | — | Versión del modelo XGBoost registrada (ej: `"v2.0.0-bcrp"`). |
| `mape_referencial` | `DECIMAL(5,2)` | No | `15.04` | Error porcentual absoluto medio histórico del modelo para soporte de confianza. |
| `shap_base_value` | `DECIMAL(12,2)` | No | — | Valor base / esperado del modelo necesario para inicializar el gráfico Waterfall en frontend. |
| `shap_values` | `JSONB` | No | — | Diccionario con las contribuciones marginales de cada feature para la gráfica explicativa. |
| `tiempo_inferencia_ms`| `INTEGER` | Sí | `NULL` | Métrica de latencia y monitoreo MLOps del servicio de inferencia. |
| `created_at` | `TIMESTAMPTZ` | No | `CURRENT_TIMESTAMP` | Momento exacto de emisión de la tasación. |

---

### 2.2. Índices recomendados para optimización

```sql
-- Historial cronológico de tasaciones emitidas por un agente
CREATE INDEX idx_predicciones_agente 
ON predicciones (agente_id, created_at DESC);

-- Histórico de tasaciones vinculadas a un inmueble específico
CREATE INDEX idx_predicciones_inmueble 
ON predicciones (inmueble_id, created_at DESC)
WHERE inmueble_id IS NOT NULL;
```

### 3. Módulo / Esquema: `schema_rentabilidad`

En este caso es imporante considerar lo siguente, ya que como tal no tenemos a todos los datos correspondiente para menjar el CAPRATE y ROI exactos, por eso haremos lo siguente (Imporatnte mensionar que si o si se implementara un modulo de prediccion para alquiler, adicional al de venta)

     [3.1]El proyecto aporta un sistema de valoración automatizada adaptado a un mercado con baja transparencia informacional, que combina tres elementos que en la literatura revisada aparecen habitualmente por separado: un modelo predictivo basado en técnicas de boosting de árboles de decisión (XGBoost), una capa de explicabilidad orientada al usuario no técnico, y un módulo financiero que traduce la predicción en indicadores de decisión comercial. Los trabajos precedentes en el contexto peruano se han concentrado en la valorización con fines bancarios o en la predicción económica agregada, sin abordar la operación comercial de la agencia independiente. Para efectos de este proyecto, se entrenan dos modelos XGBoost independientes entre sí —uno sobre el mercado de venta y otro sobre el mercado de alquiler—, cuyas salidas se combinan externamente mediante un módulo de cálculo financiero para estimar el Ingreso Operativo Neto (NOI), aplicando un ratio de gastos operativos (OER) estándar y parametrizable, dado que no se dispone de datos granulares de gasto real por predio ni de registros pareados de venta-alquiler para un mismo inmueble. El Cap Rate se calcula como el cociente entre el NOI anual estimado y el valor comercial predicho del inmueble, mientras que el ROI se calcula bajo el supuesto de ausencia de apalancamiento financiero (unlevered), por lo que ambos indicadores convergen conceptualmente bajo los supuestos del proyecto. Se declara como limitación explícita que ninguno de los dos indicadores incorpora financiamiento hipotecario, tasa de vacancia real ni apreciación de capital a largo plazo.

---

## 4. Módulo / Esquema: `schema_dataset` (Feature Store Distrital)

### 4.1. Tabla: `metricas_distritales`
* **Descripción:** Tabla dimensional y Feature Store analítico con indicadores demográficos, económicos y distancias espaciales consolidadas para los 22 distritos de Lima Metropolitana modelados. Alimenta en tiempo real al pipeline de inferencia de Machine Learning cuando se evalúa un departamento.
* **Fuentes Oficiales:** INEI (Censos y ENAHO para estratos socioeconómicos NSE), Policía Nacional del Perú (PNP - Denuncias de Seguridad Ciudadana) y OpenStreetMap (OSM Overpass API para distancias geodésicas a equipamiento urbano).
* **Naturaleza:** Dimensional analítica / Feature Store distrital en producción.

| Atributo | Tipo de Dato | Nulable | PK / FK / Default | Descripción y Propósito |
|---|---|---|---|---|
| `id` | `INTEGER` | No | **PK** (`SERIAL` o `IDENTITY`) | Identificador único del registro. |
| `distrito_id` | `INTEGER` | No | **FK** $\rightarrow$ `distritos(id)`, `UNIQUE` | Clave foránea al catálogo maestro de distritos. |
| `distrito_nombre` | `VARCHAR(100)` | No | — | Nombre oficial del distrito (ej: *"San Borja"*, *"Surco"*). |
| `distrito_encoded` | `DECIMAL(12,2)`| No | — | **Feature ML:** Target Encoding bayesiano distrital calculado durante el entrenamiento del modelo. |
| `pct_nse_a` | `DECIMAL(5,2)` | No | `CHECK (>= 0)` | % población en Estrato Socioeconómico A (Fuente: ENAHO / INEI). |
| `pct_nse_b` | `DECIMAL(5,2)` | No | `CHECK (>= 0)` | % población en Estrato Socioeconómico B. |
| `pct_nse_c` | `DECIMAL(5,2)` | No | `CHECK (>= 0)` | % población en Estrato Socioeconómico C. |
| `pct_nse_d` | `DECIMAL(5,2)` | No | `CHECK (>= 0)` | % población en Estrato Socioeconómico D. |
| `pct_nse_e` | `DECIMAL(5,2)` | No | `CHECK (>= 0)` | % población en Estrato Socioeconómico E. |
| `tasa_robo` | `DECIMAL(8,4)` | No | `CHECK (>= 0)` | Tasa de denuncias por robo por cada 10,000 hab. (Fuente: PNP). |
| `tasa_hurto` | `DECIMAL(8,4)` | No | `CHECK (>= 0)` | Tasa de denuncias por hurto por cada 10,000 hab. (Fuente: PNP). |
| `poblacion_proyectada`| `INTEGER` | No | `CHECK (> 0)` | Población distrital estimada según proyecciones INEI. |
| `area_distrito_km2` | `DECIMAL(10,4)`| No | `CHECK (> 0)` | Extensión territorial del distrito en $km^2$. |
| `densidad_hab_km2` | `DECIMAL(10,2)`| No | `CHECK (> 0)` | Densidad poblacional en hab/$km^2$ (`poblacion / area`). |
| `distancia_centro_km`| `DECIMAL(8,4)` | No | `CHECK (>= 0)` | Distancia referencial en km al Centro de Lima. |
| `dist_colegio_km` | `DECIMAL(8,4)` | No | `CHECK (>= 0)` | Distancia mediana en km a la institución educativa más cercana (OSM). |
| `dist_hospital_km` | `DECIMAL(8,4)` | No | `CHECK (>= 0)` | Distancia mediana en km al centro de salud / clínica más cercana (OSM). |
| `dist_estacion_transporte_km`| `DECIMAL(8,4)`| No | `CHECK (>= 0)`| Distancia en km a estaciones masivas de transporte (Metropolitano / Metro L1) (OSM). |
| `dist_centro_comercial_km`| `DECIMAL(8,4)`| No | `CHECK (>= 0)`| Distancia mediana en km a centros comerciales y malls (OSM). |
| `dist_parque_km` | `DECIMAL(8,4)` | No | `CHECK (>= 0)` | Distancia mediana en km al parque o área recreativa más cercana (OSM). |
| `dist_universidad_km`| `DECIMAL(8,4)` | No | `CHECK (>= 0)` | Distancia mediana en km a sedes universitarias (OSM). |
| `actualizado_en` | `TIMESTAMPTZ` | No | `CURRENT_TIMESTAMP` | Fecha de carga o calibración periódica de los indicadores. |

---

### 4.2. Índices recomendados para optimización

```sql
-- Acceso instantáneo por distrito al momento de armar el vector de inferencia
CREATE UNIQUE INDEX idx_metricas_distrito_id 
ON metricas_distritales (distrito_id);
```

---

### 4.3. Tabla: `dataset_inmuebles`
* **Descripción:** **Data Mart Analítico Consolidado (Dataset Maestro de Entrenamiento).** Almacena el histórico completo de ofertas inmobiliarias transaccionales recolectadas (BCRP 2016-2025, convenios de agencias y portales formales) desnormalizado con las variables distritales y macroeconómicas (IPC y Tipo de Cambio) correspondientes a la fecha del registro.
* **Naturaleza:** Analítica (OLAP / Data Mart para algoritmos de Machine Learning).

| Atributo | Tipo de Dato | Nulable | PK / FK / Default | Descripción y Propósito |
|---|---|---|---|---|
| `id` | `BIGSERIAL` | No | **PK** | Identificador secuencial del registro. |
| `fuente` | `VARCHAR(50)` | No | — | Origen del dato: `'BCRP'`, `'CONVENIO_AGENCIAS'`, `'PORTAL_INMOBILIARIO'`, `'MANUAL'`. |
| `id_anuncio_fuente` | `VARCHAR(100)` | Sí | — | Identificador del registro en la fuente de origen. |
| `fecha_publicacion` | `DATE` | No | — | Fecha de registro de la oferta. |
| `anio` | `SMALLINT` | No | — | **Feature ML:** Año de la oferta (2016-2025). |
| `trimestre` | `SMALLINT` | No | — | **Feature ML:** Trimestre del año (1 a 4). |
| `tipo_operacion` | `VARCHAR(20)` | No | `'VENTA'` | Modalidad de mercado: `'VENTA'` o `'ALQUILER'`. |
| `distrito_id` | `INTEGER` | No | **FK** $\rightarrow$ `distritos(id)` | Clave foránea al catálogo de distritos. |
| `distrito_nombre` | `VARCHAR(100)` | No | — | Nombre del distrito (copia para lectura analítica). |
| `distrito_encoded` | `DECIMAL(12,2)`| No | — | **Feature ML:** Target Encoding bayesiano distrital. |
| `superficie_m2` | `DECIMAL(10,2)` | No | `CHECK (> 0)` | **Feature ML:** Área total en $m^2$. |
| `habitaciones` | `SMALLINT` | No | `CHECK (>= 0)`| **Feature ML:** Dormitorios. |
| `banos` | `SMALLINT` | No | `CHECK (>= 0)`| **Feature ML:** Baños completos. |
| `estacionamientos` | `SMALLINT` | No | `DEFAULT 0` | **Feature ML:** Cocheras. |
| `piso` | `SMALLINT` | No | `DEFAULT 1` | **Feature ML:** Piso del departamento. |
| `antiguedad_anios` | `SMALLINT` | No | `DEFAULT 0` | **Feature ML:** Años de antigüedad (0 si es estreno). |
| `vista_exterior` | `BOOLEAN` | No | `DEFAULT TRUE`| **Feature ML:** Orientación exterior/interior. |
| `precio_original` | `DECIMAL(12,2)` | No | — | Monto original publicado en el anuncio. |
| `moneda_original` | `VARCHAR(3)` | No | `'USD'` | Divisa original: `'USD'` o `'PEN'`. |
| `tipo_cambio` | `DECIMAL(6,4)` | No | — | Tipo de cambio oficial BCRP a la fecha de publicación. |
| `precio_soles_nominal`| `DECIMAL(12,2)`| No | — | Precio convertido a Soles corrientes de esa fecha. |
| `ipc` | `DECIMAL(8,4)` | No | — | Índice de Precios al Consumidor (BCRP) a la fecha de publicación. |
| **`precio_soles_const`**| `DECIMAL(12,2)`| No | — | **TARGET DEL MODELO:** Precio en Soles constantes deflactados (Base 2009=100). |
| `pct_nse_a` a `pct_nse_e`| `DECIMAL(5,2)`| No | — | **Features ML:** Distribución porcentual NSE (ENAHO/INEI). |
| `tasa_robo` / `tasa_hurto`| `DECIMAL(8,4)`| No | — | **Features ML:** Tasas de delincuencia por cada 10,000 hab. (PNP). |
| `poblacion_proyectada` | `INTEGER` | No | — | **Feature ML:** Población distrital (INEI). |
| `area_distrito_km2` | `DECIMAL(10,4)`| No | — | **Feature ML:** Extensión en $km^2$. |
| `densidad_hab_km2` | `DECIMAL(10,2)`| No | — | **Feature ML:** Densidad hab/$km^2$. |
| `distancia_centro_km` | `DECIMAL(8,4)` | No | — | **Feature ML:** Distancia al Centro de Lima. |
| `dist_colegio_km` | `DECIMAL(8,4)` | No | — | **Feature ML:** Distancia al colegio más cercano. |
| `dist_hospital_km` | `DECIMAL(8,4)` | No | — | **Feature ML:** Distancia al hospital más cercano. |
| `dist_estacion_transporte_km`|`DECIMAL(8,4)`|No| — | **Feature ML:** Distancia a Metropolitano/Metro. |
| `dist_centro_comercial_km`| `DECIMAL(8,4)`| No | — | **Feature ML:** Distancia a centro comercial. |
| `dist_parque_km` | `DECIMAL(8,4)` | No | — | **Feature ML:** Distancia a parque más cercano. |
| `dist_universidad_km`| `DECIMAL(8,4)` | No | — | **Feature ML:** Distancia a universidad más cercana. |
| `es_duplicado` | `BOOLEAN` | No | `DEFAULT FALSE`| Flag de detección de ofertas repetidas. |
| `es_outlier` | `BOOLEAN` | No | `DEFAULT FALSE`| Flag de exclusión por rango intercuartílico (IQR). |
| `incluido_en_entrenamiento`|`BOOLEAN`| No | `DEFAULT TRUE` | Flag general de inclusión para el pipeline de ML. |
| **`split_dataset`** | `VARCHAR(10)` | No | `'TRAIN'` | Partición de modelado: `'TRAIN'` (2016-2023) o `'TEST'` (2024-2025). |
| `created_at` | `TIMESTAMPTZ` | No | `CURRENT_TIMESTAMP` | Fecha de carga del registro al Data Mart. |

---

### 4.4. Índices recomendados para optimización

```sql
-- Extracción óptima del dataset de entrenamiento / reentrenamiento
CREATE INDEX idx_dataset_entrenamiento 
ON dataset_inmuebles (tipo_operacion, split_dataset)
WHERE incluido_en_entrenamiento = TRUE AND es_outlier = FALSE;

-- Búsquedas analíticas por distrito y rango temporal
CREATE INDEX idx_dataset_distrito_periodo 
ON dataset_inmuebles (distrito_id, anio, trimestre);
```

---

### 4.5. Tabla: `pipeline_ejecuciones`
* **Descripción:** Log operativo y de auditoría del orquestador del Pipeline de Datos y MLOps. Registra el volumen procesado, duración, anomalías y rendimiento métrico de cada corrida de ingesta, enriquecimiento urbano o reentrenamiento de modelos.
* **Naturaleza:** Log operativo / Auditoría MLOps.

| Atributo | Tipo de Dato | Nulable | PK / FK / Default | Descripción y Propósito |
|---|---|---|---|---|
| `id` | `BIGSERIAL` | No | **PK** | Identificador único de la ejecución. |
| `tipo_ejecucion` | `VARCHAR(50)` | No | — | Tipo de trabajo: `'INGESTA_BCRP'`, `'CARGA_AGENCIAS'`, `'ENRIQUECIMIENTO_DISTRITAL'`, `'ACTUALIZACION_OSM'`, `'REENTRENAMIENTO_ML'`. |
| `ejecutado_por` | `VARCHAR(50)` | No | `'CRON_AUTOMATICO'` | Origen del disparo: `'CRON_AUTOMATICO'`, `'ADMIN_MANUAL'`, `'PIPELINE_CI_CD'`. |
| `iniciado_en` | `TIMESTAMPTZ` | No | `CURRENT_TIMESTAMP` | Momento exacto de inicio de la tarea. |
| `finalizado_en` | `TIMESTAMPTZ` | Sí | `NULL` | Momento exacto de culminación. |
| `duracion_segundos` | `INTEGER` | Sí | `NULL` | Tiempo total transcurrido en segundos. |
| `registros_procesados`| `INTEGER` | No | `DEFAULT 0` | Total de registros evaluados en el lote. |
| `registros_nuevos` | `INTEGER` | No | `DEFAULT 0` | Cantidad de ofertas nuevas incorporadas al Data Mart. |
| `registros_duplicados`| `INTEGER` | No | `DEFAULT 0` | Cantidad de ofertas repetidas descartadas. |
| `registros_outliers` | `INTEGER` | No | `DEFAULT 0` | Cantidad de ofertas descartadas por límites IQR de precio o superficie. |
| `errores` | `INTEGER` | No | `DEFAULT 0` | Conteo de incidencias o fallas durante el proceso. |
| `metricas_salida` | `JSONB` | Sí | `NULL` | Métricas generadas (ej: si fue reentrenamiento: `{"mape": 14.88, "r2": 0.7473, "train_rows": 44173}`). |
| `observaciones` | `TEXT` | Sí | `NULL` | Resumen cualitativo, advertencias o traza de error en caso de fallo. |
| `estado` | `VARCHAR(20)` | No | `'EN_PROCESO'` | Estado del job: `'EN_PROCESO'`, `'COMPLETADO'`, `'ADVERTENCIA'`, `'FALLIDO'`. |

---

### 4.6. Índices recomendados para optimización

```sql
-- Consultar las últimas corridas por tipo de proceso
CREATE INDEX idx_pipeline_tipo_fecha 
ON pipeline_ejecuciones (tipo_ejecucion, iniciado_en DESC);

-- Monitorear fallos y alertas operativas
CREATE INDEX idx_pipeline_fallos 
ON pipeline_ejecuciones (estado)
WHERE estado = 'FALLIDO';
```