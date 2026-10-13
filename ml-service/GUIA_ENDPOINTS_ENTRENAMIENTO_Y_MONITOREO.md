# Guía de Integración Backend: Endpoints de Reentrenamiento, Monitoreo y Gobernanza de Modelos ML

**Documento:** Especificación Técnica de APIs para el Módulo de Monitoreo  
**Microservicio:** `ml-service` (FastAPI + XGBoost + MLflow Tracking & Registry)  
**Proyecto:** Valuación Inmobiliaria Masiva en Lima Metropolitana  
**Fecha:** Octubre 2026  
**Versión de API:** `v1`  

---

## 1. Arquitectura de Integración y Principios de Diseño

El sistema separa las responsabilidades entre el **Backend Principal** (NestJS / Node / Spring / Django) y el **Microservicio ML** (`ml-service`):

```
┌─────────────────────────┐          HTTP REST          ┌───────────────────────────┐
│    Backend Principal    │  ─────────────────────────> │        ml-service         │
│  (Módulo de Monitoreo)  │                             │         (FastAPI)         │
│                         │  <───────────────────────── │                           │
└───────────┬─────────────┘      (Polling / Respuestas) └─────────────┬─────────────┘
            │                                                         │
            ▼                                                         ▼
┌─────────────────────────┐                             ┌───────────────────────────┐
│ PostgreSQL Operacional  │                             │ MLflow Tracking & Registry│
│ (db_operacional_valuo)  │                             │  (localhost:5000 / mlops) │
│ - Auditoría de jobs     │                             │ - Versiones de modelos    │
│ - Historial promociones │                             │ - Métricas (MAPE, IAAO)   │
└─────────────────────────┘                             └───────────────────────────┘
```

### Principios Clave:
1. **Asincronía y No-Bloqueo:** Los pipelines de entrenamiento procesan >40,000 registros y calculan SHAP values, tomando entre 25 y 45 segundos. Por ello, el inicio de entrenamiento retorna inmediatamente **HTTP 202 (Accepted)** con un `job_id`. El backend realiza polling sobre `/estado/{job_id}`.
2. **Control de Concurrencia (Semáforo):** Solo se permite **un (1) job de entrenamiento simultáneo**. Si se intenta lanzar un segundo entrenamiento mientras uno está en curso, la API responde **HTTP 409 Conflict**.
3. **Gobernanza Human-in-the-Loop:** Por defecto, los nuevos modelos se registran en MLflow como **`Candidate`** para que el administrador examine sus métricas (MAPE, $R^2$, COD IAAO) en el panel de monitoreo. Si se aprueba, el backend invoca el endpoint de promoción (`/promover`), ejecutando **Hot-Reload en memoria RAM** sin tiempo de inactividad (*zero-downtime*).
4. **Seguridad:** Todas las rutas administrativas (`/api/v1/admin/*`) requieren el header `X-Admin-Token` si la variable de entorno `ADMIN_TOKEN` está configurada en `.env`.

---

## 2. Convenciones Generales de la API

- **Base URL:** `http://localhost:8000` (configurable por variable de entorno `ML_SERVICE_URL`)
- **Headers Estándar:**
  ```http
  Content-Type: application/json
  X-Admin-Token: <ADMIN_TOKEN_CONFIGURADO>
  ```
- **Códigos de Estado HTTP:**
  - `200 OK`: Consulta o promoción exitosa.
  - `202 Accepted`: Job de entrenamiento iniciado exitosamente en background.
  - `401 Unauthorized`: Token de administrador ausente o inválido.
  - `404 Not Found`: `job_id` o modelo no encontrado.
  - `409 Conflict`: Ya existe un job de entrenamiento en ejecución.
  - `422 Unprocessable Entity`: Error de validación de payload JSON.
  - `503 Service Unavailable`: Modelos no inicializados en memoria.

---

## 3. Módulo 1: Reentrenamiento Asíncrono de Modelos

Prefijo: `/api/v1/admin/entrenamiento`

### 3.0. Especificación de Parámetros de Reentrenamiento (Obligatorios vs Opcionales)

Para lanzar el reentrenamiento, **ningún campo es obligatorio**. El microservicio ya incorpora los hiperparámetros óptimos producto de la calibración de la tesis. Puedes enviar un body vacío `{}` o personalizar cualquiera de los siguientes atributos:

| Campo | Tipo | ¿Obligatorio? | Default (Venta) | Default (Alquiler) | Descripción y Propósito |
| :--- | :--- | :---: | :---: | :---: | :--- |
| `nombre_modelo` | `string` | **Opcional** | `"xgboost_venta_v2"` | `"xgboost_alquiler_v1"` | Prefijo para nombrar el modelo y los archivos `.pkl` / `.json`. |
| `max_depth` | `int` | **Opcional** | `6` | `6` | Profundidad máxima de los árboles (evita sobreajuste). |
| `learning_rate` | `float` | **Opcional** | `0.040` | `0.035` | Tasa de aprendizaje o contracción por árbol. |
| `n_estimators` | `int` | **Opcional** | `650` | `600` | Número total de estimadores (árboles). |
| `min_child_weight` | `int` | **Opcional** | `9` | `4` | Peso mínimo requerido en un nodo hoja. |
| `usar_target_m2` | `bool` | **Opcional** | `true` | `true` | Si es `true`, entrena sobre $\ln(\text{precio}/m^2)$ (reduce sesgos en áreas dispares). |
| `factor_calibracion`| `float` | **Opcional** | `null` (Automático) | `null` (Automático) | Multiplicador de escala. Si es `null`, calcula calibración estratificada IAAO. |
| `guardar_como_activo`| `bool` | **Opcional** | `false` | `false` | **Gobernanza Human-in-the-Loop:** Si es `false`, se registra como `Candidate` en MLflow para ser comparado antes de promoverlo. Si es `true`, pasa a producción de inmediato. |
| `shap_top_n` | `int` | **Opcional** | `10` | `10` | Cantidad de atributos clave a precargar en el explicador de SHAP. |
| `hiperparametros` | `object` | **Opcional** | Ver ejemplo | Ver ejemplo | Diccionario con regularizaciones finas (`subsample`, `colsample_bytree`, `reg_alpha`, `reg_lambda`). |

> **💡 Compatibilidad con el Backend de Monitoreo (`valuo-monitoring-service`):**  
> Si tu backend en Spring Boot maneja un payload simplificado como:  
> ```json
> {
>   "operation_type": "VENTA",
>   "dataset_version": "v2026_Q1",
>   "target_metric": "MAPE"
> }
> ```
> Simplemente dirígelo a la URL correspondiente (`/venta` o `/alquiler`). Hacia `ml-service` puedes enviar `{}` o reenviar únicamente los hiperparámetros que el administrador haya decidido alterar desde el panel de control.

---

### 3.1. Iniciar Reentrenamiento de Venta
- **Método / Ruta:** `POST /api/v1/admin/entrenamiento/venta`
- **Descripción:** Lanza en segundo plano el pipeline de entrenamiento del modelo de venta sobre `dataset_inmuebles_venta` (PostgreSQL), aplicando target unitario $\ln(P/m^2)$, calibración estratificada IAAO y registro en MLflow.
- **Request Body (Opcional — todos los campos tienen defaults óptimos validados):**
```json
{
  "nombre_modelo": "xgboost_venta_v2",
  "max_depth": 6,
  "learning_rate": 0.040,
  "n_estimators": 650,
  "min_child_weight": 9,
  "usar_target_m2": true,
  "factor_calibracion": null,
  "guardar_como_activo": false,
  "shap_top_n": 10,
  "hiperparametros": {
    "subsample": 0.852,
    "colsample_bytree": 0.847,
    "reg_alpha": 4.06,
    "reg_lambda": 0.0306
  }
}
```
> **Nota sobre `guardar_como_activo`:**
> - `false` (Recomendado): Se registra como **Candidate** en MLflow para ser comparado en el panel de benchmark antes de promoverse.
> - `true`: Pasa directamente a producción y reemplaza al modelo activo en RAM al terminar.

- **Respuesta (HTTP 202 Accepted):**
```json
{
  "job_id": "7dea5252-8f22-4f10-9330-80eb14dd1f16",
  "tipo_operacion": "venta",
  "estado": "en_progreso",
  "nombre_modelo": "xgboost_venta_v2",
  "iniciado_en": "2026-10-06T04:39:41.123456Z",
  "mensaje": "Job de reentrenamiento de venta 'xgboost_venta_v2' iniciado. El pipeline corre en background y registrará el modelo en MLflow.",
  "consultar_estado_en": "/api/v1/admin/entrenamiento/estado/7dea5252-8f22-4f10-9330-80eb14dd1f16"
}
```

---

### 3.2. Iniciar Reentrenamiento de Alquiler
- **Método / Ruta:** `POST /api/v1/admin/entrenamiento/alquiler`
- **Descripción:** Lanza en segundo plano el pipeline de entrenamiento del modelo de alquiler sobre `dataset_inmuebles_alquiler` con ponderación temporal `decay = 0.85`.
- **Request Body (Opcional):**
```json
{
  "nombre_modelo": "xgboost_alquiler_v1",
  "max_depth": 6,
  "learning_rate": 0.035,
  "n_estimators": 600,
  "min_child_weight": 4,
  "usar_target_m2": true,
  "factor_calibracion": null,
  "guardar_como_activo": false,
  "shap_top_n": 10,
  "hiperparametros": {
    "subsample": 0.85,
    "colsample_bytree": 0.80,
    "reg_alpha": 0.1,
    "reg_lambda": 4.0
  }
}
```

- **Respuesta (HTTP 202 Accepted):** Idéntica estructura a Venta con `tipo_operacion: "alquiler"`.

---

### 3.3. Consultar Estado y Métricas de un Job (Polling)
- **Método / Ruta:** `GET /api/v1/admin/entrenamiento/estado/{job_id}`
- **Descripción:** Consulta el estado actual de un job en ejecución o finalizado.
- **Respuesta durante ejecución (`en_progreso`):**
```json
{
  "job_id": "7dea5252-8f22-4f10-9330-80eb14dd1f16",
  "tipo_operacion": "venta",
  "estado": "en_progreso",
  "nombre_modelo": "xgboost_venta_v2",
  "iniciado_en": "2026-10-06T04:39:41.123456Z",
  "completado_en": null,
  "duracion_segundos": null,
  "progreso": "[5/8] Entrenando XGBRegressor (venta, n_estimators=650)...",
  "metricas": null,
  "artefactos": null,
  "mlflow_run_id": null,
  "mlflow_model_version": null,
  "estado_gobernanza": "candidate",
  "error": null
}
```

- **Respuesta al finalizar (`completado`):**
```json
{
  "job_id": "7dea5252-8f22-4f10-9330-80eb14dd1f16",
  "tipo_operacion": "venta",
  "estado": "completado",
  "nombre_modelo": "xgboost_venta_v2",
  "iniciado_en": "2026-10-06T04:39:41.123456Z",
  "completado_en": "2026-10-06T04:40:15.443456Z",
  "duracion_segundos": 34.32,
  "progreso": "[8/8] Entrenamiento de venta completado y registrado como candidato.",
  "metricas": {
    "mape_pct": 14.1620,
    "r2": 0.7472,
    "mae": 51230.15,
    "rmse": 77650.40,
    "cod_pct": 14.12,
    "cov_pct": 18.35,
    "prd": 1.0084,
    "prb": -0.0125,
    "median_ratio_calibrado": 1.0000,
    "n_train": 44173,
    "n_test": 23737,
    "supera_benchmark": true,
    "iaao": {
      "median_ratio": 1.0000,
      "mean_ratio": 1.0142,
      "weighted_mean_ratio": 1.0057,
      "cod_pct": 14.12,
      "cov_pct": 18.35,
      "prd": 1.0084,
      "prb": -0.0125,
      "cumple_iaao_cod": true,
      "cumple_iaao_nivel": true,
      "cumple_iaao_prd": true
    }
  },
  "artefactos": {
    "modelo_pkl": "models/xgboost_venta_v2.pkl",
    "params_json": "models/xgboost_venta_v2_params.json",
    "metricas_json": "models/xgboost_venta_v2_metricas.json",
    "model_config_json": "config/model_config.json",
    "modelo_recargado_en_memoria": false,
    "mlflow_model_uri": "models:/xgboost_venta/8"
  },
  "mlflow_run_id": "0ac890d6d1ca45a8b77b977eb96a0cff",
  "mlflow_model_version": "8",
  "estado_gobernanza": "candidate",
  "error": null
}
```

---

### 3.4. Consultar Último Job y Disponibilidad
- **Método / Ruta:** `GET /api/v1/admin/entrenamiento/estado`
- **Descripción:** Permite saber rápidamente si el servicio está ocupado entrenando o si está disponible para un nuevo job.

### 3.5. Listar Historial de Jobs de la Sesión
- **Método / Ruta:** `GET /api/v1/admin/entrenamiento/jobs`
- **Descripción:** Retorna el array de todos los jobs lanzados en la sesión actual de FastAPI.

---

## 4. Módulo 2: Monitoreo, Gobernanza y Benchmarking

Prefijo: `/api/v1/admin/modelos`

### 4.1. Benchmark Comparativo: Producción vs Candidatos
- **Método / Ruta:** `GET /api/v1/admin/modelos/benchmark/{tipo_operacion}`
- **Parámetros Path:** `tipo_operacion` (`venta` o `alquiler`)
- **Query Params (Opcionales):**
  - `solo_ultimo`: `boolean` (default: `true`). Compara únicamente el candidato activo más reciente frente al modelo en Producción (comparativa limpia 1 a 1).
  - `version`: `string` (ej: `"8"`). Permite forzar la comparativa de una versión específica frente a Producción.
- **Descripción:** Compara matemáticamente el modelo actualmente activo en memoria RAM (`Production`) contra el modelo candidato generado por el reentrenamiento en MLflow. Excluye automáticamente versiones históricas en estado `Archived`.
- **Respuesta (HTTP 200 OK):**
```json
{
  "tipo_operacion": "venta",
  "benchmark_referencial_mape": 15.0,
  "modelo_produccion": {
    "nombre_modelo": "xgboost_venta",
    "version": "7",
    "stage": "Production",
    "run_id": "0ac890d6d1ca45a8b77b977eb96a0cff",
    "metricas": {
      "mape_pct": 14.9845,
      "r2": 0.7473,
      "mae": 52121.38,
      "rmse": 78118.43,
      "n_train": 44173,
      "n_test": 23737
    },
    "tags": {
      "autor": "Joaquín Cortez",
      "target": "precio_soles_const"
    },
    "creado_en": "2026-10-06T04:40:00Z",
    "fuente_artefacto": "models:/xgboost_venta/7"
  },
  "candidatos": [
    {
      "candidato": {
        "nombre_modelo": "xgboost_venta",
        "version": "8",
        "stage": "Candidate",
        "run_id": "b1c2d3e4f5a6b7c8d9e0",
        "metricas": {
          "mape_pct": 14.1571,
          "r2": 0.7472,
          "mae": 50897.93,
          "rmse": 78128.69,
          "n_train": 44173,
          "n_test": 23737
        },
        "tags": {
          "tipo_target": "m2",
          "calibracion_iaao": "estratificada"
        },
        "creado_en": "2026-10-07T12:00:00Z",
        "fuente_artefacto": "models:/xgboost_venta/8"
      },
      "deltas": {
        "delta_mape_pct": -0.8274,
        "delta_r2": -0.0001,
        "delta_mae": -1223.45,
        "delta_rmse": 10.26,
        "mejora_mape": true,
        "mejora_r2": false
      },
      "recomendacion": "RECOMENDADO_PARA_PRODUCCION",
      "motivo": "El candidato mejora el desempeño frente a producción (ΔMAPE: -0.83%, ΔR²: -0.0001) y supera el benchmark exigido (15.0%)."
    }
  ],
  "total_candidatos": 1,
  "generado_en": "2026-10-12T23:21:13Z"
}
```

> **Valores posibles de `recomendacion`:**
> - `RECOMENDADO_PARA_PRODUCCION`: Supera al modelo en producción y bate los benchmarks.
> - `NO_RECOMENDADO`: El error MAPE aumentó respecto a la versión en producción.
> - `REVISION_MANUAL`: Mejoras ambiguas (por ejemplo: menor MAPE pero menor $R^2$).

---

### 4.2. Historial de Versiones en Model Registry
- **Método / Ruta:** `GET /api/v1/admin/modelos/historial/{tipo_operacion}`
- **Descripción:** Lista todas las versiones históricas registradas en MLflow con sus etapas (`Production`, `Candidate`, `Archived`), fecha de creación y métricas.
- **Respuesta (HTTP 200 OK):**
```json
{
  "tipo_operacion": "venta",
  "nombre_modelo": "xgboost_venta",
  "total_versiones": 8,
  "versiones": [
    {
      "nombre_modelo": "xgboost_venta",
      "version": "8",
      "stage": "Production",
      "run_id": "b1c2d3e4f5a6b7c8d9e0",
      "metricas": {
        "mape_pct": 14.1571,
        "r2": 0.7472,
        "mae": 50897.93,
        "rmse": 78128.69
      },
      "tags": { "experimento": "target_m2_calibrado" },
      "creado_en": "2026-10-07T12:00:00Z"
    },
    {
      "nombre_modelo": "xgboost_venta",
      "version": "7",
      "stage": "Archived",
      "run_id": "0ac890d6d1ca45a8b77b977eb96a0cff",
      "metricas": {
        "mape_pct": 14.9845,
        "r2": 0.7473,
        "mae": 52121.38,
        "rmse": 78118.43
      },
      "tags": {},
      "creado_en": "2026-10-06T04:40:00Z"
    }
  ],
  "generado_en": "2026-10-12T23:22:00Z"
}
```

---

### 4.3. Promover Modelo Candidato a Producción (Aprobación Humana)
- **Método / Ruta:** `POST /api/v1/admin/modelos/promover`
- **Descripción:** Acción definitiva de gobernanza de MLOps:
  1. Marca en MLflow la versión elegida como `Production` y archiva (`Archived`) la versión previa.
  2. Si `version` no se envía (o es `null`), **resuelve automáticamente el último candidato activo**.
  3. Soporta formatos `"8"`, `8` o `"v8"` mediante sanitización automática.
  4. Descarga y carga el artefacto binario directamente en la **memoria RAM** de `ml-service`.
  5. Reconstruye en caliente el `TreeExplainer` de SHAP en RAM sin reiniciar el servidor (*Zero Downtime*).
  6. **Calcula y retorna los deltas de métricas ($\Delta\text{MAPE}$, $\Delta R^2$, $\Delta\text{MAE}$)** frente a la versión previamente activa.
- **Request Body (Simplificado — `version` y `motivo` son opcionales):**
```json
{
  "tipo_operacion": "venta",
  "motivo": "Aprobado por el comité de tasación: MAPE alcanzó 14.16% superando benchmark."
}
```
*(O con versión explícita: `{"tipo_operacion": "venta", "version": "8", "motivo": "..."}`)*.

- **Respuesta (HTTP 200 OK):**
```json
{
  "exito": true,
  "mensaje": "Modelo xgboost_venta v8 promovido exitosamente a Producción y cargado en memoria RAM.",
  "tipo_operacion": "venta",
  "version_promovida": "8",
  "version_anterior_archivada": "7",
  "modelo_cargado_en_ram": true,
  "metricas": {
    "mape_pct": 14.1571,
    "r2": 0.7472,
    "mae": 50897.93,
    "rmse": 78128.69,
    "n_train": 44173,
    "n_test": 23737
  },
  "metricas_anteriores": {
    "mape_pct": 14.9845,
    "r2": 0.7473,
    "mae": 52121.38,
    "rmse": 78118.43,
    "n_train": 44173,
    "n_test": 23737
  },
  "deltas": {
    "delta_mape_pct": -0.8274,
    "delta_r2": -0.0001,
    "delta_mae": -1223.45,
    "delta_rmse": 10.26,
    "mejora_mape": true,
    "mejora_r2": false
  },
  "promovido_en": "2026-10-13T01:16:44Z"
}
```

---

### 4.4. Rollback Preventivo a Versión Archivada
- **Método / Ruta:** `POST /api/v1/admin/modelos/rollback`
- **Descripción:** Acción de gobernanza para revertir el modelo en producción ante contingencias o anomalías:
  1. Si `version` no se especifica, es `null` o viene el placeholder `"string"` de Swagger, **revierte automáticamente a la versión previa archivada más reciente**.
  2. Si se especifica una versión (`"7"`), restaura esa versión histórica exacta a `Production` y archiva la versión actual.
  3. Ejecuta Hot-Reload seguro en RAM y reconstruye SHAP en caliente.
  4. Calcula y retorna el impacto en métricas ($\Delta\text{MAPE}$).
- **Request Body (Opcional):**
```json
{
  "tipo_operacion": "venta",
  "motivo": "Reversión preventiva por degradación de ciclo inmobiliario"
}
```
- **Respuesta (HTTP 200 OK):**
```json
{
  "exito": true,
  "mensaje": "Modelo xgboost_venta v7 promovido exitosamente a Producción y cargado en memoria RAM.",
  "tipo_operacion": "venta",
  "version_promovida": "7",
  "version_anterior_archivada": "8",
  "modelo_cargado_en_ram": true,
  "metricas": {
    "mape_pct": 14.9845,
    "r2": 0.7473,
    "mae": 52121.38,
    "rmse": 78118.43,
    "n_train": 44173,
    "n_test": 23737
  },
  "metricas_anteriores": {
    "mape_pct": 14.1571,
    "r2": 0.7472,
    "mae": 50897.93,
    "rmse": 78128.69,
    "n_train": 44173,
    "n_test": 23737
  },
  "deltas": {
    "delta_mape_pct": 0.8274,
    "delta_r2": 0.0001,
    "delta_mae": 1223.45,
    "delta_rmse": -10.26,
    "mejora_mape": false,
    "mejora_r2": true
  },
  "promovido_en": "2026-10-13T01:16:11Z"
}
```

---

## 5. Módulo 3: Observabilidad y Metadatos Activos

### 5.1. Health Check Operacional
- **Método / Ruta:** `GET /health`
- **Descripción:** Diseñado para *health check probes* de Kubernetes, Nginx o comprobación periódica del backend.
- **Respuesta (HTTP 200 OK):**
```json
{
  "status": "ok",
  "modelo_venta": "v8_mlflow",
  "modelo_alquiler": "v4_mlflow",
  "ipc_periodo": "2026-Q1"
}
```
*(Si alguno de los modelos no está cargado en RAM, `status` pasa a `"degraded"`)*.

---

### 5.2. Información y Metadatos del Modelo en Servicio
- **Método / Ruta:** `GET /api/v1/modelo/info?tipo_operacion=venta`
- **Query Params:** `tipo_operacion` (`venta` o `alquiler`, default `venta`)
- **Descripción:** Retorna el estado completo del modelo que atiende las predicciones en este momento:
```json
{
  "version": "v8_mlflow",
  "archivo": "xgboost_venta_v2.pkl",
  "metricas": {
    "mape_pct": 14.1620,
    "r2": 0.7472,
    "mae_soles": 51230.15,
    "rmse_soles": 77650.40,
    "train_periodo": "2016-2023",
    "test_periodo": "2024-2025"
  },
  "ipc_actual": {
    "valor": 169.18,
    "periodo": "2026-Q1",
    "fuente": "INEI/BCRP — serie PN01270PM, Base Dic 2009 = 100",
    "nota": "Actualizar este valor al cierre de cada trimestre sin necesidad de redespliegue"
  },
  "n_features": 33,
  "n_distritos": 22,
  "data_provider": "postgresql",
  "distritos_disponibles": [
    "Ate Vitarte",
    "Barranco",
    "Bellavista",
    "Breña",
    "Callao",
    "Carabayllo",
    "Chorrillos",
    "Comas",
    "Jesus Maria",
    "La Molina",
    "La Victoria",
    "Lima",
    "Lince",
    "Los Olivos",
    "Magdalena del Mar",
    "Miraflores",
    "Pueblo Libre",
    "San Borja",
    "San Isidro",
    "San Miguel",
    "Santiago de Surco",
    "Surquillo"
  ]
}
```

---

### 5.3. Métricas de Infraestructura (CPU y RAM)
- **Método / Ruta:** `GET /api/v1/admin/monitoreo/recursos`
- **Descripción:** Mide el consumo de CPU y memoria RAM del host/contenedor FastAPI en tiempo real para verificar salud de infraestructura.
- **Respuesta (HTTP 200 OK):**
```json
{
  "cpu_usage_pct": 12.4,
  "memory_used_mb": 1420.5,
  "memory_total_mb": 16384.0,
  "memory_usage_pct": 8.67,
  "uptime_seconds": 12450.0,
  "active_threads": 8,
  "timestamp": "2026-10-12T23:45:00Z"
}
```

---

### 5.4. Detección de Data Drift (Population Stability Index - PSI)
- **Método / Ruta:** `GET /api/v1/admin/monitoreo/drift/{tipo_operacion}`
- **Parámetros Path:** `tipo_operacion` (`venta` o `alquiler`)
- **Descripción:** Evalúa el cambio en la distribución de las variables clave del mercado inmobiliario comparando el ciclo histórico de entrenamiento (2016-2023) contra el ciclo reciente de test (2024-2025).
- **Respuesta (HTTP 200 OK):**
```json
{
  "operation_type": "venta",
  "overall_psi": 0.0482,
  "drift_status": "STABLE",
  "baseline_period": "2016-2023",
  "target_period": "2024-2025",
  "features": [
    {
      "feature_name": "superficie_total",
      "psi_score": 0.0312,
      "status": "STABLE",
      "description": "Distribución de áreas físicas sin alteraciones significativas (PSI < 0.10)."
    },
    {
      "feature_name": "precio_soles_const",
      "psi_score": 0.0645,
      "status": "STABLE",
      "description": "Precios reales deflactados por IPC conservan estabilidad de mercado."
    },
    {
      "feature_name": "antiguedad_anios",
      "psi_score": 0.0210,
      "status": "STABLE",
      "description": "Stock inmobiliario mantiene perfil de antigüedad homogéneo."
    }
  ],
  "message": "Distribución de datos estable (PSI global: 0.0482 < 0.10). No se requiere reentrenamiento urgente.",
  "evaluated_at": "2026-10-12T23:45:00Z"
}
```

---

### 5.5. Fluctuación Temporal del Ciclo de Mercado (MAPE / MAE por Periodo)
- **Método / Ruta:** `GET /api/v1/admin/monitoreo/fluctuacion-temporal/{tipo_operacion}`
- **Parámetros Path:** `tipo_operacion` (`venta` o `alquiler`)
- **Descripción:** Genera la serie de tiempo trimestral/mensual para graficar el MAPE y MAE frente a la cota máxima del 15% en el dashboard de monitoreo.
- **Respuesta (HTTP 200 OK):**
```json
{
  "operation_type": "venta",
  "threshold_mape": 15.0,
  "time_series": [
    {
      "period": "2024-Q1",
      "mape_pct": 14.32,
      "mae_soles": 51200.0,
      "upper_threshold_mape": 15.0,
      "sample_count": 3200,
      "exceeds_threshold": false
    },
    {
      "period": "2024-Q2",
      "mape_pct": 13.98,
      "mae_soles": 49850.0,
      "upper_threshold_mape": 15.0,
      "sample_count": 3450,
      "exceeds_threshold": false
    },
    {
      "period": "2024-Q3",
      "mape_pct": 14.16,
      "mae_soles": 50897.9,
      "upper_threshold_mape": 15.0,
      "sample_count": 3610,
      "exceeds_threshold": false
    }
  ],
  "average_mape": 14.15,
  "average_mae": 50649.3,
  "total_evaluated_samples": 10260,
  "generated_at": "2026-10-12T23:45:00Z"
}
```

---

### 5.6. Latencia P95 y Verificación de Artefactos
- **Método / Ruta:** `GET /api/v1/admin/monitoreo/latencia-p95/{tipo_operacion}`
- **Parámetros Path:** `tipo_operacion` (`venta` o `alquiler`)
- **Descripción:** Ejecuta una batería de micro-benchmarks de inferencia para certificar que el percentil 95 (P95) cumpla el SLA (<100ms) y valida la integridad de los artefactos `.pkl` y `.json`.
- **Respuesta (HTTP 200 OK):**
```json
{
  "operation_type": "venta",
  "model_version": "v8_mlflow",
  "p50_latency_ms": 14.2,
  "p95_latency_ms": 28.6,
  "p99_latency_ms": 42.1,
  "sla_threshold_ms": 100.0,
  "meets_sla": true,
  "sample_inferences": 100,
  "artifacts_status": {
    "modelo_pkl": true,
    "metricas_json": true,
    "params_json": true,
    "config_json": true
  },
  "all_artifacts_ready": true,
  "measured_at": "2026-10-12T23:45:00Z"
}
```

---

## 6. Diccionario de Métricas Oficiales: ML Clásico vs Normas IAAO

Para construir la interfaz y la lógica de validación en el Backend, este es el significado exacto de las métricas generadas:

| Categoría | Métrica en JSON | Rango / Umbral Objetivo | Interpretación Técnica en Tasación |
| :--- | :--- | :---: | :--- |
| **Error Relativo** | `mape_pct` | $\le 15.0\%$ (Venta)<br>$\le 17.89\%$ (Alquiler) | Porcentaje promedio de desviación absoluta sobre el valor real. Venta actual: **14.16%**, Alquiler actual: **13.76%**. |
| **Ajuste Global** | `r2` | $\ge 0.70$ (Venta)<br>$\ge 0.65$ (Alquiler) | Varianza de los precios de mercado explicada por las 33 features socioeconómicas y físicas. |
| **Error Escalar** | `mae` / `rmse` | Menor es mejor | Error en Soles constantes (sin inflación). MAE penaliza linealmente; RMSE penaliza cuadráticamente grandes desvíos. |
| **Nivel IAAO** | `median_ratio` | **`0.90 – 1.10`** (Ideal: 1.00) | Mediana del cociente $\frac{\text{Predicho}}{\text{Real}}$. Con calibración estratificada, se ubica exactamente en **1.0000**. |
| **Dispersión IAAO** | `cod_pct` | **`5.0% – 15.0%`** | Coeficiente de Dispersión (COD). Mide la equidad horizontal (dispersión de errores respecto a la mediana). Actual: **14.12%** ✅. |
| **Variación IAAO**| `cov_pct` | $\le 20.0\%$ | Coeficiente de variación paramétrico basado en desviación estándar de ratios. |
| **Equidad Vertical**| `prd` | **`0.98 – 1.03`** (Ideal: 1.00) | Diferencial Relacionado con el Precio (*Price-Related Differential*). Si es $>1.03$ hay regresividad; si es $<0.98$ progresividad. Actual: **1.0084** ✅. |
| **Sesgo de Precio** | `prb` | **`-0.05 a +0.05`** | Sesgo Relacionado con el Precio (*Price-Related Bias*). Regresión del ratio sobre el logaritmo del precio. Actual: **-0.0125** ✅. |

---

## 7. Recomendaciones para el Backend de Monitoreo (Spring Boot / NestJS)

### 7.1. Separación Conceptual: MLflow Model Registry vs Historial de Auditoría

Es fundamental distinguir los dos tipos de historiales en la arquitectura MLOps:

1. **Historial Técnico de Modelos (MLflow Tracking & Registry):**
   - **Endpoint:** `GET /api/v1/admin/modelos/historial/{tipo_operacion}`
   - **Propósito:** Registro técnico e inmutable de los artefactos (`.pkl`), tags, hiperparámetros, métricas matemáticas de evaluación y etapas del ciclo de vida (`Production`, `Candidate`, `Archived`).
   - **Almacenamiento:** Administrado por el servidor de MLflow en PostgreSQL (`mlops.model_versions`).

2. **Historial de Auditoría y Gobernanza de Negocio (Backend de Monitoreo):**
   - **Propósito:** Trazabilidad de las decisiones operacionales humanas (quién autorizó un pase a producción, cuándo se ejecutó un rollback preventivo, qué justificación se ingresó y cuál fue el impacto en $\Delta\text{MAPE}$).
   - **Almacenamiento:** Tabla relacional en la base de datos operacional (`db_operacional_valuo` / esquema `monitoring`).

```sql
CREATE TABLE monitoring.governance_audit_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    action_type VARCHAR(30) NOT NULL, -- 'TRAINING_TRIGGERED', 'MODEL_PROMOTED', 'MODEL_ROLLED_BACK'
    operation_type VARCHAR(20) NOT NULL, -- 'VENTA' | 'ALQUILER'
    from_version VARCHAR(20),            -- Versión antes del evento (ej: '7')
    to_version VARCHAR(20),              -- Versión resultante (ej: '8')
    delta_mape_pct NUMERIC(6,4),         -- Impacto en error MAPE (ej: -0.8274)
    reason TEXT,                         -- Justificación ingresada por el usuario
    triggered_by VARCHAR(100),           -- Usuario / Admin responsable (email o username)
    status VARCHAR(20) NOT NULL,         -- 'SUCCESS', 'FAILED'
    metadata JSONB,                      -- Objeto JSON con deltas y métricas completas
    created_at TIMESTAMPTZ DEFAULT clock_timestamp()
);
```

---

### 7.2. DTOs y Registros Java para Spring Boot

Copia y pega estas definiciones en tu proyecto `valuo-monitoring-service`:

```java
package pe.ventage.valuo_monitoring_service.infrastructure.client.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import java.time.OffsetDateTime;
import java.util.List;
import java.util.Map;

// 1. Reentrenamiento
public record IniciarEntrenamientoRequest(
    @JsonProperty("nombre_modelo") String nombreModelo,       // Opcional
    @JsonProperty("guardar_como_activo") Boolean guardarComoActivo, // default false
    @JsonProperty("max_depth") Integer maxDepth,             // Opcional
    @JsonProperty("learning_rate") Double learningRate,       // Opcional
    @JsonProperty("n_estimators") Integer nEstimators,       // Opcional
    @JsonProperty("usar_target_m2") Boolean usarTargetM2     // Opcional
) {}

public record JobEstadoResponse(
    @JsonProperty("job_id") String jobId,
    @JsonProperty("tipo_operacion") String tipoOperacion,
    @JsonProperty("estado") String estado, // "en_progreso", "completado", "fallido"
    @JsonProperty("progreso") String progreso,
    @JsonProperty("duracion_segundos") Double duracionSegundos,
    @JsonProperty("mlflow_run_id") String mlflowRunId,
    @JsonProperty("mlflow_model_version") String mlflowModelVersion,
    @JsonProperty("metricas") Map<String, Object> metricas,
    @JsonProperty("error") String error
) {}

// 2. Gobernanza: Promover y Rollback
public record PromocionModeloRequest(
    @JsonProperty("tipo_operacion") String tipoOperacion, // "venta" | "alquiler"
    @JsonProperty("version") String version,             // Opcional (null = auto último candidato)
    @JsonProperty("motivo") String motivo                // Opcional
) {}

public record RollbackModeloRequest(
    @JsonProperty("tipo_operacion") String tipoOperacion, // "venta" | "alquiler"
    @JsonProperty("version") String version,             // Opcional (null = auto última versión archivada)
    @JsonProperty("motivo") String motivo                // Opcional
) {}

public record PromocionModeloResponse(
    @JsonProperty("exito") Boolean exito,
    @JsonProperty("mensaje") String mensaje,
    @JsonProperty("tipo_operacion") String tipoOperacion,
    @JsonProperty("version_promovida") String versionPromovida,
    @JsonProperty("version_anterior_archivada") String versionAnteriorArchivada,
    @JsonProperty("modelo_cargado_en_ram") Boolean modeloCargadoEnRam,
    @JsonProperty("metricas") MetricasSummaryDTO metricas,
    @JsonProperty("metricas_anteriores") MetricasSummaryDTO metricasAnteriores,
    @JsonProperty("deltas") DeltasMetricasDTO deltas,
    @JsonProperty("promovido_en") OffsetDateTime promovidoEn
) {}

public record MetricasSummaryDTO(
    @JsonProperty("mape_pct") Double mapePct,
    @JsonProperty("r2") Double r2,
    @JsonProperty("mae") Double mae,
    @JsonProperty("rmse") Double rmse,
    @JsonProperty("n_train") Integer nTrain,
    @JsonProperty("n_test") Integer nTest
) {}

public record DeltasMetricasDTO(
    @JsonProperty("delta_mape_pct") Double deltaMapePct, // Negativo = Error redujo (Mejora)
    @JsonProperty("delta_r2") Double deltaR2,
    @JsonProperty("delta_mae") Double deltaMae,
    @JsonProperty("delta_rmse") Double deltaRmse,
    @JsonProperty("mejora_mape") Boolean mejoraMape,
    @JsonProperty("mejora_r2") Boolean mejoraR2
) {}

// 3. Benchmark Comparativo
public record BenchmarkResponse(
    @JsonProperty("tipo_operacion") String tipoOperacion,
    @JsonProperty("benchmark_referencial_mape") Double benchmarkReferencialMape,
    @JsonProperty("modelo_produccion") ModeloDetalleDTO modeloProduccion,
    @JsonProperty("candidatos") List<CandidatoComparacionDTO> candidatos,
    @JsonProperty("total_candidatos") Integer totalCandidatos,
    @JsonProperty("generado_en") OffsetDateTime generadoEn
) {}

public record CandidatoComparacionDTO(
    @JsonProperty("candidato") ModeloDetalleDTO candidato,
    @JsonProperty("deltas") DeltasMetricasDTO deltas,
    @JsonProperty("recomendacion") String recomendacion, // "RECOMENDADO_PARA_PRODUCCION", "NO_RECOMENDADO", etc.
    @JsonProperty("motivo") String motivo
) {}

public record ModeloDetalleDTO(
    @JsonProperty("nombre_modelo") String nombreModelo,
    @JsonProperty("version") String version,
    @JsonProperty("stage") String stage,
    @JsonProperty("run_id") String runId,
    @JsonProperty("metricas") MetricasSummaryDTO metricas,
    @JsonProperty("creado_en") String creadoEn
) {}
```

---

### 7.3. Estrategia de Polling de Reentrenamiento en Spring Boot
1. Al recibir `job_id` de `POST /api/v1/admin/entrenamiento/{venta|alquiler}`, persistir en `model_training_jobs` con `status = 'IN_PROGRESS'`.
2. Lanzar un scheduled executor o tarea periódica cada **3 a 5 segundos** consultando `GET /api/v1/admin/entrenamiento/estado/{job_id}`.
3. Timeout de seguridad: **120 segundos** (el job habitual toma ~35s).
4. Al detectar `estado === 'completado'`, actualizar la entidad con las métricas finales y `status = 'COMPLETED'`.
5. Si `estado === 'fallido'`, almacenar el mensaje de error y marcar `status = 'FAILED'`.
