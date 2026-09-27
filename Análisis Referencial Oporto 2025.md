Para responder a las observaciones de tu profesora y evaluar la comparabilidad del referente local de **Oporto D’Ugard et al. (2025)**, he analizado los detalles metodológicos del documento.

A continuación tienes la verificación punto por punto de los aspectos solicitados:

### 1\. Unidad de Análisis y Tipo de Inmueble

* **Tipo de inmueble evaluado:** El estudio **no se limita únicamente a departamentos** 1, 2\. La base contiene dos categorías principales 1:  
* **Departamento:** 26,427 registros (\~66.7%) 1, 2\.  
* **Vivienda Unifamiliar (Casas):** 12,992 registros (\~32.8%) 1, 2\.  
* **Unidad de análisis / Variable objetivo:** Es el **valor comercial de tasación** (en USD) de bienes inmuebles presentados como garantía o colateral en una entidad financiera bancaria en el Perú 1, 3, 4\.

### 2\. Definición y Cálculo del MAPE

* **Fórmula utilizada:** Utilizan la definición formal estándar del *Mean Absolute Percentage Error* 5:\\\\\\text{MAPE} \= \\frac{1}{n} \\sum\_{i=1}^{n} \\left| \\frac{y\_i \- \\hat{y}\_i}{y\_i} \\right| \\times 100\\\\  
* **Implementación en código:** En la evaluación del modelo en Python calculan explícitamente 6:np.mean(np.abs(y\_test \- predict\_t) / y\_test) \* 100  
* **Comparabilidad de la métrica:** La definición es la métrica de error relativo porcentual medio estándar en escala 0–100% 5, 6\.

### 3\. Partición Train / Test

* **Esquema de división:** Los datos se dividieron en **80% para entrenamiento (Train)** y **20% para validación/prueba (Test)** 7\.  
* **Desempeño reportado para XGBoost:**  
* **Train MAPE:** 17.96% 8\.  
* **Test MAPE:** **17.89%** (el valor citado en tu benchmark) 8, 9\.

### 4\. Fuente y Calidad de Datos

* **Fuente de información:** Utilizan una base interna de **39,619 tasaciones periciales** de una entidad bancaria peruana realizadas entre 2014 y 2023 10, 11, enriquecida con información geográfica del INEI y series económicas del BCRP (tipo de cambio e IPC) 12-15.  
* **Limpieza y filtrado de datos:**  
* Se eliminaron 53 casos atípicos en la variable *Valor Comercial* 4, 16, 7 valores extremos en *Edad* 16 y 181 casos con inconsistencias de ubicación 6\.  
* Reemplazo de valores faltantes por 0 en campos como depósitos, estacionamientos y elevador 6, 17\.  
* Imputación por la mediana en variables continuas como *Edad*, *Área de terreno* y *Área de construcción* 17\.  
* **Ingeniería de características destacada:** Desarrollaron agrupaciones geográficas basadas en latitud y longitud mediante un árbol de regresión que segmentó el territorio en 27 nodos espaciales 6, 18\.

### 5\. Detalle del Modelo y Contexto de la Métrica (17.89%)

* Cabe destacar que en el estudio, el modelo **LightGBM** obtuvo un MAPE menor en la muestra Test (**15.42%**) 7, 19\.  
* Sin embargo, los autores **eligieron XGBoost (17.89%)** como su modelo final porque les permitía aplicar restricciones de monotonicidad para alinear el comportamiento del modelo con el sentido de negocio 9, 20, 21\.

&nbsp;

estoy realizanod mi tesis respecot a un modelo de ml para la prediccion de precios inmobiliarios (departamento) en lima metropolitana, y en mi benchmanr meniconamos lo siguiente

&nbsp;

&nbsp;

Debido a que la evaluación de precisión se sustenta en resultados obtenidos sobre mercados distintos del limeño, la decisión algorítmica requerirá validación empírica. Esta se realizará durante el Sprint 1 mediante el entrenamiento sobre el dataset del BCRP y la comparación de las métricas obtenidas frente al referente local de 17.89% reportado por Oporto D’Ugard et al. (2025).\[

&nbsp;

&nbsp;

aca la profe snos dijo lo siguiente:

Buena idea usar un referente local, pero verifiquen comparabilidad antes de convertirlo en línea base: misma unidad de análisis, tipo de inmueble, definición de MAPE, partición train/test y calidad de datos. Si no son comparables, puede servir como referencia contextual, no como benchmark directo.

&nbsp;