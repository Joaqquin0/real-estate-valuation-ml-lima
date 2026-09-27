# Especificación Técnica y Metodológica — Módulo de Rentabilidad Inmobiliaria (Cap Rate & NOI)

**Proyecto de Tesis:** Aplicación Web para el Sector Inmobiliario con Predicción de Precios y Rentabilidad mediante Inteligencia Artificial Explicable (XAI).  
**Módulo:** Evaluación Financiera Inmobiliaria y Análisis de Retorno (Soporte a la Decisión para Tasadores y Profesionales).  
**Autor:** Joaquín Cortez  
**Fecha:** Septiembre 2026  

---

## 1. Alcance y Arquitectura de Integración

El módulo de rentabilidad actúa como una capa de síntesis financiera que combina de forma desacoplada las predicciones de los dos modelos de Machine Learning (XGBoost):
1. **Modelo de Venta:** Estima el Valor Comercial del Inmueble ($V_{\text{venta}}$ en Soles/Dólares).
2. **Modelo de Alquiler:** Estima el Canon de Arrendamiento Mensual ($P_{\text{alquiler}}$ en Soles/Dólares).

Ambas inferencias se evalúan sobre el mismo vector de características del departamento (distrito, superficie, habitaciones, baños, antigüedad, piso, etc.) y se procesan en un motor financiero analítico para calcular el retorno desapalancado (*unlevered*).

```
                      [Entrada de Inmueble (Usuario/Tasador)]
                                       |
                   +-------------------+-------------------+
                   |                                       |
                   v                                       v
         [Modelo ML - Venta]                    [Modelo ML - Alquiler]
                   |                                       |
        V_venta (Valor Comercial)               P_alquiler (Alquiler Mensual)
                   \                                       /
                    \                                     /
                     v                                   v
    +-------------------------------------------------------------------------+
    |                     MÓDULO FINANCIERO DE RENTABILIDAD                  |
    |                                                                         |
    |   1. GSI = P_alquiler * 12                                              |
    |   2. OpEx = GSI * OER  (OER parametrizable: slider 10%-35%, def: 15%)   |
    |   3. NOI = GSI - OpEx = GSI * (1 - OER)                                 |
    |   4. Cap Rate = (NOI / V_venta) * 100%                                  |
    |   5. GRM (Años de Recuperación) = V_venta / GSI                         |
    +-------------------------------------------------------------------------+
                                       |
                                       v
         [Visualización en UI Web: Tarjetas KPI + Cascada + Semáforo]
```

---

## 2. Formulación Matemática y Supuestos Financieros

### 2.1 Ecuaciones de Cálculo

1. **Ingreso Potencial Bruto Anual ($GSI$ - *Gross Scheduled Income*):**
   $$GSI = P_{\text{alquiler}} \times 12$$

2. **Gastos Operativos Estimados ($OpEx$ - *Operating Expenses*):**
   Regulados por el **Ratio de Gastos Operativos (OER - *Operating Expense Ratio*)**:
   $$OpEx = GSI \times OER$$
   *El OER modela costos promedio de mantenimiento que asume el propietario, arbitrios, seguro y pequeñas contingencias.*

3. **Ingreso Operativo Neto Anual ($NOI$ - *Net Operating Income*):**
   $$NOI = GSI - OpEx = (P_{\text{alquiler}} \times 12) \times (1 - OER)$$

4. **Tasa de Capitalización ($Cap\ Rate$ - Desapalancada / *Unlevered*):**
   $$\text{Cap Rate (\%)} = \left(\frac{NOI}{V_{\text{venta}}}\right) \times 100\%$$

5. **Multiplicador de Renta Bruta ($GRM$ - Años de Recupero Simple):**
   $$GRM = \frac{V_{\text{venta}}}{GSI}$$

---

### 2.2 Supuestos y Limitaciones Metodológicas (Declaración de Tesis)

Para salvaguardar el rigor académico del proyecto, se establecen y documentan las siguientes premisas:
* **Enfoque Desapalancado (*Unlevered*):** Se asume que la adquisición se efectúa íntegramente con recursos propios (100% *equity*), prescindiendo de amortizaciones de crédito hipotecario, intereses y costos de estructuración de deuda.
* **Tasa de Vacancia Agregada:** Al no disponer de registros históricos pareados de vacancia efectiva por predio en el dataset abierto del BCRP, los períodos de desocupación se subsumen dentro del OER parametrizable.
* **Exclusión de Plusvalía:** El indicador evalúa exclusivamente el rendimiento corriente por rentas (*income yield*), sin incorporar proyecciones especulativas de apreciación de capital en el tiempo.
* **Convergencia ROI y Cap Rate:** Bajo el supuesto *unlevered* sin deuda, el Retorno sobre la Inversión (ROI) converge formalmente con la Tasa de Capitalización ($ROI \equiv Cap\ Rate$), razón por la cual no se duplican como métricas independientes.

---

## 3. Calibración del Semáforo y Benchmark de Mercado

### 3.1 Justificación de Cobertura (22 Distritos de Lima)

Una interrogante habitual es si una muestra concentrada en 22 distritos (de los 50 existentes en Lima Metropolitana y Callao) es estadísticamente confiable. La respuesta técnica y económica es afirmativa:
1. **Concentración de la Oferta Formal:** Según reportes sectoriales de CAPECO y ASEI, los 22 distritos monitoreados concentran **entre el 85% y 90% de la oferta y demanda de departamentos residenciales formales** con acceso a crédito hipotecario y contratos de arrendamiento bancarizados.
2. **Alineación con el BCRP:** El propio Banco Central de Reserva del Perú limita su *Índice de Precios de Departamentos* y sus estudios de rentabilidad a este mismo grupo distrital, por carecer los distritos periféricos y balnearios de una masa crítica líquida de departamentos en altura.
3. **Población Objetivo:** La unidad muestral son **departamentos en el mercado formal líquido**, no suelo rústico ni vivienda informal autoconstruida.

---

### 3.2 Rangos del Semáforo Comparativo (Fuente: BCRP)

El BCRP publica periódicamente la serie del **Ratio Precio de Venta / Alquiler Anual** (años requeridos para repagar un departamento mediante rentas). Los umbrales del semáforo se calibran invirtiendo este ratio oficial y aplicando un OER estándar del 15%:

$$\text{Cap Rate Neto Estimado} = \left(\frac{1}{\text{Ratio Años BCRP}}\right) \times (1 - 0.15) \times 100\%$$

| Nivel de Mercado | Distritos Representativos | Ratio BCRP (Años) | Cap Rate Bruto | Cap Rate Neto (OER 15%) | Interpretación para el Tasador |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Zona Baja / Plusvalía** | San Isidro, Miraflores, San Borja, Barranco | 20 a 24 años | 4.1% – 5.0% | **< 4.2%** | Activo conservador de alta plusvalía y baja rentabilidad corriente. Menor riesgo de vacancia. |
| **Rango de Mercado (Promedio)** | Jesús María, Lince, Magdalena, Surquillo, San Miguel | 16 a 19 años | 5.2% – 6.2% | **4.2% – 6.0%** | Punto de equilibrio estándar de Lima Moderna. Balance óptimo entre liquidez de alquiler y precio. |
| **Alto Rendimiento** | Breña, Cercado de Lima, Chorrillos, La Victoria | 13 a 15 años | 6.6% – 7.7% | **> 6.0%** | Ticket de compra accesible con alta demanda de alquiler relativo. Mayor rotación potencial. |

---

## 4. Especificación para la Interfaz de Usuario (UI / Frontend)

Para que el módulo sea funcional e intuitivo dentro de la aplicación web, se diseñan los siguientes componentes interactivos:

### Componente 1: Tarjetas de Resumen Ejecutivo (KPI Cards)
* **Card 1 (Principal):** `Cap Rate Neto Estimado (%)` (Ej: `5.24%`) con etiqueta de estado según el semáforo (ej: *"En Rango de Mercado"*).
* **Card 2:** `Ingreso Operativo Neto (NOI)` (Ej: `S/. 23,460 / año` o `S/. 1,955 / mes`).
* **Card 3:** `Años de Recuperación (GRM)` (Ej: `16.2 años`).
* **Card 4 (Secundario):** `Cap Rate Bruto` (Ej: `6.16%`).

### Componente 2: Panel de Parámetros Dinámicos (Interactividad)
* **Slider de Gastos Operativos (OER):**
  * Control deslizable de 0% a 35% (Paso: 1%).
  * Valor por defecto recomendado: **15%**.
  * Leyenda de ayuda (*Tooltip*): *"Porcentaje del alquiler destinado a mantenimiento, arbitrios del propietario y contingencias. Recalcula el NOI en tiempo real."*
* **Selector de Moneda:** Botón conmutador `[ S/. Soles ] | [ US$ Dólares ]` (aplica el Tipo de Cambio de referencia).

### Componente 3: Cascada Financiera Anualizada (Desglose de Flujo)
Tabla o diagrama visual de flujo simple:
* $(+)$ **Ingreso Bruto de Alquiler (GSI):** $P_{\text{alquiler}} \times 12$
* $(-)$ **Gastos Operativos Estimados ($OER \times GSI$):** Deducción calculada
* $(=)$ **Ingreso Operativo Neto ($NOI$):** Flujo anual libre
* $(/)$ **Valor Comercial de Venta Predicho:** $V_{\text{venta}}$
* $(=)$ **Cap Rate Resultante:** Porcentaje neto anual

### Componente 4: Barra de Posicionamiento Contextual (Visual)
Una barra horizontal de progreso con 3 zonas de color:
```
  [    Bajo (<4.2%)    ] [    Rango Mercado (4.2% - 6.0%)    ] [    Alto (>6.0%)    ]
                                      ▲
                                 Tu Inmueble: 5.24%
```
* Marcador que ubica la rentabilidad del predio evaluado frente al promedio distrital y al consolidado de Lima.

### Componente 5: Caja de Sustento y Descargo de Responsabilidad (Disclaimer)
Un recuadro informativo al pie del módulo:
> **Nota Metodológica para Peritos y Analistas:**  
> *Este análisis financiero corresponde a una estimación desapalancada (Unlevered Cap Rate) basada en modelos estadísticos de mercado. No considera costos financieros hipotecarios ni fluctuaciones atípicas de desocupación. Diseñado como soporte analítico para peritos tasadores en el marco de la R.M. N° 172-2016-VIVIENDA.*

---

## 5. Implementación en Código (Lógica JavaScript para la Web)

A continuación se detalla la función matemática exacta para integrar en la lógica del frontend:

```javascript
/**
 * Calcula los indicadores de rentabilidad inmobiliaria.
 * @param {number} valorVenta - Precio de venta estimado en Soles.
 * @param {number} alquilerMensual - Precio de alquiler mensual estimado en Soles.
 * @param {number} oerPercent - Ratio de gastos operativos (ej. 15 para 15%).
 * @returns {Object} Indicadores de rentabilidad calculados.
 */
function calcularRentabilidad(valorVenta, alquilerMensual, oerPercent = 15) {
  if (!valorVenta || valorVenta <= 0 || !alquilerMensual || alquilerMensual <= 0) {
    return null;
  }

  const oerDecimal = oerPercent / 100;
  const gsiAnual = alquilerMensual * 12;
  const opexAnual = gsiAnual * oerDecimal;
  const noiAnual = gsiAnual - opexAnual;
  
  const capRateNeto = (noiAnual / valorVenta) * 100;
  const capRateBruto = (gsiAnual / valorVenta) * 100;
  const grmAnios = valorVenta / gsiAnual;

  // Clasificación del Semáforo
  let estadoSemaforo = "Rango de Mercado";
  let claseColor = "text-yellow-600 bg-yellow-50";

  if (capRateNeto < 4.2) {
    estadoSemaforo = "Bajo (Zona de Alta Plusvalía)";
    claseColor = "text-blue-600 bg-blue-50";
  } else if (capRateNeto > 6.0) {
    estadoSemaforo = "Alto Rendimiento";
    claseColor = "text-emerald-600 bg-emerald-50";
  }

  return {
    gsiAnual: Math.round(gsiAnual),
    opexAnual: Math.round(opexAnual),
    noiAnual: Math.round(noiAnual),
    noiMensual: Math.round(noiAnual / 12),
    capRateNeto: Number(capRateNeto.toFixed(2)),
    capRateBruto: Number(capRateBruto.toFixed(2)),
    grmAnios: Number(grmAnios.toFixed(1)),
    oerAplicado: oerPercent,
    estadoSemaforo,
    claseColor
  };
}
```
