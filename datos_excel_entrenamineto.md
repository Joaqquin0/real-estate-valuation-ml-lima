# Consideraciones con las 4 base de datos que estamos recolectando

## NSE: 
En este caso nostors con los datos directos del INEI estamos calculando este valor, puede ser diferente al de otros organimos poque no sabemos que otros valores esten considetnado, para evitar sesgos, eliminamos los distriytos que tienen pocas residencias encuestadas, dejandlo en un minimo de 30 encuestas para considerarlos como un dato confiable

Tambien puede sirgir el caso que hayan hecho las encuentas en algunas zonas mas pobres en distritos acomodos, como el caso de Miraflores donde tiene un poprcentaje en la calose D o E alto, importnate considerar

Construcción de la variable de Nivel Socioeconómico (NSE)

Para la incorporación del Nivel Socioeconómico (NSE) como variable predictora del modelo, se evaluó la posibilidad de replicar la metodología propietaria de la Asociación Peruana de Empresas de Investigación de Mercados (APEIM), la cual combina aproximadamente diez variables provenientes de distintos módulos de la Encuesta Nacional de Hogares (ENAHO) —equipamiento del hogar, nivel educativo y afiliación de salud del jefe de hogar, material de vivienda, entre otras— ponderadas mediante una fórmula de puntaje no pública en su totalidad y sujeta a revisiones periódicas por parte de la asociación.

Dado que el presente estudio se enmarca en el campo de la ingeniería de software y no en la ciencia de datos aplicada a estratificación socioeconómica, se optó por utilizar directamente la variable ESTRSOCIAL (Estrato Socioeconómico), calculada y publicada oficialmente por el Instituto Nacional de Estadística e Informática (INEI) dentro de la base Sumaria de la ENAHO. Esta decisión se sustenta en tres criterios:

Validez de la fuente: ESTRSOCIAL es una variable oficial, calculada por la misma entidad que produce los datos primarios, lo que garantiza consistencia metodológica interna y evita la introducción de sesgos derivados de una reconstrucción manual de fórmulas no completamente documentadas.
Reducción de riesgo metodológico: la replicación de la fórmula APEIM requeriría la integración de múltiples módulos de la ENAHO (Vivienda y Hogar, Miembros del Hogar, Educación, Salud, Equipamiento del Hogar y Sumaria), así como la aplicación de pesos específicos por variable que han sido modificados en revisiones recientes de la metodología (por ejemplo, la exclusión de las variables de horno microondas y material de paredes en la actualización 2023). Reconstruir esta fórmula sin ser el foco central de la investigación incrementaría el riesgo de error sin un beneficio proporcional para los objetivos del estudio.
Alcance del estudio: el NSE constituye una variable de entorno socioeconómico de apoyo al modelo predictivo, no el objeto central de investigación, por lo que se priorizó una fuente validada y de fácil trazabilidad sobre una reconstrucción metodológicamente compleja.

Procesamiento de la variable

Se utilizó la base Sumaria de la ENAHO 2025 (año más reciente consolidado disponible al momento del desarrollo del estudio), filtrando los hogares correspondientes a Lima Metropolitana y la Provincia Constitucional del Callao mediante el código UBIGEO. Los porcentajes de distribución del NSE por distrito (pct_NSE_A a pct_NSE_E) se calcularon de forma ponderada, utilizando el factor de expansión muestral (FACTOR07) provisto por el INEI, a fin de que cada hogar encuestado representara proporcionalmente al número de hogares reales que expande en la población total. La categoría "Rural" (ESTRSOCIAL = 6) fue excluida del análisis por no ser aplicable al ámbito geográfico de Lima Metropolitana y Callao, de naturaleza íntegramente urbana.

Tratamiento de distritos con muestra insuficiente

Siguiendo el criterio de representatividad estadística, se estableció un umbral mínimo de 30 hogares encuestados por distrito para considerar válida la estimación del NSE. De los 50 distritos que conforman el área de estudio, 14 no alcanzaron dicho umbral. Al contrastar esta lista contra los distritos efectivamente presentes en la base de precios de venta (Banco Central de Reserva del Perú, BCRP), se identificó que únicamente 4 distritos —Barranco, Magdalena del Mar, Lince y San Luis— requerían tratamiento, dado que los 10 distritos restantes sin muestra suficiente no registran transacciones en la fuente de precios utilizada.

Para estos 4 distritos, se aplicó una imputación basada en el distrito colindante con mayor proximidad geográfica que sí contara con una estimación de NSE válida, calculada mediante la distancia euclidiana entre centroides distritales (proyección UTM 18S, EPSG:32718). Este criterio se fundamenta en el patrón de distribución socioeconómica de Lima Metropolitana, caracterizado por una fuerte correlación espacial entre distritos colindantes, superior a la que existiría al utilizar un promedio general del área metropolitana. Cada observación imputada fue marcada mediante una variable indicadora (nse_imputado), a fin de permitir un análisis de sensibilidad posterior que contraste el desempeño del modelo con y sin dichas observaciones.

## Indecencia delictica: 
Se esta cosniderenado la tasa de hurto y robo de los años 2019 - 2025 segun los datos recolectados, este dato representa todo el año para los distritos y no hay mas cosnideracion en este caso (no se estan cubriendo los años 2016 - 2017 - 2018 a diferencia de los datos de venta del BCRP)

## Datos cartograficos
En este caso igualmente se usan los datos del INEI para calcular la distancia de todos los distritos con un punto centro, en este caso el distrito de Lima, luego, para determianr la distacia mas cercana de cada distrito hacia un punto interes fuera del mimso, se usa OpenStreetMap, es un mapa colaborativo, y su nivel de detalle en Lima no es uniforme entre distritos. Zonas como Miraflores, San Isidro o el centro de Lima suelen estar muy bien mapeadas (con cientos de colegios y hospitales etiquetados), mientras que distritos periféricos como Lurín, Pucusana o zonas de Lima Este pueden tener mapeo incompleto — no porque no existan esos servicios, sino porque nadie los agregó a OSM todavía.
Esto es exactamente el mismo tipo de limitación de infraestructura de datos que ya documentaste en tu sustento del problema — así que si encuentras zonas con "0 colegios cercanos" que sabes que sí tienen colegios, no es un error de tu código, es una limitación real de la fuente que debes documentar igual que hiciste con el NSE y la cobertura del BCRP.

Se calculo la variable 
Calcular área en km² (UTM está en metros → dividir entre 1_000_000)
distritos["area_km2"] = distritos.geometry.area / 1_000_000

## Proyeccion Poblacional
Datos sacados de la INEI, se esta cosnidetando la proyeccion para todos los distros en los años 2018 - 2026, se debe de cosniderar que no tenemos datos del 16-17, se debe ver como manejoar los inmuebles de estos años para su dato de proyeccion

**densidad_hab_km2 (población / área del distrito, que sacaste del shapefile hace un momento), calculo propio con los datos recolectados**

# Consideracion con las Fechas de los datos

Se esta considerando unicamente el dato estatico del 2025 con el NSE ya que es un dato el cual no se puede obtener con facilidad y no tenías series históricas confiables y hubieras tenido que construirlas desde cero con mucho riesgo de huecos.

# Imputacion de datos en el excel

para los valores en la** tasa de Criminalidad,** se esta imputando a los valores entre 2016 y 2017, 2018 con con los valores del año 2019, esto ya que equivalen a un 11% de los datos totales y esto dejarlo vacio no beneficiaria al entrenamietno en XGBOOSt

Igualemnte la poblacion_proyecta entre los años 2016 y 2017 se usaron los datos del año 2018

## Imputacion de NSE en el excel final

Al aplicar el criterio de proximidad geográfica pura, se identificó que, para los distritos de Lince y San Luis, el vecino colindante más cercano correspondía a un distrito con un perfil socioeconómico significativamente atípico respecto al distrito a imputar (San Isidro y San Borja, respectivamente, ambos caracterizados por una concentración de NSE A considerablemente superior al promedio de Lima Metropolitana). Dado que el objetivo de la imputación es aproximar el NSE real del distrito faltante y no únicamente minimizar la distancia euclidiana, se optó por sustituir el vecino asignado automáticamente por el distrito colindante con el perfil socioeconómico más representativo del entorno inmediato: Jesús María para Lince y La Victoria para San Luis, ambos con menor distancia relativa de la que introduce un valor atípico y con una composición socioeconómica más consistente con la del distrito imputado.
(10454 datos total)
Barracon: Miraflores
Magdalena: Pueblo lIbre
Lince: Jesus maria
sAN LUYIS: Victoria

# Precios de venta del BCRP

Para tu tesis, esto se documenta así: "el modelo fue entrenado únicamente con los distritos para los cuales el BCRP reporta transacciones inmobiliarias, lo cual excluye N distritos de menor dinamismo de mercado formal; esta limitación se debe a la cobertura de la fuente primaria de precios, no a la disponibilidad de variables explicativas, las cuales sí fueron construidas para los 50 distritos de Lima Metropolitana y Callao."

Distritos en dataset de precios: 26
Distritos cubiertos: ['Ate Vitarte', 'Barranco', 'Bellavista', 'Breña', 'Callao', 'Carabayllo', 'Cercado de Lima', 'Chorrillos', 'Comas', 'Jesús María', 'La Molina', 'La Perla', 'La Victoria', 'Lince', 'Los Olivos', 'Magdalena', 'Miraflores', 'Pueblo Libre', 'San Borja', 'San Isidro', 'San Juan de Lurigancho', 'San Luis', 'San Martín de Porres', 'San Miguel', 'Surco', 'Surquillo']
Distritos SIN datos de venta (no en el BCR): 24

**densidad_hab_km2 (población / área del distrito, que sacaste del shapefile hace un momento), calculo propio con los datos recolectados**
no usó un año fijo — usó, para cada fila (cada inmueble), la población correspondiente al año exacto de venta de ese inmueble. Un inmueble vendido en 2020 en Breña usa la población de Breña en 2020; uno vendido en 2024 en Breña usa la población de Breña en 2024. El área (area_km2) sí es fija (viene del shapefile, no cambia por año), pero la población variable por año hace que densidad_hab_km2 también varíe año a año para el mismo distrito.

# Filas en el excel de entreamineto final

ID	Anio	Trimestre	Precio_Dolares	Tipo_Cambio	IPC	Precio_Soles	Precio_Soles_Const	Ubigeo	Distrito	Superficie	Habitaciones	Banios	Garajes	Piso	Vista_Exterior	Antiguedad	precio_dolares_m2	pct_NSE_A	pct_NSE_B	pct_NSE_C	pct_NSE_D	pct_NSE_E	tasa_denuncias	tasa_robo	tasa_hurto	tasas_criminalidad_imputada	poblacion_proyectada	poblacion_imputada	area_distrito_km2	distancia_centro_km	dist_colegio_km	dist_hospital_km	dist_estacion_transporte_km	dist_centro_comercial_km	dist_parque_km	dist_universidad_km	densidad_hab_km2	nse_imputado


