# Análisis de Fachadas, Techos y Riesgo Urbano con YOLO

Sistema de detección de objetos aplicado a análisis urbano: identifica techos en imágenes
satelitales y elementos de fachada (ventanas, puertas, balcones, daños estructurales,
señalamiento vial, escena de calle) en fotografías a nivel de calle. Combina siete modelos
YOLO11 entrenados de forma independiente, un motor alterno basado en Detectron2 para la
detección de ventanas, estimación de altura de edificios, un score de riesgo por manzana
(daños + tráfico + altura) y exportación de resultados en PDF.

Desarrollado como parte de un proyecto de análisis de impacto espacial en la colonia
Hipódromo, Ciudad de México.

---

## Tabla de contenidos

1. [Estructura del proyecto](#estructura-del-proyecto)
2. [Instalación](#instalación)
3. [Uso rápido](#uso-rápido)
4. [Organización de datasets y entrenamiento](#1-organización-de-datasets-y-entrenamiento)
5. [Backend de inferencia](#2-backend-de-inferencia)
6. [Exportación de resultados en PDF](#3-exportación-de-resultados-en-pdf)
7. [Tráfico vial](#4-tráfico-vial-factor-de-riesgo-por-manzana)
8. [Score de riesgo por manzana](#5-score-de-riesgo-por-manzana)
9. [Entrenamiento encadenado](#6-entrenamiento-encadenado-sin-supervisión)
10. [Interfaz web](#interfaz-web)
11. [Solución de problemas](#solución-de-problemas)
12. [Créditos](#créditos-y-tecnologías)

---

## Estructura del proyecto

```
├── index.html                    # Interfaz web
├── servidor_deteccion.py         # Servidor Flask: inferencia, riesgo por manzana, PDF
├── reporte_pdf.py                # Generación del reporte PDF (reportlab + matplotlib)
├── organizar_datasets.py         # Normaliza datasets_fuente/ y los organiza en datasets/
├── trafico_tomtom.py             # Recolección de congestión vial (TomTom Traffic API)
├── descargar_sentinel.py         # Descarga imágenes satelitales del área de estudio
├── coordenadas_trafico.xlsx      # Calles de referencia para trafico_tomtom.py
├── hipodromo_manzanas.geojson    # Polígonos de manzanas (geolocaliza detecciones y tráfico)
├── detecciones.db                # Base de datos SQLite (detecciones + tráfico)
├── yolo11x.pt / yolo11l.pt       # Pesos COCO preentrenados (respaldo para categorías sin
│                                 # modelo propio todavía entrenado)
│
├── entrenamiento/                # Scripts de entrenamiento (correr desde la raíz del repo)
│   ├── entrenar_techo.py         #   Techos — 200 épocas
│   ├── entrenar_fachada.py       #   Fachadas (fusión) — 500 épocas
│   ├── entrenar_ventanas.py      #   Ventanas — 500 épocas
│   ├── entrenar_fachada_general.py  # Elementos arquitectónicos — 500 épocas
│   ├── entrenar_danos.py         #   Daños/deterioro — 500 épocas
│   ├── entrenar_senales.py       #   Señalamiento vial — 500 épocas
│   ├── entrenar_calles.py        #   Escena de calle — 500 épocas
│   ├── entrenar_todo.py          #   Orquesta los 7 anteriores en secuencia
│   └── monitor_entrenamiento.py  #   Interfaz gráfica (Tkinter) de progreso
│
├── datasets/                     # Datasets YOLO ya procesados, listos para entrenar
├── datasets_fuente/               # Datasets crudos sin procesar (Roboflow, COCO, máscaras)
├── CMP_facade_DB_base/           # Fachadas con anotaciones XML propias
├── runs/detect/                  # Pesos entrenados: entrenamiento_<categoría>/weights/best.pt
├── ricardo/deteccion-ventanas-streamlit/  # Modelo Detectron2 (motor alterno de Ventanas;
│                                 # obtener aparte, ver instalación)
└── Fotos_Calle/                  # Imágenes de ejemplo para pruebas de análisis por lote
```

> **Los 7 modelos ya entrenados vienen incluidos en el repositorio** (`runs/detect/`,
> vía Git LFS — ver [instalación](#instalación)), listos para usar sin necesidad de
> entrenar nada. Lo que **no** está incluido son los datasets crudos y procesados
> (`datasets_fuente/`, `datasets/`, `CMP_facade_DB_base/`, `Fotos_Calle/`): suman varias
> decenas de gigabytes y solo hacen falta si se quiere reentrenar o agregar categorías
> nuevas — ver [`.gitignore`](.gitignore) y la [sección 1](#1-organización-de-datasets-y-entrenamiento).

---

## Instalación

### Requisitos

- Python 3.8+
- [Git LFS](https://git-lfs.com/) — los pesos entrenados (`*.pt`, `*.pth`) se versionan con
  Git LFS, no como blobs normales de git. Instálalo **antes** de clonar:
  ```bash
  git lfs install
  git clone https://github.com/MAFY284/Analisis-Fachada-General-Techo-Yolo.git
  ```
  Si ya clonaste sin tener Git LFS instalado, los archivos de pesos quedarán como punteros
  de texto en vez de los binarios reales — corre `git lfs pull` dentro del repo para
  descargarlos.
- GPU NVIDIA con CUDA (recomendado; el proyecto se desarrolló y probó con 3× RTX A6000).
  También corre en CPU, con inferencia y entrenamiento considerablemente más lentos.
- Espacio en disco: el repositorio clonado (con los 7 modelos entrenados vía LFS) pesa
  unos pocos GB. Si además se quieren procesar los datasets fuente completos para
  reentrenar, considera 20+ GB libres adicionales.

### Dependencias

```bash
pip install flask flask-cors ultralytics pillow opencv-python-headless numpy \
            matplotlib lxml shapely reportlab pandas openpyxl requests
```

**Opcional — motor Detectron2 para Ventanas.** Detectron2 no está en PyPI; se instala
compilando desde el repositorio oficial:

```bash
pip install git+https://github.com/facebookresearch/detectron2.git
```

Además, sus pesos viven en `ricardo/deteccion-ventanas-streamlit/` — una carpeta que en
este repositorio está registrada como referencia de submódulo pero **sin un
`.gitmodules` que la resuelva**, así que un clon nuevo no la trae poblada
automáticamente. Para tener el motor Detectron2 funcionando hay que obtener el contenido
de esa carpeta por separado y colocarlo en esa misma ruta.

Si Detectron2 no está instalado o no se encuentran sus pesos, el servidor sigue
funcionando con normalidad — la categoría Ventanas simplemente ofrece solo el motor YOLO,
sin la opción Detectron2 en el menú.

### Datos y modelos

Los 7 modelos YOLO ya entrenados vienen incluidos en el repositorio (`runs/detect/`, vía
Git LFS) — tras clonar con Git LFS instalado, el servidor arranca listo para usar, sin
necesidad de entrenar nada.

Los datasets (crudos en `datasets_fuente/` y ya procesados en `datasets/`) **no** están
incluidos por su tamaño. Solo hacen falta si se quiere reentrenar desde cero, agregar una
categoría nueva, o ampliar una existente:

1. Coloca los datasets fuente en `datasets_fuente/` (ver [sección 1](#1-organización-de-datasets-y-entrenamiento)
   para el formato esperado) y corre `organizar_datasets.py` para generarlos en `datasets/`.
2. Entrena con `entrenamiento/entrenar_todo.py` (ver [sección 6](#6-entrenamiento-encadenado-sin-supervisión)) —
   por defecto continúa entrenando (fine-tuning) sobre los pesos que ya vienen en el
   repositorio en vez de partir de cero.

---

## Uso rápido

```bash
python3 servidor_deteccion.py          # arranca en el puerto 3005 por defecto
```

El puerto es configurable por variable de entorno o argumento:

```bash
python3 servidor_deteccion.py 3000          # puerto como argumento
PUERTO=3000 python3 servidor_deteccion.py   # puerto como variable de entorno
```

Abre `http://127.0.0.1:3005` (o el puerto elegido) en el navegador — ver
[Interfaz web](#interfaz-web) para el flujo completo.

---

## 1. Organización de datasets y entrenamiento

### 1.1 Organizar los datasets fuente

```bash
python3 organizar_datasets.py --dry-run   # previsualiza conteos sin copiar nada
python3 organizar_datasets.py             # organiza de verdad
```

`datasets_fuente/Facade/` y `datasets_fuente/techos/` contienen datasets en formatos
heterogéneos — YOLO-bbox, YOLO-segmentación (polígono), COCO (JSON) y máscaras raster
binarias — de fuentes distintas. `organizar_datasets.py` normaliza todo a YOLO-bbox y lo
separa en siete categorías, para poder entrenar cada una por separado o como una fusión
combinada:

| Categoría | Carpeta destino | Clases | Fuentes principales |
|---|---|---|---|
| Techo | `datasets/dataset_techo_yolo/` | 1: `edificio` | Imágenes aéreas (COCO), datasets satelitales con máscaras raster convertidas a cajas, y varios datasets YOLO de edificaciones (bbox y polígono) |
| Ventanas | `datasets/dataset_ventanas/` | 1: `window` | 3 fuentes de fachadas urbanas (incluye formatos en polígono) |
| Fachada general | `datasets/dataset_fachada_general/` | 25 (elementos arquitectónicos) | Dataset de fachadas en polígono + `CMP_facade_DB_base/` (378 fachadas con cajas XML propias) |
| Daños | `datasets/dataset_danos/` | 10 (grietas, corrosión, desprendimientos, humedad, pintura, etc.) | Múltiples datasets de defectos y deterioro de fachada |
| **Fachadas (fusión)** | `datasets/dataset_fachadas/` | 17 (estructura + daño) | Unión remapeada de ventanas + fachada general + daños + puertas + árboles — es la que usa el análisis principal del servidor |
| Señales | `datasets/dataset_senales/` | 47 (señalamiento vial) | Datasets de señalamiento vial en varios idiomas |
| Calles | `datasets/dataset_calles/` | 12 (vehículos, peatones) | Escena de calle: autos, motocicletas, peatones, etc. |

Notas sobre el comportamiento del script:

- **Detección automática de polígonos.** Varias fuentes vienen en formato
  YOLO-segmentación en vez de bbox; el script detecta el número de valores por línea y, si
  es un polígono, calcula la caja envolvente.
- **Conversión de COCO y máscaras raster.** Los datasets en formato COCO se convierten vía
  su JSON de anotaciones; las máscaras binarias se convierten a cajas mediante análisis de
  componentes conexos (`cv2.connectedComponentsWithStats`).
- **Idempotente.** Si se vuelve a ejecutar tras agregar una fuente nueva, no reprocesa lo
  que ya existe en el destino (usa nombres de archivo determinísticos con hash corto para
  evitar colisiones y el límite de 255 caracteres del sistema de archivos).
- `Facade/Señales/` y `Facade/streets/` se organizan pero **no se fusionan** con
  `dataset_fachadas/` — no son elementos de fachada, quedan como categorías independientes.

> **Al modificar una lista de clases (`CLASES_*`):** agrega clases nuevas siempre al final
> de la lista. El script es idempotente por nombre de archivo, no por contenido — si se
> inserta una clase en medio de una lista existente, las etiquetas ya generadas en corridas
> anteriores no se reescriben (se detectan como "ya procesadas") y quedan apuntando al
> índice equivocado del nuevo esquema. La forma segura de corregirlo es borrar la carpeta
> de destino afectada y regenerar desde cero.

El reporte de la última corrida queda en `reporte_organizacion_datasets.json`.

### 1.2 Entrenar

**Los 7 modelos, sin supervisión** (forma recomendada — ver [sección 6](#6-entrenamiento-encadenado-sin-supervisión)):

```bash
nohup python3 entrenamiento/entrenar_todo.py > entrenamiento/log_entrenar_todo.txt 2>&1 &
python3 entrenamiento/monitor_entrenamiento.py   # opcional, progreso en tiempo real
```

**Un modelo individual** (por ejemplo, tras agregar un dataset nuevo a una sola categoría):

```bash
python3 entrenamiento/entrenar_techo.py      # o cualquiera de los otros 6
```

Cada script entrena con las 3 GPUs disponibles (`device='0,1,2'`) — 200 épocas para Techos,
500 para el resto. Si ya existe un modelo entrenado previamente para esa categoría
(`runs/detect/entrenamiento_<id>/weights/best.pt`), el script continúa entrenando desde
esos pesos (fine-tuning) en lugar de partir de cero; la corrida anterior se conserva en una
carpeta `..._previo/` como respaldo. Para entrenar varios modelos en paralelo en vez de uno
por uno, ajusta manualmente el parámetro `device` de cada script a una GPU distinta.

---

## 2. Backend de inferencia

La selección de categoría es manual — se evaluó una auto-detección (correr todos los
modelos y quedarse con el de mayor confianza) pero producía resultados inconsistentes en la
práctica. El frontend usa un menú de casillas: se puede marcar una sola categoría o varias a
la vez sobre la misma imagen.

`servidor_deteccion.py` carga las 7 categorías al arrancar (cada una con respaldo a un peso
COCO preentrenado si todavía no existe un modelo propio entrenado). `POST /detectar` recibe
uno o varios campos `tipo` y ejecuta la inferencia de cada modelo pedido sobre la misma
imagen, devolviendo un resultado por categoría:

```json
{
  "success": true,
  "resultados": {
    "ventanas": { "tipo": "ventanas", "conteo_clases": {"window": 6}, "imagen_base64": "..." },
    "calles":   { "tipo": "calles",   "conteo_clases": {"car": 2},    "imagen_base64": "..." }
  }
}
```

Solo la categoría `fachada` (la fusión combinada) calcula altura estimada, guarda en base
de datos y separa daños de otras clases — es la única con clases de piso/ventana/daño en su
taxonomía. Las demás categorías devuelven conteo por clase e imagen anotada.

`GET /tipos-disponibles` expone la lista de categorías disponibles para que el frontend
arme el menú dinámicamente.

### Inferencia multi-GPU

Al arrancar, el servidor detecta cuántas GPUs hay disponibles y distribuye la inferencia
entre todas (`'0,1,2'` si hay tres, `'cpu'` si no hay ninguna). Para una sola imagen esto no
implica una aceleración de 3×, pero evita que el servidor quede limitado a una sola tarjeta
cuando hay varias peticiones simultáneas o al procesar un lote de imágenes.

### Umbral de confianza ajustable

El frontend incluye un control deslizante (0.05–0.95) que se envía como campo `conf` en
`/detectar` y `/batch/start`, aplicable a todos los modelos y motores por igual. Confianza
más alta produce menos detecciones pero más confiables; confianza más baja detecta más, a
costa de más falsos positivos.

### Ventanas: dos motores de detección

Además del modelo YOLO propio, la categoría Ventanas ofrece un segundo motor: un Faster
R-CNN R50-FPN (Detectron2) tomado del repositorio `ricardo/deteccion-ventanas-streamlit/`
(ver [instalación](#instalación) sobre cómo obtener esa carpeta). El frontend permite
elegir entre YOLO, Detectron2, o ambos — con "ambos" se ejecutan los dos motores y se
muestran dos tarjetas de resultado, más una única estimación de altura combinada.

Detalles de la integración:

- El predictor de Detectron2 se carga una sola vez al arrancar el servidor, con un umbral
  de confianza bajo fijo; el umbral real del usuario se aplica después, filtrando las
  instancias devueltas — así no es necesario reconstruir el modelo cada vez que cambia el
  control deslizante.
- La salida se normaliza al mismo formato que usa el motor YOLO (conteo por clase, imagen
  anotada, estimación de pisos/altura, coordenadas) para que el resto del pipeline no
  necesite distinguir de qué motor provino cada resultado.
- Si Detectron2 no está instalado o no se encuentran los pesos, el servidor arranca con
  normalidad y el selector de motor simplemente no aparece
  (`GET /tipos-disponibles` expone `"detectron2_disponible": true/false`).

### Estimación de altura y número de pisos

El número de pisos se estima agrupando las cajas detectadas de clase `window` por posición
vertical (cajas que caen en la misma fila, dentro de un margen relativo al alto promedio de
ventana). La altura aproximada se calcula como `número_de_pisos × 3.0 m`.

Esta estimación aplica a cualquier categoría con clase `window` en su taxonomía (`fachada`,
`ventanas`, `fachada_general`). Cuando se analiza una imagen con varias de esas categorías a
la vez, el sistema no suma las ventanas de todas — para evitar contar el mismo hueco más de
una vez, se elige el resultado del modelo más confiable disponible, con esta prioridad:
`ventanas` (especializado) > `fachada` (estructura completa) > `fachada_general` (más
clases, más ruido para esta estimación en particular). El resultado se expone como
`altura_estimada` en la respuesta, además de que cada categoría individual sigue
reportando su propia estimación.

### Daños y deterioro

El modelo de fachadas fusionado incluye 10 clases de daño (`crack`, `ac_bracket_corrosion`,
`concrete_spalling`, `exposed_reinforcement`, `peeling_plaster`, `tile_detachment`,
`corrosion`, `delamination`, `dirty_mold`, `paint_defect`); el modelo `danos`, entrenado por
separado, busca las mismas 10. Cada respuesta separa estas clases del resto:

```json
{
  "conteo_clases": {"window": 4, "floor": 2, "crack": 3, "concrete_spalling": 1},
  "danos_detectados": {"crack": 3, "concrete_spalling": 1},
  "total_danos": 4
}
```

Cuando la categoría es `fachada`, `total_danos` se guarda en la base de datos y se suma al
contador global de daños detectados.

### Endpoints principales

| Endpoint | Descripción |
|---|---|
| `POST /detectar` | Sube una imagen + uno o varios campos `tipo` (+ opcional `conf`, 0.05–0.95). Devuelve un resultado por categoría solicitada más la estimación de altura combinada. |
| `POST /batch/start` | Procesamiento por lote sobre una carpeta del servidor. Acepta `{"folder", "tipos", "conf"}`. |
| `GET /batch/progress` | Progreso del lote en curso: archivo actual, modelo en ejecución, porcentaje, y una bitácora de las últimas líneas de actividad. |
| `GET /batch/results` / `POST /batch/cancel` | Resultados acumulados del lote / cancela el procesamiento en curso. |
| `GET /tipos-disponibles` | Categorías de detección disponibles. |
| `GET /stats` / `GET /db-stats` | Métricas para el dashboard. |
| `GET /exportar-pdf` | Descarga el reporte PDF (ver [sección 3](#3-exportación-de-resultados-en-pdf)). |
| `GET /trafico-manzanas` | Congestión vial promedio por manzana (ver [sección 4](#4-tráfico-vial-factor-de-riesgo-por-manzana)). |
| `GET /riesgo-por-manzana` | Score de riesgo combinado por manzana (ver [sección 5](#5-score-de-riesgo-por-manzana)). |

---

## 3. Exportación de resultados en PDF

`GET /exportar-pdf` genera, con `reportlab` y `matplotlib`, un reporte PDF que incluye:

1. Métricas globales (edificios analizados, ventanas detectadas, daños detectados,
   registros en base de datos, manzanas cubiertas).
2. Distribución de edificios por número de pisos.
3. Tabla de daños y deterioro detectados, agregada por tipo.
4. Tabla de congestión vial por manzana (requiere haber corrido `trafico_tomtom.py`).
5. Tabla de datos agregados por manzana: promedio y máximo de pisos, ventanas totales,
   fotos analizadas y altura promedio (hasta 40 manzanas, ordenadas por cobertura).

---

## 4. Tráfico vial (factor de riesgo por manzana)

La congestión vial se recolecta con **TomTom Traffic Flow API** (nivel gratuito: 2,500
llamadas/día, sin requerir tarjeta de crédito).

```bash
export TOMTOM_API_KEY="tu_key_aqui"   # gratis en https://developer.tomtom.com/
python3 trafico_tomtom.py             # recolecta una vez
python3 trafico_tomtom.py --loop      # recolecta cada 30 minutos indefinidamente
```

El script toma las calles de referencia listadas en `coordenadas_trafico.xlsx`, consulta la
velocidad actual frente a la velocidad de flujo libre en el punto medio de cada una,
calcula un porcentaje de congestión, identifica en qué manzana cae (usando
`hipodromo_manzanas.geojson`) y guarda los resultados en la tabla `trafico_calles` de
`detecciones.db` — junto a los datos de detección de edificios, lista para combinarse como
factor de riesgo adicional. `GET /trafico-manzanas` expone el promedio agregado.

**Nota de seguridad:** la clave de API debe pasarse por variable de entorno
(`TOMTOM_API_KEY`); no debe escribirse en ningún archivo del repositorio.

### Retención de datos

La recolección es manual (se ejecuta cuando se desee); no corre como servicio automático.
Cada corrida purga automáticamente lecturas con más de 90 días de antigüedad
(`RETENCION_DIAS` en `trafico_tomtom.py`). Usa `python3 trafico_tomtom.py --vacuum`
periódicamente para compactar la base de datos tras acumular purgas.

### Índices de base de datos

`servidor_deteccion.py` crea, si no existen, índices en `detecciones(cvegeo, fecha)` y
`trafico_calles(cvegeo, fecha)` al arrancar — son las columnas más consultadas (agregación
por manzana, purga por fecha). `trafico_tomtom.py` crea los mismos índices de forma
independiente por si se ejecuta antes de levantar el servidor.

---

## 5. Score de riesgo por manzana

`GET /riesgo-por-manzana` combina, para cada manzana con datos disponibles, tres señales
normalizadas a [0, 1] en un score ponderado:

| Señal | Peso | Normalización |
|---|---|---|
| Daños detectados, ponderados por severidad | 40% | `min(1, daños_ponderados / 10) × confianza` |
| Congestión vial, priorizando horas pico | 30% | ya viene en [0, 1] |
| Altura promedio (pisos) | 30% | `min(1, pisos_promedio / 10) × confianza` |

Es una priorización relativa entre manzanas, no un análisis de riesgo formal. Los pesos y
umbrales de normalización son constantes configurables al inicio del endpoint en
`servidor_deteccion.py` (`PESO_DANOS`, `PESO_CONGESTION`, `PESO_ALTURA`,
`DANOS_NORMALIZACION`, `PISOS_NORMALIZACION`).

**Ponderación por severidad de daño.** Las 10 clases de daño no pesan igual: fallas
estructurales (`crack`, `concrete_spalling`, `exposed_reinforcement`, `corrosion`) pesan
×1.0; deterioro de acabados (`peeling_plaster`, `tile_detachment`, `ac_bracket_corrosion`,
`delamination`) pesa ×0.5; deterioro estético (`paint_defect`, `dirty_mold`) pesa ×0.1. La
respuesta incluye tanto el conteo crudo (`total_danos`) como el ponderado
(`danos_ponderados`).

**Factor de confianza por volumen de muestra.** El score de daños y de altura se multiplica
por `1 - e^(-0.5 × num_fotos)`: con 1 foto la confianza es de ~39%, con 3 fotos ~78%, con 5
~92%, con 10 o más ya es prácticamente 100%. Esto evita que una sola foto con un hallazgo
severo pese lo mismo que 50 fotos que confirman el mismo nivel de deterioro. La congestión
no se amortigua de esta forma porque no depende de `num_fotos`. El campo `confianza` (0–1)
viene incluido en la respuesta de cada manzana.

**Congestión en horas pico.** Se prioriza el promedio de lecturas realizadas en horas pico
(`HORAS_PICO`, por defecto 8–10h y 18–20h), cuando el tráfico es un factor real para
evacuación y acceso de servicios de emergencia; si una manzana todavía no tiene lecturas en
esas franjas, se usa como respaldo el promedio de todas las horas. El campo
`congestion_hora_pico` indica cuál de los dos se utilizó.

### Visualización

La pestaña "Riesgo por manzana" del frontend consume este endpoint junto con
`GET /manzanas-geojson` y muestra:

- Un mapa (Leaflet, tiles de OpenStreetMap) con cada manzana coloreada por score: verde
  (bajo), amarillo (medio), rojo (alto). Al hacer clic aparece un popup con barras
  horizontales por señal (riesgo global, estructura, deterioro, evacuación) y una nota de
  confianza.
- Una tabla ordenada de mayor a menor riesgo con las mismas señales.

### Consideraciones sobre la cobertura de datos

Es un análisis agregado por manzana — no depende de una imagen específica en el momento de
la consulta, sino de todo lo acumulado en `detecciones.db`:

- Solo la categoría **Fachadas** guarda en base de datos; analizar con otras categorías no
  modifica el mapa de riesgo.
- Una fotografía solo se ubica en una manzana si su nombre de archivo codifica coordenadas
  (patrón `lat_lon.ext`, por ejemplo `19.4102_-99.1684.jpg`) y esas coordenadas caen dentro
  de un polígono del GeoJSON de manzanas.
- Las manzanas sin ninguna detección ni lectura de tráfico se muestran en gris — no
  significa riesgo cero, sino ausencia de datos.
- Una manzana con datos de tráfico pero sin fotografías de fachada obtiene su score solo de
  la congestión (con confianza 0 en las otras dos señales), lo que típicamente resulta en
  un score bajo.

---

## 6. Entrenamiento encadenado sin supervisión

`entrenamiento/entrenar_todo.py` entrena los 7 modelos en secuencia, cada uno con las 3
GPUs completas, sin requerir intervención manual:

```bash
nohup python3 entrenamiento/entrenar_todo.py > entrenamiento/log_entrenar_todo.txt 2>&1 &
```

Épocas por modelo: 200 para Techos, 500 para el resto — ya configuradas dentro de cada
script individual. Si un entrenamiento falla, el orquestador registra el error y continúa
con el siguiente en vez de detener todo el proceso.

**Reanudar desde un punto específico.** Para continuar la cola sin repetir modelos que ya
se entrenaron por separado:

```bash
python3 entrenamiento/entrenar_todo.py --desde ventanas
```

Esto salta todo lo que esté antes de `ventanas` en el plan (se marca como ya completado) y,
antes de arrancar, espera a que no haya otro `entrenar_*.py` corriendo (mediante un sistema
de candados basado en PID, para no competir por las mismas GPUs con un entrenamiento ya en
curso).

### Sin interrupción anticipada (EarlyStopping)

Ultralytics detiene el entrenamiento antes de tiempo por defecto si no hay mejora de mAP en
100 épocas seguidas. Los 7 scripts corren con `patience=0` (interrupción anticipada
desactivada) para que el número de épocas configurado se cumpla siempre.

Cada script revisa además si ya existe un modelo entrenado previamente para esa categoría
(`runs/detect/entrenamiento_<id>/weights/best.pt`); si existe, continúa entrenando desde
esos pesos en lugar de partir de cero, y la corrida anterior se conserva como respaldo en
una carpeta `..._previo/`. Esto solo aplica a nuevas ejecuciones del script — no reanuda
automáticamente un proceso que ya esté corriendo en ese momento.

### Monitoreo del progreso

Además de los archivos de log y `results.csv` que genera Ultralytics por época,
`entrenar_todo.py` mantiene `entrenamiento/estado_entrenamiento.json` con el estado de cada
modelo (pendiente, entrenando, listo o fallido). Para verlo sin usar la terminal:

```bash
python3 entrenamiento/monitor_entrenamiento.py
```

La interfaz gráfica (Tkinter, sin dependencias adicionales) muestra el progreso dentro de
la época actual y el progreso total, una tabla con el estado de los 7 modelos y sus
métricas más recientes (mAP50, mAP50-95, box_loss), y un registro de eventos. Se actualiza
automáticamente cada 3 segundos y puede abrirse o cerrarse en cualquier momento sin afectar
el entrenamiento, que continúa en segundo plano de forma independiente.

---

## Interfaz web

1. Abre `http://127.0.0.1:3005` (o el puerto configurado).
2. En la pestaña **Analizar**, carga imágenes de alguna de estas tres formas:
   - **Clic** para abrir el selector de archivos (Ctrl/Cmd + clic para elegir varias).
   - **Cargar carpeta** para seleccionar una carpeta completa del sistema de archivos.
   - **Arrastrar y soltar** uno o varios archivos, o una carpeta completa.

   También existe la opción de procesar una ruta ya existente en el disco del servidor
   (útil para carpetas grandes que no conviene subir por el navegador), disponible tras el
   enlace "procesar una ruta ya en el servidor".
3. Marca una o varias categorías (Fachadas, Techos, Ventanas, Fachada general, Daños,
   Señales, Calles), o usa "Analizar todo". Al marcar Ventanas aparece el selector de motor
   (YOLO / Detectron2 / ambos) si Detectron2 está disponible. Ajusta el control de
   confianza mínima según se necesiten más o menos detecciones.
4. Haz clic en **Analizar**. Con una sola imagen se muestra el detalle completo: estimación
   de altura combinada, y una tarjeta de resultado por categoría con su imagen anotada,
   conteo por clase y datos geoespaciales. Con varias imágenes se muestra una barra de
   progreso y una tabla de resultados agregados.
5. Usa **Exportar reporte PDF** para descargar el informe completo, o **Reiniciar stats**
   para vaciar la base de datos.
6. La tabla de resultados por lote incluye, por archivo: categorías analizadas, total de
   detecciones (con desglose por clase al pasar el cursor), pisos y altura estimada, y una
   columna de daños resaltada cuando hay detecciones. Las columnas que no aplican a la
   categoría seleccionada (por ejemplo, altura con solo Techos marcado) se muestran vacías
   en vez de con un cero engañoso. Cuando la altura proviene de la categoría Ventanas, se
   incluye una etiqueta indicando qué motor (YOLO / Detectron2) generó ese resultado.
7. Durante el procesamiento por lote se muestra el progreso en tiempo real: qué imagen y
   qué modelo se está ejecutando en ese momento, más una bitácora de actividad reciente.
8. La pestaña **Riesgo por manzana** muestra el mapa y la tabla de score combinado
   (sección 5), recalculado con los datos más recientes disponibles.

---

## Solución de problemas

**El puerto ya está en uso.** El puerto es configurable — no es necesario liberar el que
esté ocupado:

```bash
python3 servidor_deteccion.py 3005          # puerto como argumento
PUERTO=3005 python3 servidor_deteccion.py   # puerto como variable de entorno
```

Si de todos modos se quiere liberar un puerto específico:

```bash
lsof -ti:3000 | xargs -r kill -9
```

**No hay modelo entrenado para una categoría.** El servidor usa automáticamente el modelo
COCO preentrenado de Ultralytics como respaldo hasta que se entrene uno propio. Ejecuta
`organizar_datasets.py` y luego `entrenamiento/entrenar_todo.py`, o un
`entrenamiento/entrenar_<categoría>.py` individual.

**Cambiar GPUs, tamaño de lote o número de épocas.** Edita las variables correspondientes
(`device`, `batch`, `epochs`) directamente en el `entrenamiento/entrenar_<categoría>.py`
que se quiera ajustar.

---

## Créditos y tecnologías

- [**Ultralytics YOLO11**](https://github.com/ultralytics/ultralytics) — detección de objetos
- [**Detectron2**](https://github.com/facebookresearch/detectron2) — motor alterno de detección de ventanas
- **Flask** — servidor web
- **OpenCV / Pillow** — procesamiento de imágenes
- **Matplotlib** — gráficas del dashboard y del reporte PDF
- **ReportLab** — generación del reporte PDF
- **Shapely** — geolocalización de detecciones por manzana
- **TomTom Traffic API** — datos de congestión vial
- **Leaflet / OpenStreetMap** — visualización del mapa de riesgo

---

## Historial de cambios

- **v3.10** — Publicación en GitHub con los 7 modelos entrenados incluidos vía Git LFS
  (`runs/detect/`), listos para usar sin necesidad de entrenar nada de cero.
- **v3.9** — Reorganización del repositorio de cara a su publicación: separación clara
  entre lo necesario para ejecutar el proyecto y material histórico/exploratorio; migración
  del dataset de Techos a `datasets/` junto con el resto de las categorías; `.gitignore`
  para datasets, pesos y artefactos generados.
- **v3.8** — Datasets adicionales para Señales y Calles (señalamiento vial en dos idiomas,
  vehículos y peatones).
- **v3.7** — Rediseño del score de riesgo por manzana: ponderación de daños por severidad,
  factor de confianza por volumen de muestra, congestión priorizando horas pico. Popup del
  mapa rediseñado con indicadores visuales por señal. Bitácora en tiempo real durante el
  procesamiento por lote. Indicador de motor de detección en la tabla de resultados.
- **v3.6** — Carga de archivos unificada: selección múltiple, carga de carpeta completa, y
  arrastrar y soltar.
- **v3.5** — Motor Detectron2 como alternativa para la detección de ventanas; control de
  confianza mínima ajustable desde la interfaz.
- **v3.4** — Incorporación de `CMP_facade_DB_base` al modelo de fachada general y a la
  fusión combinada; reescritura de la estimación de altura por agrupación de filas de
  ventana.
- **v3.3** — Datasets adicionales de árboles, daños y techos; pipeline de entrenamiento
  encadenado con interfaz gráfica de progreso.
- **v3.2** — Reorganización del proyecto en carpetas; score de riesgo con mapa;
  optimización de la base de datos.
- **v3.1** — Menú de selección múltiple de categorías; siete modelos con respaldo a pesos
  preentrenados.
