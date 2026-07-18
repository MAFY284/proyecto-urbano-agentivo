# Plataforma de Análisis Urbano — Sistema Multi-Agente

**Multi-Agent Deep Learning System for Urban Seismic Risk Assessment Integrating Facade,
Satellite and Mobility Analytics in Mexico City**

![Python](https://img.shields.io/badge/python-3.12-blue?logo=python&logoColor=white)
![YOLO11](https://img.shields.io/badge/Ultralytics-YOLO11-111f68)
![Detectron2](https://img.shields.io/badge/Meta-Detectron2-0467df)
![SAM3](https://img.shields.io/badge/Meta-SAM3-8b5cf6)
![Flask](https://img.shields.io/badge/Flask-servidor%20web-000?logo=flask)
![Git LFS](https://img.shields.io/badge/git--lfs-pesos%20versionados-f64935?logo=git-lfs&logoColor=white)

Proyecto integrador del **Verano Científico (Programa Delfín)**: sistema multi-agente de
visión por computadora y análisis espacial para la **priorización del riesgo urbano por
manzana** (estructural, vial y sísmico) en la colonia Hipódromo-Condesa, Ciudad de México.
Detección de fachadas y daños con YOLO11, segmentación satelital YOLOv8-XL, comparación
multi-modelo de ventanas (incluida SAM3 zero-shot), tráfico vial TomTom y análisis SIG de
movilidad — todo operado desde un solo frontend web.

## Características

| Módulo del sistema | Descripción |
|---|---|
| 🗺️ **Centro de mando** | Mapa coroplético de riesgo por manzana (CVEGEO), KPIs globales, pipeline multi-agente en un clic, exportación PDF y GeoJSON |
| 🏢 **Análisis de fachadas** | Pool de 7 modelos YOLO11 (fachadas, techos, ventanas, daños, señales, calles) con estimación de pisos y daños ponderados por severidad |
| 🔍 **Verificación autónoma** | Si la confianza de ventanas < 65 %, el Agente de Visión contrasta automáticamente con Detectron2 o SAM3 y corrige el conteo |
| 🛰️ **Segmentación satelital** | YOLOv8-XL por mosaicos con traslape configurable, área en m², georreferenciación y GeoJSON para QGIS/uMap |
| 🧪 **Laboratorio** | Benchmarking lado a lado de 5 motores de ventanas: YOLOv8, YOLOv8-seg, YOLOv11-seg, Detectron2 y SAM3 (prompt de texto) |
| 📈 **Simulador de crisis** | Proyección del índice de riesgo 24 h al variar día, hora y densidad vial sobre el perfil histórico de congestión |
| 🚦 **Tráfico vial** | TomTom Traffic API con *fallback* automático a perfiles históricos 7×24 |

## Índice de riesgo

```
R = 0.4·min(1, D/10)·c + 0.3·G + 0.3·min(1, P/10)·c ,   c = 1 − e^(−0.5·n)
```

`D` = daños ponderados por severidad (estructural ×1.0, acabados ×0.5, estético ×0.1) ·
`G` = congestión en hora pico · `P` = pisos promedio · `n` = muestras por manzana.
Todos los pesos son configurables en [config/settings.yaml](config/settings.yaml).

## Estructura del proyecto

```
├── index.html               # Frontend general (opera todo el sistema)
├── styles.css               #   · estilos
├── app.js                   #   · lógica del cliente
├── servidor.py              # Servidor Flask: API + frontend
├── main.py                  # CLI multi-agente
├── install.sh               # Instalador (venv principal + Detectron2 + SAM3 opcionales)
├── requirements.txt
├── config/                  # settings.yaml · checkpoints/ (TODOS los pesos, vía LFS)
│                            #   · geojson de manzanas · CSV SAM3 · curvas de congestión
├── database/detecciones.db  # Detecciones + tráfico + riesgo (clave espacial CVEGEO)
├── docs/                    # Reporte técnico formal (DOCX) + figuras
├── src/
│   ├── agents/              # Orquestador · SIG · Visión (oráculo) · Riesgo
│   ├── tools/               # fachada · satelite · comparador · trafico · gis_utils
│   └── dashboard/app.py     # Interfaz alterna en Streamlit
├── tests/                   # Prueba end-to-end (sin GPU, ~5 s)
└── modulos/                 # Los 4 proyectos individuales que dieron origen al sistema
    ├── analisis-fachada-techo-yolo/    # Mario  — entrenamiento de los 7 modelos YOLO11
    ├── app-streamlit-deteccion/        # Angel  — demo satélite + ventanas (Streamlit)
    ├── deteccion-ventanas-streamlit/   # Ricardo — comparador de 5 modelos (Streamlit)
    └── proyecto-delfin/                # Evelyn — movilidad y riesgo sísmico (QGIS)
```

**Todos los pesos entrenados viven en un solo lugar: `config/checkpoints/`** (≈1.2 GB vía
Git LFS). Las apps de `modulos/` los cargan desde ahí — no hay copias duplicadas.

## Equipo

| Integrante | Aportación | Módulo de origen |
|---|---|---|
| **Mario** | Detección de fachadas, techos y daños — 7 modelos YOLO11 + score de riesgo | [`modulos/analisis-fachada-techo-yolo/`](modulos/analisis-fachada-techo-yolo/) |
| **Ricardo** | Comparador multi-modelo de ventanas (YOLOv8, Detectron2, SAM3) | [`modulos/deteccion-ventanas-streamlit/`](modulos/deteccion-ventanas-streamlit/) |
| **Angel** | Segmentación satelital de edificios (YOLOv8-seg) | [`modulos/app-streamlit-deteccion/`](modulos/app-streamlit-deteccion/) |
| **Evelyn** | Análisis espacial de movilidad y riesgo sísmico (QGIS + notebooks) | [`modulos/proyecto-delfin/`](modulos/proyecto-delfin/) |

## Requisitos

- **Python 3.12** (versión de referencia; el motor SAM3 la exige)
- **Git** y **[Git LFS](https://git-lfs.com/)** — los pesos (`*.pt`, `*.pth`) se versionan
  con LFS; instálalo **antes** de clonar o corre `git lfs pull` después
- **GPU NVIDIA con CUDA** recomendada (funciona en CPU, más lento)

## Instalación

```bash
git lfs install
git clone <URL_DEL_REPOSITORIO>
cd <nombre-del-repositorio>

bash install.sh          # entorno principal (venv + requirements.txt)
bash install.sh --todo   # + Detectron2 compilado + venv aislado de SAM3 (opcional)
```

**SAM3 (opcional).** El checkpoint `facebook/sam3` es un repositorio *gated* de Hugging
Face: solicita acceso en <https://huggingface.co/facebook/sam3> y autentícate con
`src/tools/env_sam3/bin/huggingface-cli login` (o exporta `HF_TOKEN`). Sin ello, el motor
simplemente se oculta del selector.

**TomTom (opcional).** `export TOMTOM_API_KEY="tu_key"` para congestión en tiempo real;
sin key el sistema usa el perfil histórico automáticamente.

## Uso

```bash
source venv/bin/activate
python3 servidor.py                    # frontend general → http://127.0.0.1:3005
python3 main.py --help                 # CLI: analizar, riesgo, trafico, manzanas
streamlit run src/dashboard/app.py     # interfaz alterna en Streamlit
python3 tests/test_flujo_completo.py   # prueba end-to-end (sin GPU, ~5 s)
```

Los módulos individuales también pueden ejecutarse por separado (cada uno tiene su propio
README):

```bash
cd modulos/deteccion-ventanas-streamlit && streamlit run app.py
cd modulos/app-streamlit-deteccion && streamlit run Inicio.py
```

## Reentrenamiento

Los scripts de entrenamiento de los 7 modelos YOLO11 y las herramientas de organización de
datasets están en [`modulos/analisis-fachada-techo-yolo/`](modulos/analisis-fachada-techo-yolo/)
(ver su README). Los datasets crudos (decenas de GB) no se incluyen en el repositorio;
solo hacen falta para reentrenar. Los pesos ya entrenados vienen incluidos en
`config/checkpoints/`, listos para usar.

## Nota sobre Git LFS

Los pesos suman **~1.2 GB**, por encima de la cuota gratuita de GitHub (1 GB de
almacenamiento LFS / 1 GB de banda al mes). Si `git push` rechaza los objetos LFS por
cuota: compra un [Data Pack](https://docs.github.com/es/billing/managing-billing-for-git-large-file-storage/about-billing-for-git-large-file-storage)
(US$5/mes por 50 GB) o mueve `config/checkpoints/comparador/detectron2_best.pth` (315 MB)
a GitHub Releases / Hugging Face Hub y documenta el enlace de descarga aquí.

## Documentación

- 📄 [Reporte técnico formal (DOCX)](docs/Reporte_Sistema_Urbano_Agentivo.docx) — artículo
  completo con metodología, resultados y figuras.
- Resultados: 100/107 manzanas evaluadas; 1 745 de 3 727 edificios SAM3 georreferenciados;
  confianza estadística 98–100 % en el componente de exposición.

## Créditos y tecnologías

Construido sobre [Ultralytics YOLO](https://github.com/ultralytics/ultralytics),
[Detectron2](https://github.com/facebookresearch/detectron2),
[SAM3](https://github.com/facebookresearch/sam3), Flask, Streamlit, TomTom Traffic API,
Leaflet/CARTO, QGIS/GeoPandas y el Marco Geoestadístico del INEGI.

Proyecto colaborativo del Verano Científico (Programa Delfín).
