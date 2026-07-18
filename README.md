# Análisis Urbano Multi-Agente — Condesa/Hipódromo, CDMX

**Proyecto integrador del Verano Científico (Programa Delfín).** Sistema de visión por
computadora y análisis espacial para priorizar riesgo urbano por manzana (estructural,
vial y sísmico) en las colonias Condesa e Hipódromo, Ciudad de México: detección de
techos, fachadas, ventanas y daños con YOLO11/Detectron2/SAM3, segmentación satelital,
tráfico vial en tiempo real y análisis SIG de movilidad y riesgo sísmico.

Este repositorio consolida el trabajo de los 4 integrantes del equipo en un solo lugar,
listo para clonar y ejecutar.

## Equipo y módulos

| Integrante | Aportación | Carpeta |
|---|---|---|
| **Mario** (yo) | Detección de fachadas, techos y daños — 7 modelos YOLO11 + score de riesgo por manzana | [`analisis-fachada-techo-yolo/`](analisis-fachada-techo-yolo/) |
| **Ricardo** | Comparador multi-modelo de detección de ventanas (YOLOv8, Detectron2, SAM3) | [`deteccion-ventanas-streamlit/`](deteccion-ventanas-streamlit/) |
| **Angel** | Segmentación satelital de edificios (YOLOv8-seg) y demo Streamlit | [`app-streamlit-deteccion/`](app-streamlit-deteccion/) |
| **Evelyn** | Análisis espacial de movilidad y riesgo sísmico (QGIS + notebooks) | [`proyecto-delfin/`](proyecto-delfin/) |
| **Los 4** | **Plataforma integradora** — une los cuatro módulos en un solo sistema multi-agente con dashboard, API y score de riesgo combinado | [`proyecto-urbano-agentivo/`](proyecto-urbano-agentivo/) |

> `proyecto-urbano-agentivo/` es el entregable principal: consolida las cuatro líneas de
> trabajo anteriores en una sola plataforma. Las otras cuatro carpetas son los proyectos
> individuales tal como se desarrollaron — se conservan completas porque cada una tiene su
> propia interfaz y sirve como evidencia independiente del trabajo de cada integrante.

## Estructura del repositorio

```
├── proyecto-urbano-agentivo/       # Plataforma integradora (entregable principal)
├── analisis-fachada-techo-yolo/    # Fachadas, techos, daños — 7 modelos YOLO11 + Flask
├── app-streamlit-deteccion/        # Satélite + ventanas — demo Streamlit
├── deteccion-ventanas-streamlit/   # Comparador de 5 modelos de ventanas — Streamlit
├── proyecto-delfin/                # Movilidad y riesgo sísmico — QGIS + notebooks
├── .gitattributes                  # Reglas de Git LFS (pesos de modelos)
└── .gitignore
```

Cada carpeta es autocontenida: tiene su propio `README.md` con instalación y uso
detallados, y (donde aplica) su propio `requirements.txt`. Este README solo cubre lo
necesario para clonar y preparar el entorno; para correr un módulo en particular entra a
su carpeta y sigue su README.

## Requisitos generales

- **Python 3.12** — versión de referencia para todo el proyecto. La mayoría de los
  módulos también funcionan con Python 3.8+; solo el motor **SAM3** (dentro de
  `deteccion-ventanas-streamlit/` y `proyecto-urbano-agentivo/`) requiere específicamente
  3.12, por lo que se estandarizó esa versión en todo el repositorio.
- **Git** y **[Git LFS](https://git-lfs.com/)** — los pesos entrenados (`*.pt`, `*.pth`,
  ≈1.2 GB en total) se versionan con Git LFS, no como blobs normales.
- **GPU NVIDIA con CUDA** (recomendado) — todo corre en CPU también, pero
  considerablemente más lento para inferencia y entrenamiento.

## Instalación

```bash
# 1) Git LFS (una sola vez por máquina)
git lfs install

# 2) Clonar
git clone <URL_DEL_REPOSITORIO>
cd <nombre-del-repositorio>

# Si ya clonaste sin tener Git LFS instalado, los pesos quedan como punteros de texto:
git lfs pull
```

Cada módulo tiene su propio entorno virtual y dependencias (para no mezclar versiones de
PyTorch/CUDA entre proyectos). Entra a la carpeta que te interese y sigue su README:

```bash
cd proyecto-urbano-agentivo && cat README.md      # plataforma integradora
cd analisis-fachada-techo-yolo && cat README.md
cd app-streamlit-deteccion && cat README.md
cd deteccion-ventanas-streamlit && cat README.md
cd proyecto-delfin && cat README.md
```

Instalación rápida de la plataforma integradora (la forma más directa de ver el sistema
completo funcionando):

```bash
cd proyecto-urbano-agentivo
python3.12 -m venv venv && source venv/bin/activate
bash install.sh              # entorno principal
# bash install.sh --todo     # + Detectron2 + venv aislado de SAM3 (opcional)
python3 servidor.py          # http://127.0.0.1:3005
```

## Nota sobre el tamaño del repositorio y Git LFS

Los pesos entrenados (7 modelos YOLO11 + Detectron2, repetidos entre módulos porque cada
integrante los usa en su propia app) suman **~1.2 GB únicos** vía Git LFS. Esto está muy
cerca de la cuota gratuita de GitHub (1 GB de almacenamiento LFS / 1 GB de banda ancha al
mes). Si al hacer `git push` GitHub rechaza los objetos LFS por cuota:

- Compra un [Data Pack](https://docs.github.com/es/billing/managing-billing-for-git-large-file-storage/about-billing-for-git-large-file-storage) (US$5/mes por 50 GB adicionales), o
- Sube los pesos más pesados (`detectron2_best.pth`, 315 MB) a GitHub Releases o Hugging
  Face Hub y deja solo un enlace de descarga en el README del módulo correspondiente.

## Datos no incluidos

Los datasets crudos y procesados para reentrenar (decenas de GB) **no** están incluidos —
solo hacen falta si se quiere reentrenar desde cero; ver la sección de entrenamiento en
[`analisis-fachada-techo-yolo/README.md`](analisis-fachada-techo-yolo/README.md). Los
modelos ya entrenados sí vienen incluidos, listos para usar sin entrenar nada.

## Créditos y tecnologías

[Ultralytics YOLO11](https://github.com/ultralytics/ultralytics) ·
[Detectron2](https://github.com/facebookresearch/detectron2) ·
[SAM3](https://github.com/facebookresearch/sam3) · Flask · Streamlit ·
[TomTom Traffic API](https://developer.tomtom.com/) · QGIS · GeoPandas · Leaflet/CARTO ·
Marco Geoestadístico del INEGI.

Desarrollado como parte del Verano Científico (Programa Delfín).
