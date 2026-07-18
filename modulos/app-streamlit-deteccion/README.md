# 🏙️ IA Mapeo Analítico — La Condesa

Detección automática de edificios sobre imágenes satelitales de La Condesa (CDMX) usando YOLOv8-seg, con exportación a GeoJSON y visualización en mapa (Folium), pensado para integrarse a flujos de trabajo tipo OpenStreetMap/HOT.

## Características

- Detección y segmentación de edificios sobre imágenes satelitales de alta resolución
- Procesamiento por tiles (para imágenes grandes sin saturar la memoria de la GPU)
- Cálculo de área estimada por edificio (m²)
- Exportación a GeoJSON compatible con QGIS, uMap y JOSM
- Visualización interactiva en mapa (Folium)
- Reporte descargable en CSV

> Este es un **módulo** del repositorio unificado de la Plataforma de Análisis Urbano —
> clona el repositorio completo (con Git LFS) para tener los pesos disponibles.

## Instalación

```bash
cd modulos/app-streamlit-deteccion

python3 -m venv venv
source venv/bin/activate   # En Windows: venv\Scripts\activate

pip install -r requirements.txt
```

**Opcional — motor Detectron2 para la página de Ventanas.** No está en PyPI:

```bash
pip install git+https://github.com/facebookresearch/detectron2.git
```

## Modelos entrenados

Los pesos viven en el almacén único del proyecto, `config/checkpoints/` (raíz del
repositorio, vía Git LFS) — las páginas los cargan desde ahí automáticamente:

- Satélite: `config/checkpoints/satelite_yolov8xl_seg.pt`
- Ventanas: `config/checkpoints/comparador/` (YOLOv8/v11 + Detectron2)

## Uso

```bash
streamlit run Inicio.py
```

Abre `http://localhost:8501` en tu navegador.

## Metodología

Este proyecto sigue el pipeline: `imagen satelital → tiles → detección YOLOv8-seg → coordenadas → GeoJSON → mapa`, como parte de un proyecto de investigación sobre detección de características de mapa relevantes para la comunidad, orientado a flujos de trabajo de OpenStreetMap/Humanitarian OpenStreetMap Team (HOT).

## Autor

Angel Vargas — Tecnológico Nacional de México, Campus Apatzingán
