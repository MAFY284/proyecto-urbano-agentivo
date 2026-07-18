# 🪟 Detección de Ventanas

### Comparador multi-modelo de detección y segmentación de ventanas en fachadas, sobre Streamlit

Sube una o varias imágenes, elige entre **YOLOv8, YOLOv8-seg, YOLOv11-seg, Detectron2** o **SAM3 (zero-shot por texto)**, y compara resultados de detección lado a lado en tu navegador.

[![Demo en línea](https://img.shields.io/badge/demo-en%20línea-brightgreen?logo=streamlit&logoColor=white)](https://deteccion-ventanas-app-gfhaveom9ybv3dd7s558dn.streamlit.app/)
![Python](https://img.shields.io/badge/python-3.10+-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/streamlit-1.59-FF4B4B?logo=streamlit&logoColor=white)
![Git LFS](https://img.shields.io/badge/git--lfs-pesos%20versionados-orange?logo=git-lfs&logoColor=white)
[![Licencia](https://img.shields.io/badge/licencia-[TIPO_DE_LICENCIA]-lightgrey)](LICENSE)
![Estado](https://img.shields.io/badge/estado-en%20desarrollo-yellow)

---

## 📑 Tabla de contenidos

- [Características principales](#-características-principales)
- [Arquitectura y modelos](#-arquitectura-y-modelos)
- [Requisitos previos](#-requisitos-previos)
- [Instalación y configuración](#-instalación-y-configuración)
- [Uso](#-uso)
- [Estructura del proyecto](#-estructura-del-proyecto)
- [Notas sobre despliegue](#-notas-sobre-despliegue)
- [Roadmap](#-roadmap)
- [Créditos y agradecimientos](#-créditos-y-agradecimientos)
- [Licencia](#-licencia)

---

## ✨ Características principales

- 🔍 **5 modelos intercambiables** desde la misma interfaz: YOLOv8 (detección), YOLOv8-seg y YOLOv11-seg (segmentación), Detectron2 (Faster R-CNN) y SAM3 (segmentación zero-shot guiada por texto).
- 🖼️ **Carga múltiple de imágenes** — procesa un lote completo en una sola corrida.
- 🎚️ **Umbral de confianza ajustable** en tiempo real mediante un slider.
- 🆚 **Comparación visual lado a lado**: imagen original vs. imagen anotada, para cada archivo subido.
- 🧠 **Prompts de texto libres para SAM3** (por defecto `"window"`), ideal para detección zero-shot sin reentrenar.
- ⚡ **Modelos cacheados** con `st.cache_resource`: cada modelo se carga una sola vez por sesión.
- 🧩 **Detección de entorno automática**: si Detectron2 o el entorno de SAM3 no están disponibles, esas opciones simplemente se ocultan del selector (sin romper la app).
- ☁️ **Listo para Streamlit Community Cloud**, con manejo explícito de dependencias pesadas/incompatibles con builds sin GPU.
- 📦 **Pesos versionados con Git LFS** (`*.pt`, `*.pth`).

---

## 🏗️ Arquitectura y modelos

| Modelo | Tipo | Motor | Disponibilidad |
|---|---|---|---|
| `YOLOv8 (detección)` | Cajas delimitadoras | `ultralytics` | Siempre (incluido en `requirements.txt`) |
| `YOLOv8-seg (segmentación)` | Máscaras de instancia | `ultralytics` | Siempre |
| `YOLOv11-seg (segmentación)` | Máscaras de instancia | `ultralytics` | Siempre |
| `Detectron2 (detección)` | Cajas (Faster R-CNN R50-FPN) | `detectron2` | Solo si está instalado localmente |
| `SAM3 (zero-shot, texto)` | Cajas por prompt de texto | subproceso en `venv_sam3` | Solo si existe el venv dedicado |

**Por qué SAM3 corre como subproceso:** SAM3 requiere Python 3.12 + PyTorch 2.7, incompatible con el entorno principal (Python 3.10 + PyTorch usado por Detectron2/YOLO). `app.py` invoca [`sam3_worker.py`](sam3_worker.py) como proceso independiente con el intérprete de ese venv aparte, y se comunican vía JSON por `stdout`.

---

## ✅ Requisitos previos

- **Python 3.10+**
- **pip** y **venv** (o el gestor de entornos que prefieras: `conda`, `pyenv`, etc.)
- **Git** con **[Git LFS](https://git-lfs.com/)** instalado (los pesos `.pt`/`.pth` se versionan con LFS)
- *(Opcional, solo para Detectron2 local)* Compilador de C++ y headers de Python para compilar Detectron2 desde su repo
- *(Opcional, solo para SAM3)* Un entorno virtual separado con Python 3.12 y PyTorch 2.7, con el paquete `sam3` instalado
- *(Opcional)* GPU NVIDIA + CUDA para inferencia acelerada (la app funciona en CPU, pero más lento)

---

## 🚀 Instalación y configuración

### 1. Clonar el repositorio

> Este es un **módulo** del repositorio unificado de la Plataforma de Análisis Urbano.
> Los pesos viven en el almacén único `config/checkpoints/comparador/` (raíz del
> repositorio, vía Git LFS) — la app los carga desde ahí automáticamente.

```bash
git lfs install
git clone [URL_DEL_REPOSITORIO_UNIFICADO]
cd <repo>/modulos/deteccion-ventanas-streamlit
```

### 2. Crear y activar un entorno virtual

```bash
python -m venv venv
source venv/bin/activate      # En Windows: venv\Scripts\activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. (Opcional) Instalar Detectron2

Detectron2 **no está incluido** en `requirements.txt` porque no está publicado en PyPI y su build falla en entornos sin GPU (como Streamlit Community Cloud). Si quieres usarlo en local:

```bash
pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

### 5. (Opcional) Configurar el entorno de SAM3

SAM3 vive en un entorno virtual completamente aparte por incompatibilidad de versiones de PyTorch:

```bash
python3.12 -m venv venv_sam3
source venv_sam3/bin/activate
pip install torch==2.7.* torchvision  # ajusta según tu CUDA
pip install [PAQUETE_O_REPO_DE_SAM3]
```

Luego actualiza la ruta al intérprete de ese venv en [`app.py`](app.py):

```python
VENV_SAM3_PYTHON = "[RUTA_A_TU_VENV_SAM3]/bin/python3"
```

### 6. Verificar los pesos de los modelos

Confirma que `config/checkpoints/comparador/` (raíz del repositorio) contenga los
`.pt` / `.pth` reales descargados vía Git LFS y no punteros de texto:

```bash
git lfs pull
ls -lh ../../config/checkpoints/comparador/
```

---

## 💻 Uso

### Ejecutar la app localmente

```bash
streamlit run app.py
```

Abre tu navegador en:

```
http://localhost:8501
```

### Flujo de uso

1. En la barra lateral, elige el **modelo** a usar.
2. Ajusta el **umbral de confianza** con el slider (`0.0` – `1.0`).
3. Si eliges **SAM3**, escribe el **prompt de texto** (por defecto `"window"`).
4. Sube una o varias imágenes (`.jpg`, `.jpeg`, `.png`) en el uploader principal.
5. Revisa el resultado: por cada imagen se muestra la **original** junto a la **anotada**, con el conteo de ventanas detectadas.

```
┌─────────────────────────────┬─────────────────────────────┐
│   Original: fachada_01.jpg  │  Detecciones (7 ventanas)    │
│   ┌───────────────────┐     │   ┌───────────────────┐     │
│   │                   │     │   │ [■] [■]  [■]      │     │
│   │      🏢           │     │   │      🏢  [■]      │     │
│   │                   │     │   │ [■]      [■] [■]  │     │
│   └───────────────────┘     │   └───────────────────┘     │
└─────────────────────────────┴─────────────────────────────┘
```

### Ejemplo: usar `sam3_worker.py` de forma independiente

También puedes invocar el worker de SAM3 directamente por línea de comandos, sin pasar por Streamlit:

```bash
/ruta/a/venv_sam3/bin/python3 sam3_worker.py "window" 0.3 imagen1.jpg imagen2.jpg
```

Salida (JSON por `stdout`):

```json
{
  "imagen1.jpg": {
    "cajas": [
      {"box": [120.5, 80.2, 340.1, 260.7], "score": 0.91}
    ]
  },
  "imagen2.jpg": {
    "error": "mensaje de error, si aplica"
  }
}
```

---

## 📖 Documentación de la API / comandos

`app.py` no expone una API REST; es una app monolítica de Streamlit. Las funciones internas relevantes son:

| Función | Descripción |
|---|---|
| `cargar_yolo(path)` | Carga y cachea un modelo YOLO (`ultralytics.YOLO`) |
| `cargar_detectron2(path, num_classes, conf_thresh)` | Construye y cachea un `DefaultPredictor` de Detectron2 |
| `inferir_yolo(modelo, imagen_bgr, conf)` | Corre inferencia YOLO y devuelve imagen anotada + conteo |
| `inferir_detectron2(predictor, imagen_bgr)` | Corre inferencia Detectron2 y devuelve imagen anotada + conteo |
| `inferir_sam3_batch(rutas_imagenes, prompt, conf)` | Ejecuta `sam3_worker.py` como subproceso sobre un lote de imágenes |
| `procesar_imagen(nombre_modelo, conf, archivo_imagen)` | Orquesta carga + inferencia según el modelo elegido |

`sam3_worker.py` (CLI):

```bash
python3 sam3_worker.py <prompt_texto> <umbral_conf> <ruta_img1> [<ruta_img2> ...]
```

---

## 🗂️ Estructura del proyecto

```
modulos/deteccion-ventanas-streamlit/
├── app.py                    # App principal de Streamlit (UI + orquestación de inferencia)
├── sam3_worker.py            # Worker de SAM3, ejecutado como subproceso en su propio venv
├── requirements.txt          # Dependencias del entorno principal
└── README.md

# Pesos entrenados (almacén único del repositorio, vía Git LFS):
../../config/checkpoints/comparador/
├── yolov8_det_best.pt
├── yolov8_seg_best.pt
├── yolov11_seg_best.pt
└── detectron2_best.pth
```

---

## ☁️ Notas sobre despliegue

- La [demo en línea](https://deteccion-ventanas-app-gfhaveom9ybv3dd7s558dn.streamlit.app/) corre en **Streamlit Community Cloud sin Detectron2 ni SAM3**, ya que ese entorno no soporta builds con GPU ni entornos virtuales adicionales.
- La app detecta en tiempo de ejecución si `detectron2` está instalado (`try/except ImportError`) y si el venv de SAM3 existe (`os.path.exists`); si no, oculta esas opciones del selector sin fallar.
- Para desplegar con **todas** las opciones habilitadas, usa un entorno con GPU (VM propia, contenedor Docker, etc.) donde puedas instalar Detectron2 y provisionar el venv de SAM3.

---

## 🗺️ Roadmap

- [ ] [Agregar métricas de evaluación (mAP, IoU) por modelo]
- [ ] [Soporte para exportar resultados en formato COCO/YOLO]
- [ ] [Dockerfile para despliegue reproducible con GPU]
- [ ] [Endpoint de API REST independiente de la UI]
- [ ] [URL del roadmap / tablero de proyecto]

---

## 🙌 Créditos y agradecimientos

- [**Ultralytics YOLO**](https://github.com/ultralytics/ultralytics) — modelos YOLOv8 / YOLOv11.
- [**Detectron2**](https://github.com/facebookresearch/detectron2) (Meta AI Research) — Faster R-CNN.
- [**SAM3**](https://github.com/facebookresearch/segment-anything) (Meta AI) — segmentación zero-shot por texto.
- [**Streamlit**](https://streamlit.io/) — framework de la interfaz web.
- Desarrollado por [TU_NOMBRE_O_EQUIPO].
- Dataset de fachadas/ventanas: [FUENTE_DEL_DATASET].

**Contacto:**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-[TU_USUARIO]-0A66C2?logo=linkedin&logoColor=white)]([URL_LINKEDIN])
[![GitHub](https://img.shields.io/badge/GitHub-[TU_USUARIO]-181717?logo=github&logoColor=white)]([URL_GITHUB])
[![Correo](https://img.shields.io/badge/Email-[TU_CORREO]-D14836?logo=gmail&logoColor=white)](mailto:[TU_CORREO])

---

## 📄 Licencia

Este proyecto está bajo la licencia **[TIPO_DE_LICENCIA, ej. MIT]**. Consulta el archivo [`LICENSE`](LICENSE) para más detalles.


```bash
pip install 'git+https://github.com/facebookresearch/detectron2.git'
```
