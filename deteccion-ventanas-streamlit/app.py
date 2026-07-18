import os
import json
import subprocess
import tempfile
import streamlit as st
import numpy as np
import cv2
from PIL import Image

# =========================================================
# ⚙️ CONFIGURACIÓN — RUTAS RELATIVAS A ESTA CARPETA
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

YOLOV8_PATH = os.path.join(MODELS_DIR, "yolov8_det_best.pt")
YOLOV8SEG_PATH = os.path.join(MODELS_DIR, "yolov8_seg_best.pt")
YOLOV11SEG_PATH = os.path.join(MODELS_DIR, "yolov11_seg_best.pt")

DETECTRON2_WEIGHTS = os.path.join(MODELS_DIR, "detectron2_best.pth")
DETECTRON2_BASE_CFG = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
DETECTRON2_NUM_CLASSES = 1

MODELOS_DISPONIBLES = {
    "YOLOv8 (detección)": {"tipo": "yolo", "path": YOLOV8_PATH},
    "YOLOv8-seg (segmentación)": {"tipo": "yolo", "path": YOLOV8SEG_PATH},
    "YOLOv11-seg (segmentación)": {"tipo": "yolo", "path": YOLOV11SEG_PATH},
}

try:
    import detectron2  # noqa: F401
    DETECTRON2_DISPONIBLE = True
    MODELOS_DISPONIBLES["Detectron2 (detección)"] = {"tipo": "detectron2", "path": DETECTRON2_WEIGHTS}
except ImportError:
    DETECTRON2_DISPONIBLE = False

# SAM3 vive en un venv aparte (Python 3.12 + torch 2.7, incompatible con
# Detectron2 en este venv), así que se corre como subproceso.
VENV_SAM3_PYTHON = "/home/veranocientifico/Downloads/Resultados/venv_sam3/bin/python3"
SAM3_WORKER_SCRIPT = os.path.join(BASE_DIR, "sam3_worker.py")
SAM3_DISPONIBLE = os.path.exists(VENV_SAM3_PYTHON) and os.path.exists(SAM3_WORKER_SCRIPT)

if SAM3_DISPONIBLE:
    MODELOS_DISPONIBLES["SAM3 (zero-shot, texto)"] = {"tipo": "sam3", "path": None}

st.set_page_config(page_title="Detección de ventanas", layout="wide")


# =========================================================
# 🤖 CARGA DE MODELOS (cacheada: solo se carga una vez)
# =========================================================

@st.cache_resource(show_spinner="Cargando modelo YOLO...")
def cargar_yolo(path):
    from ultralytics import YOLO
    return YOLO(path)


@st.cache_resource(show_spinner="Cargando modelo Detectron2...")
def cargar_detectron2(path, num_classes, conf_thresh):
    import torch
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor
    from detectron2 import model_zoo

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(DETECTRON2_BASE_CFG))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.WEIGHTS = path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = conf_thresh
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    return DefaultPredictor(cfg)


# =========================================================
# 🔍 INFERENCIA
# =========================================================

def inferir_yolo(modelo, imagen_bgr, conf):
    resultados = modelo.predict(imagen_bgr, conf=conf, verbose=False)
    anotada_bgr = resultados[0].plot()  # ya dibuja cajas o máscaras según el modelo
    n_detecciones = len(resultados[0].boxes) if resultados[0].boxes is not None else 0
    return anotada_bgr, n_detecciones


def inferir_detectron2(predictor, imagen_bgr):
    from detectron2.utils.visualizer import Visualizer
    from detectron2.data import MetadataCatalog

    outputs = predictor(imagen_bgr)
    instancias = outputs["instances"].to("cpu")

    metadata = MetadataCatalog.get("ventanas_metadata")
    metadata.thing_classes = ["window"]

    vis = Visualizer(imagen_bgr[:, :, ::-1], metadata=metadata, scale=1.0)
    salida = vis.draw_instance_predictions(instancias)
    anotada_rgb = salida.get_image()

    return anotada_rgb[:, :, ::-1], len(instancias)  # regresamos en BGR por consistencia


def inferir_sam3_batch(rutas_imagenes, prompt, conf):
    cmd = [VENV_SAM3_PYTHON, SAM3_WORKER_SCRIPT, prompt, str(conf), *rutas_imagenes]
    resultado = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if resultado.returncode != 0:
        raise RuntimeError(resultado.stderr.strip()[-2000:] or "sam3_worker falló sin mensaje de error")
    return json.loads(resultado.stdout)


def dibujar_cajas_sam3(imagen_bgr, cajas):
    anotada = imagen_bgr.copy()
    for c in cajas:
        x1, y1, x2, y2 = [int(round(v)) for v in c["box"]]
        cv2.rectangle(anotada, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            anotada, f"{c['score']:.2f}", (x1, max(y1 - 5, 0)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA,
        )
    return anotada


def procesar_imagen(nombre_modelo, conf, archivo_imagen):
    info = MODELOS_DISPONIBLES[nombre_modelo]

    imagen_pil = Image.open(archivo_imagen).convert("RGB")
    imagen_np = np.array(imagen_pil)
    imagen_bgr = cv2.cvtColor(imagen_np, cv2.COLOR_RGB2BGR)

    if info["tipo"] == "yolo":
        modelo = cargar_yolo(info["path"])
        anotada_bgr, n = inferir_yolo(modelo, imagen_bgr, conf)
    else:
        predictor = cargar_detectron2(info["path"], DETECTRON2_NUM_CLASSES, conf)
        anotada_bgr, n = inferir_detectron2(predictor, imagen_bgr)

    anotada_rgb = cv2.cvtColor(anotada_bgr, cv2.COLOR_BGR2RGB)
    return anotada_rgb, n


# =========================================================
# 🖥️ INTERFAZ
# =========================================================

st.title("🪟 Detección de ventanas")
st.caption("Prueba tus 4 modelos entrenados: YOLOv8, YOLOv8-seg, YOLOv11-seg y Detectron2")

with st.sidebar:
    st.header("Configuración")
    modelo_elegido = st.selectbox("Modelo a usar", list(MODELOS_DISPONIBLES.keys()))
    conf = st.slider("Umbral de confianza", 0.0, 1.0, 0.30, 0.05)

    prompt_sam3 = "window"
    if MODELOS_DISPONIBLES[modelo_elegido]["tipo"] == "sam3":
        prompt_sam3 = st.text_input("Prompt de texto (SAM3)", value="window")
        st.caption("ℹ️ SAM3 corre en un subproceso aparte (venv_sam3); el modelo se recarga en cada tanda de imágenes.")

    if not DETECTRON2_DISPONIBLE:
        st.caption("ℹ️ Detectron2 no está instalado en este entorno; corre la app localmente para usarlo.")
    if not SAM3_DISPONIBLE:
        st.caption("ℹ️ SAM3 no está disponible en este entorno (falta venv_sam3); corre la app localmente para usarlo.")

archivos = st.file_uploader(
    "Sube una o varias imágenes",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
)

if archivos:
    st.subheader(f"Resultados con: {modelo_elegido}")

    if MODELOS_DISPONIBLES[modelo_elegido]["tipo"] == "sam3":
        with tempfile.TemporaryDirectory() as tmpdir:
            rutas = []
            for archivo in archivos:
                ruta = os.path.join(tmpdir, archivo.name)
                with open(ruta, "wb") as f:
                    f.write(archivo.getbuffer())
                rutas.append(ruta)

            with st.spinner(f"Corriendo SAM3 sobre {len(rutas)} imagen(es)..."):
                try:
                    resultados_sam3 = inferir_sam3_batch(rutas, prompt_sam3, conf)
                except Exception as e:
                    st.error(f"Error corriendo SAM3: {e}")
                    resultados_sam3 = {}

            for archivo, ruta in zip(archivos, rutas):
                info = resultados_sam3.get(ruta, {"error": "sin resultado"})
                if "error" in info:
                    st.error(f"Error procesando {archivo.name}: {info['error']}")
                    continue

                imagen_bgr = cv2.cvtColor(np.array(Image.open(ruta).convert("RGB")), cv2.COLOR_RGB2BGR)
                anotada_bgr = dibujar_cajas_sam3(imagen_bgr, info.get("cajas", []))
                anotada_rgb = cv2.cvtColor(anotada_bgr, cv2.COLOR_BGR2RGB)

                col1, col2 = st.columns(2)
                with col1:
                    st.image(archivo, caption=f"Original: {archivo.name}", use_container_width=True)
                with col2:
                    n = len(info.get("cajas", []))
                    st.image(anotada_rgb, caption=f"Detecciones ({n} ventanas)", use_container_width=True)
    else:
        for archivo in archivos:
            with st.spinner(f"Procesando {archivo.name}..."):
                try:
                    anotada, n_detecciones = procesar_imagen(modelo_elegido, conf, archivo)
                except Exception as e:
                    st.error(f"Error procesando {archivo.name}: {e}")
                    continue

            col1, col2 = st.columns(2)
            with col1:
                st.image(archivo, caption=f"Original: {archivo.name}", use_container_width=True)
            with col2:
                st.image(anotada, caption=f"Detecciones ({n_detecciones} ventanas)", use_container_width=True)
else:
    st.info("Sube una o varias imágenes para comenzar.")
