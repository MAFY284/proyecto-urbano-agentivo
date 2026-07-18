import os
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
    "Detectron2 (detección)": {"tipo": "detectron2", "path": DETECTRON2_WEIGHTS},
}

st.set_page_config(page_title="Detección de ventanas", layout="wide")

# =========================================================
# 🧠 MEMORIA DE SESIÓN (Para no perder datos al cambiar de pestaña)
# =========================================================
if 'archivos_procesados' not in st.session_state:
    st.session_state['archivos_procesados'] = []
if 'total_ventanas' not in st.session_state:
    st.session_state['total_ventanas'] = 0

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
    return anotada_rgb, n, imagen_pil


# =========================================================
# 🖥️ INTERFAZ
# =========================================================

st.title("🪟 Detección de ventanas")
st.caption("Prueba tus 4 modelos entrenados: YOLOv8, YOLOv8-seg, YOLOv11-seg y Detectron2")

with st.sidebar:
    st.header("Configuración")
    modelo_elegido = st.selectbox("Modelo a usar", list(MODELOS_DISPONIBLES.keys()))
    conf = st.slider("Umbral de confianza", 0.0, 1.0, 0.30, 0.05)

archivos = st.file_uploader(
    "Sube una o varias imágenes",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
)

if archivos:
    st.subheader(f"Resultados con: {modelo_elegido}")
    
    # Limpiar memoria temporal para los nuevos archivos
    st.session_state['archivos_procesados'] = []
    conteo_total_temporal = 0

    for archivo in archivos:
        # MEJORA UX 1: Agregamos el spinner animado de procesamiento
        with st.spinner(f"🏢 Analizando estructura de la fachada en {archivo.name}..."):
            try:
                anotada, n_detecciones, original_pil = procesar_imagen(modelo_elegido, conf, archivo)
                
                # Guardar en la memoria global
                st.session_state['archivos_procesados'].append({
                    "nombre": archivo.name,
                    "original": original_pil,
                    "anotada": anotada,
                    "detecciones": n_detecciones
                })
                conteo_total_temporal += n_detecciones
                
            except Exception as e:
                st.error(f"Error procesando {archivo.name}: {e}")
                continue

        # Mostrar en pantalla
        col1, col2 = st.columns(2)
        with col1:
            st.image(original_pil, caption=f"Original: {archivo.name}", use_container_width=True)
        with col2:
            st.image(anotada, caption=f"Detecciones ({n_detecciones} ventanas)", use_container_width=True)

    # Actualizar conteo maestro para la pestaña de Fusión y Reportes
    st.session_state['total_ventanas'] = conteo_total_temporal
    
    # MEJORA UX 2: Notificación emergente al finalizar
    st.toast("¡Análisis de fachada finalizado con éxito! 🪟", icon="✅")

# MEJORA MEMORIA: Si no se subió un archivo nuevo, pero ya había guardados
elif st.session_state['archivos_procesados']:
    st.info("ℹ️ Mostrando el último análisis guardado. (Sube nuevas imágenes para actualizar)")
    st.metric("Total de ventanas detectadas en memoria", f"{st.session_state['total_ventanas']} unidades")
    
    for item in st.session_state['archivos_procesados']:
        col1, col2 = st.columns(2)
        with col1:
            st.image(item["original"], caption=f"Original: {item['nombre']}", use_container_width=True)
        with col2:
            st.image(item["anotada"], caption=f"Detecciones ({item['detecciones']} ventanas)", use_container_width=True)

else:
    st.info("Sube una o varias imágenes para comenzar.")