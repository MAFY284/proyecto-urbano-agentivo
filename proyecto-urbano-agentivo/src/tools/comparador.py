"""Comparador multi-modelo de detección de ventanas (5 motores) y "oráculo"
de segmentación fina del Agente de Visión.

Hereda deteccion-ventanas-streamlit/app.py. Motores:
  - YOLOv8 (detección), YOLOv8-seg, YOLOv11-seg  → ultralytics
  - Detectron2 (Faster R-CNN R50-FPN)            → solo si está instalado
  - SAM3 (zero-shot por texto)                   → SUBPROCESO en el venv
    aislado src/tools/env_sam3 (Python 3.12 + torch 2.7, incompatible con
    el core), comunicándose por JSON vía stdout — nunca en este proceso.

Cada motor devuelve el mismo formato (cajas, scores, conteo, imagen
anotada) para poder compararlos lado a lado o usarlos como corrección.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src import settings

_yolo_cache: dict = {}
_detectron2_cache: dict = {}

MOTOR_YOLOV8 = "YOLOv8 (detección)"
MOTOR_YOLOV8_SEG = "YOLOv8-seg (segmentación)"
MOTOR_YOLOV11_SEG = "YOLOv11-seg (segmentación)"
MOTOR_DETECTRON2 = "Detectron2 (detección)"
MOTOR_SAM3 = "SAM3 (zero-shot, texto)"


@dataclass
class ResultadoComparador:
    motor: str
    n_detecciones: int
    cajas: list                 # [[x1,y1,x2,y2], ...] en píxeles
    scores: list
    imagen_anotada: np.ndarray | None = None   # RGB
    error: str | None = None
    extra: dict = field(default_factory=dict)

    @property
    def confianza_promedio(self) -> float:
        return float(np.mean(self.scores)) if self.scores else 0.0


def _cfg() -> dict:
    return settings.cargar()["comparador"]


def ruta_python_sam3() -> Path:
    return settings.ruta(_cfg()["env_sam3_python"])


def sam3_disponible() -> bool:
    worker = Path(__file__).with_name("sam3_worker.py")
    return ruta_python_sam3().exists() and worker.exists()


def detectron2_disponible() -> bool:
    try:
        import detectron2  # noqa: F401
    except ImportError:
        return False
    return settings.ruta(_cfg()["detectron2_pesos"]).exists()


def motores_disponibles() -> dict:
    """{nombre: tipo}. Los motores sin entorno/pesos simplemente no aparecen
    (detección de entorno automática, sin romper la app)."""
    cfg = _cfg()
    motores = {}
    if settings.ruta(cfg["yolov8_det"]).exists():
        motores[MOTOR_YOLOV8] = "yolo"
    if settings.ruta(cfg["yolov8_seg"]).exists():
        motores[MOTOR_YOLOV8_SEG] = "yolo"
    if settings.ruta(cfg["yolov11_seg"]).exists():
        motores[MOTOR_YOLOV11_SEG] = "yolo"
    if detectron2_disponible():
        motores[MOTOR_DETECTRON2] = "detectron2"
    if sam3_disponible():
        motores[MOTOR_SAM3] = "sam3"
    return motores


def _ruta_pesos(motor: str) -> Path:
    cfg = _cfg()
    return settings.ruta({
        MOTOR_YOLOV8: cfg["yolov8_det"],
        MOTOR_YOLOV8_SEG: cfg["yolov8_seg"],
        MOTOR_YOLOV11_SEG: cfg["yolov11_seg"],
        MOTOR_DETECTRON2: cfg["detectron2_pesos"],
    }[motor])


def _cargar_yolo(ruta: Path):
    if ruta not in _yolo_cache:
        from ultralytics import YOLO
        _yolo_cache[ruta] = YOLO(str(ruta))
    return _yolo_cache[ruta]


def _cargar_detectron2():
    """Predictor cargado una sola vez con umbral bajo fijo; el conf real del
    usuario se filtra después — así el slider no reconstruye el modelo."""
    if "predictor" not in _detectron2_cache:
        import torch
        from detectron2 import model_zoo
        from detectron2.config import get_cfg
        from detectron2.data import MetadataCatalog
        from detectron2.engine import DefaultPredictor

        cfg_local = _cfg()
        d2 = get_cfg()
        d2.merge_from_file(model_zoo.get_config_file(cfg_local["detectron2_base_cfg"]))
        d2.MODEL.ROI_HEADS.NUM_CLASSES = 1
        d2.MODEL.WEIGHTS = str(settings.ruta(cfg_local["detectron2_pesos"]))
        d2.MODEL.ROI_HEADS.SCORE_THRESH_TEST = cfg_local["detectron2_umbral_carga"]
        d2.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        MetadataCatalog.get("ventanas_metadata").thing_classes = ["window"]
        _detectron2_cache["predictor"] = DefaultPredictor(d2)
    return _detectron2_cache["predictor"]


def _a_bgr(imagen) -> np.ndarray:
    import cv2
    from PIL import Image
    if not isinstance(imagen, Image.Image):
        imagen = Image.open(imagen)
    return cv2.cvtColor(np.array(imagen.convert("RGB")), cv2.COLOR_RGB2BGR)


def inferir_yolo(motor: str, imagen, conf: float) -> ResultadoComparador:
    import cv2
    modelo = _cargar_yolo(_ruta_pesos(motor))
    bgr = _a_bgr(imagen)
    r = modelo.predict(bgr, conf=conf, verbose=False)[0]
    cajas = r.boxes.xyxy.cpu().numpy().tolist() if r.boxes is not None else []
    scores = [float(v) for v in r.boxes.conf] if r.boxes is not None else []
    return ResultadoComparador(
        motor=motor, n_detecciones=len(cajas), cajas=cajas, scores=scores,
        imagen_anotada=cv2.cvtColor(r.plot(), cv2.COLOR_BGR2RGB))


def inferir_detectron2(imagen, conf: float) -> ResultadoComparador:
    import cv2
    from detectron2.data import MetadataCatalog
    from detectron2.utils.visualizer import Visualizer

    predictor = _cargar_detectron2()
    bgr = _a_bgr(imagen)
    outputs = predictor(bgr)
    inst = outputs["instances"].to("cpu")
    inst = inst[inst.scores >= conf]     # el conf real se aplica aquí

    vis = Visualizer(bgr[:, :, ::-1], metadata=MetadataCatalog.get("ventanas_metadata"), scale=1.0)
    anotada_rgb = vis.draw_instance_predictions(inst).get_image()

    return ResultadoComparador(
        motor=MOTOR_DETECTRON2, n_detecciones=len(inst),
        cajas=inst.pred_boxes.tensor.tolist(),
        scores=[float(s) for s in inst.scores],
        imagen_anotada=anotada_rgb)


def inferir_sam3_batch(rutas_imagenes: list[str], prompt: str | None = None,
                       conf: float = 0.3, timeout: int = 300) -> dict:
    """Interfaz de subproceso SEGURO hacia el venv aislado de SAM3: invoca
    sam3_worker.py con el intérprete de env_sam3 y parsea el JSON de stdout.
    Los conflictos de PyTorch nunca tocan el proceso principal."""
    prompt = prompt or _cfg()["sam3_prompt_defecto"]
    worker = str(Path(__file__).with_name("sam3_worker.py"))
    cmd = [str(ruta_python_sam3()), worker, prompt, str(conf), *rutas_imagenes]
    resultado = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if resultado.returncode != 0:
        raise RuntimeError(resultado.stderr.strip()[-2000:] or "sam3_worker falló sin mensaje de error")
    return json.loads(resultado.stdout)


def _dibujar_cajas(bgr: np.ndarray, cajas: list, scores: list) -> np.ndarray:
    import cv2
    anotada = bgr.copy()
    for caja, score in zip(cajas, scores):
        x1, y1, x2, y2 = [int(round(v)) for v in caja]
        cv2.rectangle(anotada, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(anotada, f"{score:.2f}", (x1, max(y1 - 5, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    return anotada


def inferir_sam3(imagen, prompt: str | None = None, conf: float = 0.3) -> ResultadoComparador:
    """Versión de una sola imagen (acepta PIL o ruta); serializa a un archivo
    temporal porque el worker solo entiende rutas."""
    import cv2
    from PIL import Image

    if isinstance(imagen, (str, os.PathLike)):
        rutas = [str(imagen)]
        tmp = None
    else:
        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        imagen.convert("RGB").save(tmp.name, format="JPEG")
        rutas = [tmp.name]
    try:
        salida = inferir_sam3_batch(rutas, prompt=prompt, conf=conf)
        info = salida.get(rutas[0], {"error": "sin resultado"})
        if "error" in info:
            return ResultadoComparador(motor=MOTOR_SAM3, n_detecciones=0,
                                       cajas=[], scores=[], error=info["error"])
        cajas = [c["box"] for c in info.get("cajas", [])]
        scores = [c["score"] for c in info.get("cajas", [])]
        bgr = _a_bgr(Image.open(rutas[0]))
        anotada = cv2.cvtColor(_dibujar_cajas(bgr, cajas, scores), cv2.COLOR_BGR2RGB)
        return ResultadoComparador(motor=MOTOR_SAM3, n_detecciones=len(cajas),
                                   cajas=cajas, scores=scores, imagen_anotada=anotada)
    finally:
        if tmp is not None:
            os.unlink(tmp.name)


def inferir(motor: str, imagen, conf: float = 0.3,
            prompt_sam3: str | None = None) -> ResultadoComparador:
    """Punto de entrada único: despacha al motor pedido."""
    tipo = motores_disponibles().get(motor)
    if tipo is None:
        return ResultadoComparador(motor=motor, n_detecciones=0, cajas=[], scores=[],
                                   error=f"Motor '{motor}' no disponible en este entorno")
    if tipo == "yolo":
        return inferir_yolo(motor, imagen, conf)
    if tipo == "detectron2":
        return inferir_detectron2(imagen, conf)
    return inferir_sam3(imagen, prompt=prompt_sam3, conf=conf)


def comparar(imagen, motores: list[str] | None = None, conf: float = 0.3,
             prompt_sam3: str | None = None) -> dict[str, ResultadoComparador]:
    """Corre varios motores sobre la misma imagen y regresa sus resultados
    lado a lado (base del Laboratorio de Benchmarking del dashboard)."""
    motores = motores or list(motores_disponibles())
    out = {}
    for m in motores:
        try:
            out[m] = inferir(m, imagen, conf=conf, prompt_sam3=prompt_sam3)
        except Exception as e:
            out[m] = ResultadoComparador(motor=m, n_detecciones=0, cajas=[],
                                         scores=[], error=str(e))
    return out
