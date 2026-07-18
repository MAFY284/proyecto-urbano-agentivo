"""Evaluación de fachadas con el pool de 7 modelos YOLO11.

Backend del antiguo servidor Flask (`servidor_deteccion.py`) convertido en
herramienta reutilizable: sin rutas HTTP, sin estado global de aplicación.
Conserva íntegra la lógica heredada — estimación de pisos por filas de
ventanas, separación y ponderación de daños por severidad, combinación de
altura entre modelos — ahora parametrizable vía config/settings.yaml.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src import settings

# ── Clases de daño/deterioro (heredadas del proyecto 1) ──
CLASES_DANO = {
    'crack', 'ac_bracket_corrosion', 'concrete_spalling', 'exposed_reinforcement',
    'peeling_plaster', 'tile_detachment', 'corrosion', 'delamination',
    'dirty_mold', 'paint_defect',
}

_modelos_cache: dict = {}
_device_cache: str | None = None


def peso_dano_por_clase() -> dict:
    return settings.cargar()["riesgo"]["peso_dano_por_clase"]


def calcular_danos_ponderados(conteo_clases: dict) -> float:
    """Suma las clases de daño aplicando severidad (estructural ×1.0,
    acabados ×0.5, estético ×0.1) en vez de contar todo parejo."""
    pesos = peso_dano_por_clase()
    return sum(cant * pesos.get(clase, 0) for clase, cant in conteo_clases.items())


def dispositivo_inferencia() -> str:
    """Detecta las GPUs disponibles y las usa todas ('0,1,2' con tres A6000),
    o cae a CPU. Cacheado: la detección solo corre una vez por proceso."""
    global _device_cache
    if _device_cache is None:
        try:
            import torch
            n = torch.cuda.device_count() if torch.cuda.is_available() else 0
        except ImportError:
            n = 0
        _device_cache = ','.join(str(i) for i in range(n)) if n else 'cpu'
    return _device_cache


def tipos_disponibles() -> dict:
    """{tipo: nombre legible} de las 7 categorías del pool."""
    return {t: cfg["nombre"] for t, cfg in settings.cargar()["modelos_fachada"].items()}


def cargar_modelo(tipo: str):
    """Carga (una sola vez por proceso) el modelo YOLO11 de una categoría,
    con respaldo al peso COCO preentrenado si el checkpoint no existe."""
    if tipo in _modelos_cache:
        return _modelos_cache[tipo]
    from ultralytics import YOLO
    cfg = settings.cargar()["modelos_fachada"][tipo]
    ruta = settings.ruta(cfg["pesos"])
    modelo = YOLO(str(ruta)) if ruta.exists() else YOLO(cfg["fallback"])
    _modelos_cache[tipo] = modelo
    return modelo


def parsear_conf(valor, por_defecto: float | None = None) -> float:
    """Umbral de confianza saneado a [0.05, 0.95] — nunca 0 (mostraría hasta
    el ruido) ni 1 (no mostraría nada)."""
    if por_defecto is None:
        por_defecto = settings.cargar()["inferencia"]["conf_por_defecto"]
    try:
        return max(0.05, min(0.95, float(valor)))
    except (TypeError, ValueError):
        return por_defecto


def estimar_pisos_por_filas(ventana_boxes, factor_umbral: float | None = None,
                            umbral_minimo: float | None = None) -> int:
    """Cuenta pisos agrupando cajas de ventana por posición vertical (fila).

    Ventanas de un mismo piso quedan muy cerca en yc — mucho más cerca entre
    sí que la separación típica entre pisos — así que agruparlas por cercanía
    vertical da número de filas = número de pisos. Sustituye al heurístico
    `ventanas // 3`, que fallaba con fachadas de ≠3 ventanas por nivel.

    ventana_boxes: lista de (yc, h) normalizados (0-1), centro y alto por caja.
    Los umbrales son parametrizables (settings.yaml → inferencia).
    """
    if not ventana_boxes:
        return 0
    inf = settings.cargar()["inferencia"]
    factor = factor_umbral if factor_umbral is not None else inf["umbral_fila_factor"]
    minimo = umbral_minimo if umbral_minimo is not None else inf["umbral_fila_minimo"]

    ys = sorted(yc for yc, _ in ventana_boxes)
    alto_prom = sum(h for _, h in ventana_boxes) / len(ventana_boxes)
    umbral = max(alto_prom * factor, minimo)

    filas = 1
    y_prev = ys[0]
    for y in ys[1:]:
        if y - y_prev > umbral:
            filas += 1
        y_prev = y
    return filas


@dataclass
class ResultadoFachada:
    """Salida normalizada de cualquier motor (YOLO11 o Detectron2), para que
    el resto del pipeline no distinga de qué motor vino el resultado."""
    tipo: str
    conteo_clases: dict
    ventanas: int
    ventana_boxes: list          # [(yc, h) normalizados]
    confianzas: list             # score de cada detección (0-1)
    numero_pisos: int
    altura_aproximada: float
    danos_detectados: dict
    total_danos: int
    danos_ponderados: float
    imagen_anotada: np.ndarray | None = None   # RGB
    extra: dict = field(default_factory=dict)

    @property
    def confianza_promedio(self) -> float:
        return float(np.mean(self.confianzas)) if self.confianzas else 0.0


def _altura(pisos: int) -> float:
    altura_piso = settings.cargar()["inferencia"]["altura_piso_m"]
    return round(pisos * altura_piso, 1) if pisos else 0.0


def detectar(imagen, tipo: str = 'fachada', conf: float | None = None,
             device: str | None = None, con_imagen: bool = True) -> ResultadoFachada:
    """Inferencia YOLO11 de una categoría del pool sobre una imagen PIL/ruta/np.

    La selección de categoría es explícita — la auto-detección por confianza
    se probó en el proyecto original y confundía fachadas con techos.
    """
    import cv2

    conf = parsear_conf(conf)
    modelo = cargar_modelo(tipo)
    results = modelo.predict(source=imagen, conf=conf, imgsz=640,
                             device=device or dispositivo_inferencia(), verbose=False)
    r = results[0]

    imagen_anotada = None
    if con_imagen:
        imagen_anotada = cv2.cvtColor(r.plot(), cv2.COLOR_BGR2RGB)

    nombres = r.names
    conteo_clases: dict = {}
    for c in r.boxes.cls:
        clase = nombres[int(c)]
        conteo_clases[clase] = conteo_clases.get(clase, 0) + 1

    confianzas = [float(v) for v in r.boxes.conf]

    ventana_boxes = [
        (float(box[1]), float(box[3]))  # (yc, h) normalizados
        for box, c in zip(r.boxes.xywhn, r.boxes.cls)
        if nombres[int(c)].lower() == 'window'
    ]

    numero_pisos = estimar_pisos_por_filas(ventana_boxes)
    if numero_pisos == 0:
        # respaldo: clase 'floor' si el modelo la tiene
        numero_pisos = conteo_clases.get('floor', 0) or conteo_clases.get('Floor', 0)

    danos = {k: v for k, v in conteo_clases.items() if k in CLASES_DANO}

    return ResultadoFachada(
        tipo=tipo,
        conteo_clases=conteo_clases,
        ventanas=len(ventana_boxes),
        ventana_boxes=ventana_boxes,
        confianzas=confianzas,
        numero_pisos=numero_pisos,
        altura_aproximada=_altura(numero_pisos),
        danos_detectados=danos,
        total_danos=sum(danos.values()),
        danos_ponderados=calcular_danos_ponderados(conteo_clases),
        imagen_anotada=imagen_anotada,
    )


def detectar_multiple(imagen, tipos: list[str], conf: float | None = None,
                      device: str | None = None) -> dict[str, ResultadoFachada]:
    """Corre varias categorías del pool sobre la misma imagen, una por una
    (evita la presión de VRAM de varios modelos grandes en simultáneo)."""
    return {t: detectar(imagen, t, conf=conf, device=device) for t in tipos}


PRIORIDAD_ALTURA = ['ventanas', 'ventanas_detectron2', 'fachada', 'fachada_general']


def combinar_altura_estimada(resultados: dict[str, ResultadoFachada]) -> dict | None:
    """Elige la estimación de altura más confiable entre los modelos corridos,
    en vez de sumar ventanas de todos (un mismo hueco puede salir detectado
    por varios modelos a la vez — sumarlos infla el conteo de filas).
    Prioridad: motores dedicados a ventanas > fachada > fachada_general."""
    origen = next((t for t in PRIORIDAD_ALTURA
                   if t in resultados and resultados[t].numero_pisos > 0), None)
    if not origen:
        return None
    return {
        "origen": origen,
        "numero_pisos": resultados[origen].numero_pisos,
        "altura_aproximada": resultados[origen].altura_aproximada,
    }
