
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from ultralytics import YOLO
import os
import cv2
import re
import json
import math
import sqlite3
import numpy as np
from PIL import Image
import io
import base64
from collections import defaultdict
from shapely.geometry import Point, shape
import threading
from reporte_pdf import generar_pdf_reporte

app = Flask(__name__)
CORS(app)

# ── Base de datos SQLite ──
DB_PATH = 'detecciones.db'

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS detecciones (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        archivo TEXT,
        lat REAL,
        lon REAL,
        cvegeo TEXT,
        numero_pisos INTEGER,
        altura_aproximada REAL,
        ventanas INTEGER,
        conteo_clases TEXT,
        danos_total INTEGER DEFAULT 0,
        fecha TEXT DEFAULT (datetime('now','localtime'))
    )''')
    try:
        c.execute('ALTER TABLE detecciones ADD COLUMN danos_total INTEGER DEFAULT 0')
    except sqlite3.OperationalError:
        pass  # la columna ya existe (base de datos creada antes de esta versión)

    # Misma tabla que crea trafico_tomtom.py — se declara aquí también para poder
    # indexarla incluso si el servidor arranca antes de correr ese script alguna vez.
    c.execute('''CREATE TABLE IF NOT EXISTS trafico_calles (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        vialidad TEXT,
        lat REAL,
        lon REAL,
        cvegeo TEXT,
        velocidad_actual REAL,
        velocidad_libre REAL,
        congestion REAL,
        fecha TEXT DEFAULT (datetime('now','localtime'))
    )''')

    # Índices: las consultas por manzana (riesgo, PDF) y por fecha (retención,
    # series de tiempo futuras) son las más frecuentes en ambas tablas.
    c.execute('CREATE INDEX IF NOT EXISTS idx_detecciones_cvegeo ON detecciones(cvegeo)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_detecciones_fecha ON detecciones(fecha)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_trafico_cvegeo ON trafico_calles(cvegeo)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_trafico_fecha ON trafico_calles(fecha)')

    conn.commit()
    conn.close()

init_db()

# Clases de daño/deterioro de fachada (fusionadas desde datasets de grietas,
# defectos y calidad de superficie). Se separan del resto para que el
# deterioro se muestre como un bloque propio en los resultados, no mezclado
# con los conteos estructurales (balcony/entrance/fence/floor/window).
CLASES_DANO = {
    'crack', 'ac_bracket_corrosion', 'concrete_spalling', 'exposed_reinforcement',
    'peeling_plaster', 'tile_detachment', 'corrosion', 'delamination',
    'dirty_mold', 'paint_defect',
}

# Severidad de cada clase de daño para el score de riesgo (sección "Score de riesgo
# por manzana" más abajo) — sumar todas las clases de daño con el mismo peso hacía
# que 5 manchas de humedad (dirty_mold) pesaran igual que 5 columnas con varilla
# expuesta (exposed_reinforcement), cuando el riesgo real es muy distinto.
PESO_DANO_POR_CLASE = {
    # Alto: fallas estructurales — comprometen la integridad del edificio.
    'crack': 1.0, 'concrete_spalling': 1.0, 'exposed_reinforcement': 1.0, 'corrosion': 1.0,
    # Medio: deterioro de acabados/instalaciones — no estructural pero sí relevante.
    'peeling_plaster': 0.5, 'tile_detachment': 0.5, 'ac_bracket_corrosion': 0.5, 'delamination': 0.5,
    # Bajo: estético/mantenimiento — casi no aporta al riesgo real.
    'paint_defect': 0.1, 'dirty_mold': 0.1,
}


def calcular_danos_ponderados(conteo_clases):
    """Suma las clases de daño de un conteo_clases (dict clase->cantidad) aplicando
    PESO_DANO_POR_CLASE, en vez de contar cada detección como si pesara lo mismo."""
    return sum(cantidad * PESO_DANO_POR_CLASE.get(clase, 0) for clase, cantidad in conteo_clases.items())

# ── Variables globales ──
global_stats = {
    "total_buildings_analyzed": 0,
    "total_windows_detected": 0,
    "total_damage_detections": 0,
    "floor_distribution": defaultdict(int)
}

# Estado del procesamiento en lote. 'current_tipo' y 'log' le dan visibilidad en
# tiempo real de qué modelo se está corriendo ahora mismo, no solo qué imagen —
# antes la interfaz mostraba "procesando imagen 14/100" sin decir con qué modelo,
# lo cual era muy poco informativo con varias categorías elegidas a la vez.
batch_state = {
    "running": False,
    "total": 0,
    "processed": 0,
    "current_file": "",
    "current_tipo": "",
    "log": [],
    "results": []
}
BATCH_LOG_MAX = 30  # líneas recientes que se guardan/exponen, para no crecer sin límite

# ── Cargar GeoJSON de manzanas ──
geojson_path = 'hipodromo_manzanas.geojson'
manzanas_geojson = None
manzanas_shapes = []

if os.path.exists(geojson_path):
    with open(geojson_path, 'r') as f:
        manzanas_geojson = json.load(f)
    for feat in manzanas_geojson['features']:
        try:
            manzanas_shapes.append({
                'cvegeo': feat['properties'].get('CVEGEO', ''),
                'shape': shape(feat['geometry'])
            })
        except Exception:
            pass
    print(f"✅ Cargadas {len(manzanas_shapes)} manzanas del GeoJSON")
else:
    print(f"⚠️ No se encontró {geojson_path}")

def find_manzana(lat, lon):
    """Busca en qué manzana cae un punto (lat, lon)."""
    pt = Point(lon, lat)  # shapely usa (x=lon, y=lat)
    for m in manzanas_shapes:
        if m['shape'].contains(pt):
            return m['cvegeo']
    return None

def parse_coords_from_filename(filename):
    """Extrae lat, lon del nombre de archivo tipo '19.4102_-99.1684.jpg'."""
    base = os.path.splitext(os.path.basename(filename))[0]
    match = re.match(r'(-?\d+\.\d+)_(-?\d+\.\d+)', base)
    if match:
        return float(match.group(1)), float(match.group(2))
    return None, None

def save_detection_to_db(archivo, lat, lon, cvegeo, numero_pisos, altura_aproximada, ventanas, conteo_clases, danos_total=0):
    """Guarda una detección en la base de datos SQLite."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''INSERT INTO detecciones (archivo, lat, lon, cvegeo, numero_pisos, altura_aproximada, ventanas, conteo_clases, danos_total)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
              (archivo, lat, lon, cvegeo, numero_pisos, altura_aproximada, ventanas,
               json.dumps(conteo_clases), danos_total))
    conn.commit()
    conn.close()

# ── Dispositivo de inferencia: usar las 3 GPU A6000 en vez de solo la GPU 0 ──
try:
    import torch
    _n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
except ImportError:
    _n_gpus = 0

if _n_gpus >= 3:
    DEVICE_INFERENCIA = '0,1,2'
elif _n_gpus > 0:
    DEVICE_INFERENCIA = ','.join(str(i) for i in range(_n_gpus))
else:
    DEVICE_INFERENCIA = 'cpu'
print(f"🖥️  Inferencia en dispositivo(s): {DEVICE_INFERENCIA}")

# ── Cargar modelos YOLO ──
# Configuración de las 7 categorías entrenables (ver organizar_datasets.py / README).
# 'nombre' es la etiqueta legible que se muestra en el menú del frontend.
MODELOS_CONFIG = {
    'techo':           {'run': 'entrenamiento_techo',           'fallback': 'yolo11x.pt', 'nombre': 'Techos (satelital)'},
    'fachada':         {'run': 'entrenamiento_fachada',         'fallback': 'yolo11x.pt', 'nombre': 'Fachadas (estructura + daños)'},
    'ventanas':        {'run': 'entrenamiento_ventanas',        'fallback': 'yolo11l.pt', 'nombre': 'Ventanas'},
    'fachada_general': {'run': 'entrenamiento_fachada_general', 'fallback': 'yolo11l.pt', 'nombre': 'Fachada general (arquitectónico)'},
    'danos':           {'run': 'entrenamiento_danos',           'fallback': 'yolo11l.pt', 'nombre': 'Daños/deterioro'},
    'senales':         {'run': 'entrenamiento_senales',         'fallback': 'yolo11l.pt', 'nombre': 'Señalamiento vial'},
    'calles':          {'run': 'entrenamiento_calles',          'fallback': 'yolo11l.pt', 'nombre': 'Calles (vehículos/peatones)'},
}


def nombre_legible_tipo(tipo):
    """Nombre para mostrar en logs/bitácora — incluye el caso especial
    'ventanas_detectron2' que no está en MODELOS_CONFIG (ver TIPOS_VALIDOS)."""
    if tipo == 'ventanas_detectron2':
        return 'Ventanas (Detectron2)'
    return MODELOS_CONFIG.get(tipo, {}).get('nombre', tipo)


modelos = {}
for tipo, cfg in MODELOS_CONFIG.items():
    ruta = f"runs/detect/{cfg['run']}/weights/best.pt"
    if os.path.exists(ruta):
        modelos[tipo] = YOLO(ruta)
        print(f"✅ Cargado modelo de {cfg['nombre']} entrenado: {ruta}")
    else:
        modelos[tipo] = YOLO(cfg['fallback'])
        print(f"⚠️ No hay modelo entrenado de {cfg['nombre']}, usando {cfg['fallback']} preentrenado")

# ── Ventanas con Detectron2 (motor alterno, además del YOLO 'ventanas' de arriba) ──
# Modelo de un compañero (repo ricardo/deteccion-ventanas-streamlit): Faster R-CNN
# R50-FPN, 1 sola clase 'window'. Se carga con un umbral de confianza bajo y fijo
# (el umbral real que elige el usuario se aplica después, filtrando — así no hay
# que reconstruir el predictor en cada request solo porque cambió el slider).
DETECTRON2_PESOS = os.path.join('ricardo', 'deteccion-ventanas-streamlit', 'models', 'detectron2_best.pth')
DETECTRON2_BASE_CFG = 'COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml'
DETECTRON2_UMBRAL_CARGA = 0.05  # se filtra por el conf real de cada request más abajo

detectron2_predictor = None
DETECTRON2_DISPONIBLE = False
try:
    if os.path.exists(DETECTRON2_PESOS):
        import torch as _torch
        from detectron2.config import get_cfg
        from detectron2.engine import DefaultPredictor
        from detectron2 import model_zoo
        from detectron2.data import MetadataCatalog

        _cfg = get_cfg()
        _cfg.merge_from_file(model_zoo.get_config_file(DETECTRON2_BASE_CFG))
        _cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
        _cfg.MODEL.WEIGHTS = DETECTRON2_PESOS
        _cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = DETECTRON2_UMBRAL_CARGA
        _cfg.MODEL.DEVICE = 'cuda' if _torch.cuda.is_available() else 'cpu'
        detectron2_predictor = DefaultPredictor(_cfg)
        MetadataCatalog.get('ventanas_detectron2_metadata').thing_classes = ['window']
        DETECTRON2_DISPONIBLE = True
        print(f"✅ Cargado modelo de Ventanas (Detectron2): {DETECTRON2_PESOS} (device={_cfg.MODEL.DEVICE})")
    else:
        print(f"⚠️ No se encontraron pesos de Detectron2 en {DETECTRON2_PESOS}, esa opción queda oculta")
except ImportError:
    print("⚠️ Detectron2 no está instalado, la opción 'Ventanas (Detectron2)' queda oculta")

ALTURA_PISO_PROMEDIO_M = 3.0  # altura promedio por piso (m), usada para pasar de "pisos" a "altura aproximada"


def estimar_pisos_por_filas(ventana_boxes):
    """Cuenta pisos agrupando cajas de ventana por posición vertical (fila), en vez
    de solo dividir el total de ventanas entre un número fijo por piso — ese
    heurístico anterior (ventanas // 3) fallaba en fachadas con más o menos de 3
    ventanas por nivel. Aquí, ventanas de un mismo piso quedan muy cerca en yc
    (mucho más cerca entre sí que la separación típica entre pisos), así que
    agruparlas por cercanía vertical da el número de filas = número de pisos.

    ventana_boxes: lista de (yc, h) normalizados (0-1), centro y alto de cada caja.
    """
    if not ventana_boxes:
        return 0
    ys = sorted(yc for yc, _ in ventana_boxes)
    alto_prom = sum(h for _, h in ventana_boxes) / len(ventana_boxes)
    umbral = max(alto_prom * 0.6, 0.02)  # ventanas de la misma fila difieren mucho menos que esto

    filas = 1
    y_prev = ys[0]
    for y in ys[1:]:
        if y - y_prev > umbral:
            filas += 1
        y_prev = y
    return filas


CONF_POR_DEFECTO = 0.25


def parsear_conf(valor):
    """Convierte el 'conf' que manda el frontend (string, puede faltar o venir
    corrupto) a un float válido entre 0.05 y 0.95 — nunca se deja en 0 (mostraría
    hasta el ruido) ni en 1 (no mostraría nada)."""
    try:
        return max(0.05, min(0.95, float(valor)))
    except (TypeError, ValueError):
        return CONF_POR_DEFECTO


def run_detection(img, model_type, filename='', conf=CONF_POR_DEFECTO):
    """Ejecuta detección sobre una imagen PIL. Retorna dict con resultados.

    model_type ('techo' o 'fachada') lo decide el usuario en la interfaz —
    la auto-detección por confianza se probó y confundía fachadas con techos
    (p.ej. ventanas etiquetadas como 'edificio' al enrutarse al modelo equivocado),
    así que se volvió a la selección manual.

    conf es el umbral de confianza mínimo de YOLO (0-1): más alto = menos cajas
    pero más seguras, más bajo = más cajas (incluye detecciones dudosas). Antes
    estaba fijo en 0.25; ahora lo elige el usuario desde el slider del frontend.

    Corre en las 3 GPUs A6000 disponibles (antes solo usaba la GPU 0 por defecto).
    """
    model = modelos[model_type]
    results = model.predict(source=img, conf=conf, imgsz=640, device=DEVICE_INFERENCIA, verbose=False)

    # Imagen anotada
    img_array = results[0].plot()
    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    img_procesada = Image.fromarray(img_array)

    buffered = io.BytesIO()
    img_procesada.save(buffered, format="JPEG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    # Conteo por clase
    nombres_clases = results[0].names
    conteo_clases = {}
    for c in results[0].boxes.cls:
        clase = nombres_clases[int(c)]
        conteo_clases[clase] = conteo_clases.get(clase, 0) + 1

    # Altura estimada: agrupa las cajas de ventana por fila (ver estimar_pisos_por_filas).
    # No se limita a un solo model_type — cualquier modelo con clase 'window'/'Window'
    # (fachada, ventanas, fachada_general) puede aportar esta estimación.
    ventana_boxes = [
        (float(box[1]), float(box[3]))  # (yc, h) normalizados
        for box, c in zip(results[0].boxes.xywhn, results[0].boxes.cls)
        if nombres_clases[int(c)].lower() == 'window'
    ]
    ventanas = len(ventana_boxes)

    numero_pisos = estimar_pisos_por_filas(ventana_boxes)
    if numero_pisos == 0:
        # sin ventanas detectadas: como respaldo, usa la clase 'floor'/'Ground Floor' etc. si el modelo la tiene
        numero_pisos = conteo_clases.get('floor', 0) or conteo_clases.get('Floor', 0)
    altura_aproximada = round(numero_pisos * ALTURA_PISO_PROMEDIO_M, 1) if numero_pisos else 0

    # Daños/deterioro detectados (grietas, defectos, óxido, daño de superficie)
    danos_detectados = {k: v for k, v in conteo_clases.items() if k in CLASES_DANO}
    total_danos = sum(danos_detectados.values())

    # Coordenadas y manzana
    lat, lon = parse_coords_from_filename(filename)
    cvegeo = None
    if lat is not None and lon is not None:
        cvegeo = find_manzana(lat, lon)

    return {
        "success": True,
        "tipo": model_type,
        "archivo": filename,
        "conteo_clases": conteo_clases,
        "imagen_base64": img_base64,
        "numero_pisos": numero_pisos,
        "altura_aproximada": altura_aproximada,
        "ventanas": ventanas,
        "ventana_boxes": ventana_boxes,
        "danos_detectados": danos_detectados,
        "total_danos": total_danos,
        "lat": lat,
        "lon": lon,
        "cvegeo": cvegeo
    }


def run_detection_detectron2(img, filename='', conf=CONF_POR_DEFECTO):
    """Igual que run_detection(), pero con el modelo de Detectron2 (Faster R-CNN
    R50-FPN, 1 sola clase 'window') en vez de YOLO — devuelve el mismo formato de
    dict para que el resto del pipeline (combinar_altura_estimada, guardado en BD,
    tarjetas del frontend) no tenga que distinguir de qué motor vino el resultado.

    El predictor se cargó una vez al arrancar el servidor con un umbral bajo fijo
    (DETECTRON2_UMBRAL_CARGA); aquí se filtra por el 'conf' real de este request,
    para no reconstruir el modelo en cada llamada solo por el valor del slider.
    """
    from detectron2.data import MetadataCatalog
    from detectron2.utils.visualizer import Visualizer

    img_rgb = np.array(img.convert('RGB'))
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    alto, ancho = img_bgr.shape[:2]

    outputs = detectron2_predictor(img_bgr)
    instancias = outputs['instances'].to('cpu')
    instancias = instancias[instancias.scores >= conf]

    metadata = MetadataCatalog.get('ventanas_detectron2_metadata')
    vis = Visualizer(img_rgb, metadata=metadata, scale=1.0)
    anotada_rgb = vis.draw_instance_predictions(instancias).get_image()

    buffered = io.BytesIO()
    Image.fromarray(anotada_rgb).save(buffered, format='JPEG')
    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    ventana_boxes = []
    for x1, y1, x2, y2 in instancias.pred_boxes.tensor.tolist():
        yc = ((y1 + y2) / 2) / alto
        h = (y2 - y1) / alto
        ventana_boxes.append((yc, h))
    ventanas = len(ventana_boxes)

    numero_pisos = estimar_pisos_por_filas(ventana_boxes)
    altura_aproximada = round(numero_pisos * ALTURA_PISO_PROMEDIO_M, 1) if numero_pisos else 0

    lat, lon = parse_coords_from_filename(filename)
    cvegeo = None
    if lat is not None and lon is not None:
        cvegeo = find_manzana(lat, lon)

    return {
        "success": True,
        "tipo": "ventanas_detectron2",
        "archivo": filename,
        "conteo_clases": {"window": ventanas} if ventanas else {},
        "imagen_base64": img_base64,
        "numero_pisos": numero_pisos,
        "altura_aproximada": altura_aproximada,
        "ventanas": ventanas,
        "ventana_boxes": ventana_boxes,
        "danos_detectados": {},
        "total_danos": 0,
        "lat": lat,
        "lon": lon,
        "cvegeo": cvegeo
    }


def ejecutar_deteccion(img, model_type, filename='', conf=CONF_POR_DEFECTO):
    """Despacha al motor correcto: YOLO (todo lo que está en 'modelos') o el caso
    especial 'ventanas_detectron2' (motor alterno para la categoría Ventanas)."""
    if model_type == 'ventanas_detectron2':
        return run_detection_detectron2(img, filename, conf=conf)
    return run_detection(img, model_type, filename, conf=conf)


def combinar_altura_estimada(resultados):
    """A partir de un dict {tipo: data} (uno o varios modelos corridos sobre la misma
    imagen), elige la estimación de altura más confiable en vez de sumar ventanas de
    todos los modelos entre sí (un mismo hueco puede salir detectado por 'fachada' Y
    'ventanas' Y 'fachada_general' a la vez, con cajas ligeramente distintas — sumarlas
    infla el conteo de filas). Prioridad: los dos motores dedicados a ventanas primero
    (el que se haya corrido) > 'fachada' (estructura completa) > 'fachada_general'
    (muchas subclases, más ruido para esto)."""
    PRIORIDAD_ALTURA = ['ventanas', 'ventanas_detectron2', 'fachada', 'fachada_general']
    origen = next((t for t in PRIORIDAD_ALTURA if t in resultados and resultados[t]['numero_pisos'] > 0), None)
    if not origen:
        return None
    return {
        "origen": origen,
        "numero_pisos": resultados[origen]["numero_pisos"],
        "altura_aproximada": resultados[origen]["altura_aproximada"],
    }


# 'ventanas_detectron2' es un caso especial: no es una categoría propia del menú (no
# aparece en MODELOS_CONFIG ni en /tipos-disponibles), es un motor alterno para la
# categoría 'ventanas' que el frontend agrega a la lista de tipos cuando el usuario
# lo elige en el selector de motor — por eso se valida aparte, no está en 'modelos'.
TIPOS_VALIDOS = set(modelos) | ({'ventanas_detectron2'} if DETECTRON2_DISPONIBLE else set())

# ── Rutas de la API ──

@app.route('/tipos-disponibles', methods=['GET'])
def tipos_disponibles():
    """Lista las categorías de detección disponibles para el menú del frontend."""
    return jsonify({
        "tipos": [{"id": tipo, "nombre": cfg['nombre']} for tipo, cfg in MODELOS_CONFIG.items()],
        "detectron2_disponible": DETECTRON2_DISPONIBLE,
    })

@app.route('/detectar', methods=['POST'])
def detectar():
    """Detecta objetos en la imagen enviada, para una o varias categorías
    elegidas manualmente por el usuario en la interfaz (checkboxes).
    Devuelve un resultado por cada categoría seleccionada."""
    global global_stats
    if 'imagen' not in request.files:
        return jsonify({"error": "No se proporcionó ninguna imagen"}), 400

    tipos = request.form.getlist('tipo') or ['fachada']
    tipos_invalidos = [t for t in tipos if t not in TIPOS_VALIDOS]
    if tipos_invalidos:
        return jsonify({"error": f"Tipo(s) de modelo inválido(s): {', '.join(tipos_invalidos)}"}), 400

    conf = parsear_conf(request.form.get('conf'))

    file = request.files['imagen']
    if file.filename == '':
        return jsonify({"error": "No se seleccionó ninguna imagen"}), 400

    try:
        img = Image.open(file.stream)
        resultados = {}
        for model_type in tipos:
            data = ejecutar_deteccion(img, model_type, file.filename, conf=conf)
            resultados[model_type] = data

            if model_type == 'fachada':
                global_stats["total_buildings_analyzed"] += 1
                global_stats["total_windows_detected"] += data["ventanas"]
                global_stats["total_damage_detections"] += data["total_danos"]
                if data["numero_pisos"] > 0:
                    global_stats["floor_distribution"][data["numero_pisos"]] += 1

                save_detection_to_db(
                    data["archivo"], data["lat"], data["lon"], data["cvegeo"],
                    data["numero_pisos"], data["altura_aproximada"],
                    data["ventanas"], data["conteo_clases"], data["total_danos"]
                )

        altura_estimada = combinar_altura_estimada(resultados)

        for data in resultados.values():
            del data["ventana_boxes"]  # detalle interno, no hace falta mandarlo al frontend

        return jsonify({"success": True, "resultados": resultados, "altura_estimada": altura_estimada})

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/stats', methods=['GET'])
def get_stats():
    return jsonify({
        "total_buildings_analyzed": global_stats["total_buildings_analyzed"],
        "total_windows_detected": global_stats["total_windows_detected"],
        "total_damage_detections": global_stats["total_damage_detections"],
        "floor_distribution": dict(global_stats["floor_distribution"])
    })

@app.route('/reset-stats', methods=['POST'])
def reset_stats():
    global global_stats
    global_stats = {
        "total_buildings_analyzed": 0,
        "total_windows_detected": 0,
        "total_damage_detections": 0,
        "floor_distribution": defaultdict(int)
    }
    # También limpiar la DB
    conn = sqlite3.connect(DB_PATH)
    conn.execute('DELETE FROM detecciones')
    conn.commit()
    conn.close()
    return jsonify({"success": True})

@app.route('/db-stats', methods=['GET'])
def db_stats():
    """Estadísticas desde la base de datos (persistentes)."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT COUNT(*) FROM detecciones')
    total = c.fetchone()[0]
    c.execute('SELECT SUM(ventanas) FROM detecciones')
    total_ventanas = c.fetchone()[0] or 0
    c.execute('SELECT SUM(danos_total) FROM detecciones')
    total_danos = c.fetchone()[0] or 0
    c.execute('SELECT numero_pisos, COUNT(*) FROM detecciones WHERE numero_pisos > 0 GROUP BY numero_pisos ORDER BY numero_pisos')
    dist = {str(row[0]): row[1] for row in c.fetchall()}
    c.execute('SELECT COUNT(DISTINCT cvegeo) FROM detecciones WHERE cvegeo IS NOT NULL')
    manzanas_con_datos = c.fetchone()[0]
    conn.close()
    return jsonify({
        "total_registros": total,
        "total_ventanas": total_ventanas,
        "total_danos": total_danos,
        "floor_distribution": dist,
        "manzanas_con_datos": manzanas_con_datos
    })

# ── Exportar reporte PDF ──
@app.route('/exportar-pdf', methods=['GET'])
def exportar_pdf():
    """Genera un reporte PDF con métricas globales, gráficas y datos por manzana."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT COUNT(*) FROM detecciones')
    total_registros = c.fetchone()[0]
    c.execute('SELECT SUM(ventanas) FROM detecciones')
    total_ventanas = c.fetchone()[0] or 0
    c.execute('SELECT SUM(danos_total) FROM detecciones')
    total_danos = c.fetchone()[0] or 0
    c.execute('SELECT numero_pisos, COUNT(*) FROM detecciones WHERE numero_pisos > 0 GROUP BY numero_pisos ORDER BY numero_pisos')
    floor_distribution = {str(row[0]): row[1] for row in c.fetchall()}
    c.execute('''SELECT cvegeo,
                    ROUND(AVG(numero_pisos), 1) as avg_pisos,
                    MAX(numero_pisos) as max_pisos,
                    SUM(ventanas) as total_ventanas,
                    COUNT(*) as num_fotos,
                    ROUND(AVG(altura_aproximada), 1) as avg_altura
                 FROM detecciones
                 WHERE cvegeo IS NOT NULL
                 GROUP BY cvegeo
                 ORDER BY num_fotos DESC''')
    manzanas_rows = c.fetchall()

    # Desglose de daños por clase: se agrega en Python a partir de conteo_clases
    # (guardado como JSON por detección) porque SQLite no tiene agregación JSON nativa aquí.
    c.execute('SELECT conteo_clases FROM detecciones WHERE danos_total > 0')
    damage_breakdown = defaultdict(int)
    for (conteo_json,) in c.fetchall():
        conteo = json.loads(conteo_json)
        for clase, cantidad in conteo.items():
            if clase in CLASES_DANO:
                damage_breakdown[clase] += cantidad

    # Congestión vial por manzana (trafico_tomtom.py) — puede no existir aún la tabla
    try:
        c.execute('''SELECT cvegeo, ROUND(AVG(congestion), 3) as congestion_promedio, COUNT(*) as num_lecturas
                     FROM trafico_calles
                     WHERE cvegeo IS NOT NULL AND congestion IS NOT NULL
                     GROUP BY cvegeo
                     ORDER BY congestion_promedio DESC''')
        trafico_rows = c.fetchall()
    except sqlite3.OperationalError:
        trafico_rows = []
    conn.close()

    pdf_buffer = generar_pdf_reporte(
        total_buildings_analyzed=global_stats["total_buildings_analyzed"],
        total_windows_detected=global_stats["total_windows_detected"],
        total_registros=total_registros,
        total_ventanas=total_ventanas,
        total_danos=total_danos,
        damage_breakdown=dict(damage_breakdown),
        floor_distribution=floor_distribution,
        manzanas_rows=manzanas_rows,
        trafico_rows=trafico_rows,
    )

    return send_file(pdf_buffer, as_attachment=True, download_name='reporte_deteccion.pdf',
                     mimetype='application/pdf')

@app.route('/trafico-manzanas', methods=['GET'])
def trafico_manzanas():
    """Congestión vial promedio por manzana, recolectada por trafico_tomtom.py."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        c.execute('''SELECT cvegeo, ROUND(AVG(congestion), 3) as congestion_promedio, COUNT(*) as num_lecturas,
                            MAX(fecha) as ultima_lectura
                     FROM trafico_calles
                     WHERE cvegeo IS NOT NULL AND congestion IS NOT NULL
                     GROUP BY cvegeo
                     ORDER BY congestion_promedio DESC''')
        rows = c.fetchall()
    except sqlite3.OperationalError:
        rows = []
    conn.close()
    return jsonify({
        "manzanas": [
            {"cvegeo": r[0], "congestion_promedio": r[1], "num_lecturas": r[2], "ultima_lectura": r[3]}
            for r in rows
        ]
    })

# Pesos del score de riesgo combinado — daños detectados, congestión vial y
# altura del edificio (más pisos = más gente/mayor exposición en caso de sismo).
PESO_DANOS = 0.4
PESO_CONGESTION = 0.3
PESO_ALTURA = 0.3
DANOS_NORMALIZACION = 10   # 10+ puntos de daño ponderado (ver PESO_DANO_POR_CLASE) = score máximo (1.0)
PISOS_NORMALIZACION = 10   # 10+ pisos promedio = score de altura máximo (1.0)

# Factor de confianza por volumen de muestra: con pocas fotos, el promedio/suma por
# manzana no es confiable (una sola foto con un daño severo no debería pesar igual
# que 50 fotos que confirman el mismo nivel de deterioro). Se amortigua el score de
# daños/altura con 1 - e^(-k * num_fotos): con k=0.5, 1 foto da ~39% de confianza,
# 3 fotos ~78%, 5 fotos ~92%, 10+ fotos ya es prácticamente 100%.
FACTOR_CONFIANZA_K = 0.5


def factor_confianza(num_fotos):
    return 1 - math.exp(-FACTOR_CONFIANZA_K * num_fotos)


# Horas pico (24h) en las que la congestión vial sí es un factor de riesgo real para
# evacuación/acceso de emergencias — correr trafico_tomtom.py de madrugada daría
# congestión ~0 y bajaría el score de riesgo de forma artificial si se promediara
# parejo con todo el día.
HORAS_PICO = [(8, 10), (18, 20)]


@app.route('/manzanas-geojson', methods=['GET'])
def manzanas_geojson_endpoint():
    """Expone el GeoJSON de manzanas (ya cargado en memoria) para dibujar el mapa en el frontend."""
    if manzanas_geojson is None:
        return jsonify({"error": "No se encontró hipodromo_manzanas.geojson en el servidor"}), 404
    return jsonify(manzanas_geojson)

@app.route('/riesgo-por-manzana', methods=['GET'])
def riesgo_por_manzana():
    """Score de riesgo combinado por manzana: daños detectados (fachadas, ponderados
    por severidad) + congestión vial en horas pico (trafico_tomtom.py) + altura
    promedio de los edificios, amortiguado por cuántas fotos hay de esa manzana.
    No sustituye un análisis de riesgo formal — es una priorización relativa
    entre las manzanas ya cubiertas por ambas fuentes de datos."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Se trae conteo_clases por fila (no un SUM pre-agregado) para poder ponderar
    # cada clase de daño por severidad antes de sumar (ver calcular_danos_ponderados).
    c.execute('''SELECT cvegeo, conteo_clases, numero_pisos
                 FROM detecciones
                 WHERE cvegeo IS NOT NULL''')
    detecciones = defaultdict(lambda: {'num_fotos': 0, 'danos_ponderados': 0.0, 'danos_crudos': 0, 'pisos': []})
    for cvegeo, conteo_json, numero_pisos in c.fetchall():
        grupo = detecciones[cvegeo]
        grupo['num_fotos'] += 1
        try:
            conteo_clases = json.loads(conteo_json) if conteo_json else {}
        except (TypeError, ValueError):
            conteo_clases = {}
        grupo['danos_ponderados'] += calcular_danos_ponderados(conteo_clases)
        grupo['danos_crudos'] += sum(v for k, v in conteo_clases.items() if k in CLASES_DANO)
        if numero_pisos:
            grupo['pisos'].append(numero_pisos)

    # Congestión: primero horas pico; si una manzana no tiene lecturas en horas pico
    # todavía, cae al promedio de todas las horas (mejor un dato de respaldo que nada).
    condiciones_pico = ' OR '.join(
        f"(CAST(strftime('%H', fecha) AS INTEGER) BETWEEN {h1} AND {h2})" for h1, h2 in HORAS_PICO
    )
    trafico_pico, trafico_todas_horas = {}, {}
    try:
        c.execute(f'''SELECT cvegeo, AVG(congestion) FROM trafico_calles
                      WHERE cvegeo IS NOT NULL AND congestion IS NOT NULL AND ({condiciones_pico})
                      GROUP BY cvegeo''')
        trafico_pico = {row[0]: row[1] for row in c.fetchall()}

        c.execute('''SELECT cvegeo, AVG(congestion) FROM trafico_calles
                     WHERE cvegeo IS NOT NULL AND congestion IS NOT NULL
                     GROUP BY cvegeo''')
        trafico_todas_horas = {row[0]: row[1] for row in c.fetchall()}
    except sqlite3.OperationalError:
        pass
    conn.close()

    trafico = dict(trafico_todas_horas)
    trafico.update(trafico_pico)  # las lecturas de hora pico, cuando existen, ganan

    resultados = []
    for cvegeo in set(detecciones) | set(trafico):
        grupo = detecciones.get(cvegeo)
        num_fotos = grupo['num_fotos'] if grupo else 0
        danos_ponderados = grupo['danos_ponderados'] if grupo else 0.0
        danos_crudos = grupo['danos_crudos'] if grupo else 0
        pisos_promedio = (sum(grupo['pisos']) / len(grupo['pisos'])) if grupo and grupo['pisos'] else 0
        congestion = trafico.get(cvegeo, 0.0)

        confianza = factor_confianza(num_fotos)
        danos_norm = min(1.0, danos_ponderados / DANOS_NORMALIZACION) * confianza
        pisos_norm = min(1.0, pisos_promedio / PISOS_NORMALIZACION) * confianza
        score = round(danos_norm * PESO_DANOS + congestion * PESO_CONGESTION + pisos_norm * PESO_ALTURA, 3)

        resultados.append({
            "cvegeo": cvegeo,
            "score_riesgo": score,
            "num_fotos": num_fotos,
            "total_danos": danos_crudos,
            "danos_ponderados": round(danos_ponderados, 1),
            "danos_norm": round(danos_norm, 3),   # 0-1, ya con severidad + confianza aplicadas
            "pisos_norm": round(pisos_norm, 3),   # 0-1, ya con confianza aplicada
            "confianza": round(confianza, 2),
            "congestion": round(congestion, 3),
            "congestion_hora_pico": cvegeo in trafico_pico,
            "altura_promedio_pisos": round(pisos_promedio, 1),
        })

    resultados.sort(key=lambda x: x["score_riesgo"], reverse=True)
    return jsonify({"manzanas": resultados})

# ── Procesamiento en lote ──
def batch_process_worker(folder_path, tipos, conf=CONF_POR_DEFECTO):
    """Worker que procesa todas las imágenes de una carpeta en segundo plano.
    Por cada imagen corre las categorías elegidas una por una (mismo patrón que
    /detectar con varios 'tipo'), no todas a la vez — evita la presión de VRAM de
    tener varios modelos grandes prediciendo en simultáneo."""
    global batch_state, global_stats

    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    files = sorted([f for f in os.listdir(folder_path)
                    if os.path.splitext(f)[1].lower() in extensions])

    batch_state["total"] = len(files)
    batch_state["processed"] = 0
    batch_state["results"] = []
    batch_state["current_file"] = ""
    batch_state["current_tipo"] = ""
    batch_state["log"] = []

    for i, fname in enumerate(files):
        if not batch_state["running"]:
            break  # Cancelado

        batch_state["current_file"] = fname
        filepath = os.path.join(folder_path, fname)

        try:
            img = Image.open(filepath)
            resultados = {}
            for model_type in tipos:
                batch_state["current_tipo"] = model_type
                batch_state["log"].append(
                    f"[imagen {i + 1}/{len(files)}] {fname} — corriendo modelo '{nombre_legible_tipo(model_type)}'..."
                )
                batch_state["log"] = batch_state["log"][-BATCH_LOG_MAX:]

                data = ejecutar_deteccion(img, model_type, fname, conf=conf)
                del data["ventana_boxes"]
                resultados[model_type] = data

                if model_type == 'fachada':
                    global_stats["total_buildings_analyzed"] += 1
                    global_stats["total_windows_detected"] += data["ventanas"]
                    global_stats["total_damage_detections"] += data["total_danos"]
                    if data["numero_pisos"] > 0:
                        global_stats["floor_distribution"][data["numero_pisos"]] += 1

                    save_detection_to_db(
                        data["archivo"], data["lat"], data["lon"], data["cvegeo"],
                        data["numero_pisos"], data["altura_aproximada"],
                        data["ventanas"], data["conteo_clases"], data["total_danos"]
                    )

            altura_estimada = combinar_altura_estimada(resultados)
            cvegeo = next((r["cvegeo"] for r in resultados.values() if r["cvegeo"]), None)
            total_danos = sum(r["total_danos"] for r in resultados.values())

            batch_state["results"].append({
                "archivo": fname,
                "tipos": tipos,
                "resultados": resultados,
                "altura_estimada": altura_estimada,
                "danos": total_danos,
                "cvegeo": cvegeo,
            })

        except Exception as e:
            batch_state["results"].append({
                "archivo": fname,
                "error": str(e)
            })

        batch_state["processed"] = i + 1

    batch_state["running"] = False
    batch_state["current_file"] = ""

@app.route('/batch/start', methods=['POST'])
def batch_start():
    global batch_state
    if batch_state["running"]:
        return jsonify({"error": "Ya hay un procesamiento en lote en curso"}), 409

    body = request.json if request.is_json else {}
    folder = body.get('folder', 'Fotos_Calle')
    tipos = body.get('tipos') or ([body['tipo']] if body.get('tipo') else ['fachada'])
    tipos_invalidos = [t for t in tipos if t not in TIPOS_VALIDOS]
    if tipos_invalidos:
        return jsonify({"error": f"Tipo(s) de modelo inválido(s): {', '.join(tipos_invalidos)}"}), 400

    conf = parsear_conf(body.get('conf'))

    folder_path = os.path.abspath(folder)
    if not os.path.isdir(folder_path):
        return jsonify({"error": f"La carpeta '{folder}' no existe"}), 404

    batch_state["running"] = True
    thread = threading.Thread(target=batch_process_worker, args=(folder_path, tipos, conf), daemon=True)
    thread.start()

    return jsonify({"success": True, "message": f"Procesamiento iniciado en '{folder}'"})

@app.route('/batch/progress', methods=['GET'])
def batch_progress():
    return jsonify({
        "running": batch_state["running"],
        "total": batch_state["total"],
        "processed": batch_state["processed"],
        "current_file": batch_state["current_file"],
        "current_tipo": batch_state["current_tipo"],
        "current_tipo_nombre": nombre_legible_tipo(batch_state["current_tipo"]) if batch_state["current_tipo"] else "",
        "log": batch_state["log"][-BATCH_LOG_MAX:],
        "percent": round((batch_state["processed"] / batch_state["total"] * 100), 1) if batch_state["total"] > 0 else 0
    })

@app.route('/batch/cancel', methods=['POST'])
def batch_cancel():
    global batch_state
    batch_state["running"] = False
    return jsonify({"success": True})

@app.route('/batch/results', methods=['GET'])
def batch_results():
    return jsonify({
        "total_processed": len(batch_state["results"]),
        "results": batch_state["results"][-50:]  # últimos 50 resultados
    })

@app.route('/')
def index():
    return send_file('index.html', mimetype='text/html')

if __name__ == "__main__":
    # Puerto configurable: variable de entorno PUERTO, o primer argumento de línea
    # de comandos, para no depender siempre del 3000 si ya está ocupado.
    # Ejemplos: PUERTO=3005 python3 servidor_deteccion.py
    #           python3 servidor_deteccion.py 3005
    import sys
    puerto = int(sys.argv[1]) if len(sys.argv) > 1 else int(os.environ.get('PUERTO', 3005))
    print(f"🌐 Servidor escuchando en http://127.0.0.1:{puerto}")
    app.run(host="127.0.0.1", port=puerto, debug=False)
