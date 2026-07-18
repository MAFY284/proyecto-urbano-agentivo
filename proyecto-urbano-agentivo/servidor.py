#!/usr/bin/env python3
"""Servidor web del sistema multi-agente (Flask + index.html).

Frontend único estilo HUD desde el que se opera TODO el sistema:
mapa de riesgo por manzana, análisis de fachadas con el pool YOLO11
(+ oráculo Detectron2/SAM3), segmentación satelital, laboratorio de
benchmarking, simulador de crisis y recolección de tráfico.

Uso:
    python3 servidor.py            # puerto 3005
    python3 servidor.py 4000
    PUERTO=4000 python3 servidor.py
"""

from __future__ import annotations

import base64
import datetime
import io
import json
import logging
import os
import sys
import threading
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ))

from flask import Flask, jsonify, request, send_file  # noqa: E402
from flask_cors import CORS                            # noqa: E402

from src import db, settings                           # noqa: E402
from src.agents import AgenteRiesgo, AgenteSIG, AgenteVision, Orquestador  # noqa: E402
from src.tools import comparador, fachada, satelite, trafico               # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
app = Flask(__name__)
CORS(app)

# ── Agentes (una sola instancia por proceso) ──
sig = AgenteSIG()
vision = AgenteVision()
riesgo = AgenteRiesgo()
orq = Orquestador()
orq.sig, orq.vision, orq.riesgo = sig, vision, riesgo

# Bitácora reactiva: el frontend la lee por polling para el log en vivo
BITACORA_MAX = 80
bitacora: list[dict] = []
_lock = threading.Lock()


def _evento(e: dict) -> None:
    with _lock:
        bitacora.append(e)
        del bitacora[:-BITACORA_MAX]


orq.suscribir(_evento)


def log_manual(etapa: str, **datos) -> None:
    _evento({"etapa": etapa, "ts": datetime.datetime.now().isoformat(), **datos})


def np_a_base64(arr) -> str:
    """np.ndarray RGB → JPEG base64 (reescala si es gigante)."""
    from PIL import Image
    img = Image.fromarray(arr) if not hasattr(arr, "save") else arr
    if max(img.size) > 1600:
        img.thumbnail((1600, 1600))
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=88)
    return base64.b64encode(buf.getvalue()).decode()


def resumen_resultado(r: fachada.ResultadoFachada) -> dict:
    return {
        "tipo": r.tipo,
        "conteo_clases": r.conteo_clases,
        "ventanas": r.ventanas,
        "numero_pisos": r.numero_pisos,
        "altura_aproximada": r.altura_aproximada,
        "danos_detectados": r.danos_detectados,
        "total_danos": r.total_danos,
        "danos_ponderados": round(r.danos_ponderados, 2),
        "confianza_promedio": round(r.confianza_promedio, 3),
        "imagen_base64": np_a_base64(r.imagen_anotada) if r.imagen_anotada is not None else None,
    }


# ══════════════════════════════ Rutas ══════════════════════════════

@app.route("/")
def index():
    return send_file(RAIZ / "src" / "dashboard" / "index.html", mimetype="text/html")


@app.route("/api/estado")
def estado():
    import torch
    riesgos = db.listar_riesgos()
    return jsonify({
        "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "device": fachada.dispositivo_inferencia(),
        "manzanas": len(sig.listar_manzanas()),
        "manzanas_evaluadas": len(riesgos),
        "tipos_fachada": fachada.tipos_disponibles(),
        "motores_comparador": list(comparador.motores_disponibles()),
        "detectron2": comparador.detectron2_disponible(),
        "sam3": comparador.sam3_disponible(),
        "umbral_oraculo": vision.umbral_oraculo,
        "tomtom_key": bool(trafico.api_key()),
        "bitacora": bitacora[-25:],
    })


@app.route("/api/manzanas-geojson")
def manzanas_geojson():
    extra = {r["cvegeo"]: {
        "score_riesgo": r["score_riesgo"], "congestion": r["congestion"],
        "confianza": r["confianza"], "num_fotos": r["num_fotos"],
        "danos_ponderados": r["danos_ponderados"],
        "altura_promedio_pisos": r["altura_promedio_pisos"],
        "poblacion_estimada": r["poblacion_estimada"],
        "fuente_congestion": r["fuente_congestion"],
    } for r in db.listar_riesgos()}
    return app.response_class(sig.exportar_geojson(propiedades_extra=extra),
                              mimetype="application/geo+json")


@app.route("/api/riesgo")
def ranking():
    return jsonify({"manzanas": db.listar_riesgos()})


@app.route("/api/analizar", methods=["POST"])
def analizar_manzana():
    """Pipeline multi-agente completo para una clave cvegeo (sin imágenes
    nuevas: recalcula con lo acumulado + tráfico en vivo/histórico)."""
    body = request.json or {}
    cvegeo = body.get("cvegeo")
    if not cvegeo or not sig.existe(cvegeo):
        return jsonify({"error": f"CVEGEO inválido: {cvegeo}"}), 400
    resultado = orq.analizar_manzana_sync(cvegeo)
    return jsonify({"success": True, "riesgo": resultado["riesgo"],
                    "delimitacion": resultado["delimitacion"]})


@app.route("/api/analizar-todas", methods=["POST"])
def analizar_todas():
    resultados = orq.analizar_todas_sync()
    return jsonify({"success": True, "evaluadas": len(resultados)})


@app.route("/api/detectar", methods=["POST"])
def detectar():
    """Análisis de fachada con el pool YOLO11 + oráculo autónomo del Agente
    de Visión (equivalente al /detectar del servidor original)."""
    if "imagen" not in request.files:
        return jsonify({"error": "No se proporcionó ninguna imagen"}), 400
    archivo = request.files["imagen"]
    tipos = request.form.getlist("tipo") or ["fachada"]
    invalidos = [t for t in tipos if t not in fachada.tipos_disponibles()]
    if invalidos:
        return jsonify({"error": f"Tipos inválidos: {invalidos}"}), 400
    conf = fachada.parsear_conf(request.form.get("conf"))

    from PIL import Image
    img = Image.open(archivo.stream).convert("RGB")

    log_manual("vision_inicio", archivo=archivo.filename, tipos=tipos, conf=conf)
    analisis = vision.analizar_fachada(img, tipos=tipos, conf=conf)

    geo = sig.localizar_archivo(archivo.filename or "")
    if "fachada" in analisis["resultados"]:
        r_f = analisis["resultados"]["fachada"]
        db.guardar_deteccion(archivo.filename, geo["lat"], geo["lon"], geo["cvegeo"],
                             analisis["numero_pisos"], analisis["altura_aproximada"],
                             analisis["ventanas"], r_f.conteo_clases, r_f.total_danos)
    log_manual("vision_fin", archivo=archivo.filename,
               ventanas=analisis["ventanas"], pisos=analisis["numero_pisos"],
               oraculo=(analisis["correccion"] or {}).get("motor"))

    return jsonify({
        "success": True,
        "resultados": {t: resumen_resultado(r) for t, r in analisis["resultados"].items()},
        "altura_estimada": analisis["altura_estimada"],
        "ventanas": analisis["ventanas"],
        "numero_pisos": analisis["numero_pisos"],
        "confianza_ventanas": analisis["confianza_ventanas"],
        "correccion": analisis["correccion"],
        "geo": geo,
    })


@app.route("/api/comparador", methods=["POST"])
def api_comparador():
    if "imagen" not in request.files:
        return jsonify({"error": "No se proporcionó ninguna imagen"}), 400
    from PIL import Image
    img = Image.open(request.files["imagen"].stream).convert("RGB")
    motores = request.form.getlist("motor") or list(comparador.motores_disponibles())
    conf = fachada.parsear_conf(request.form.get("conf"), 0.3)
    prompt = request.form.get("prompt") or None

    salida = {}
    for m in motores:
        r = comparador.inferir(m, img, conf=conf, prompt_sam3=prompt)
        salida[m] = {
            "n_detecciones": r.n_detecciones,
            "confianza_promedio": round(r.confianza_promedio, 3),
            "error": r.error,
            "imagen_base64": np_a_base64(r.imagen_anotada) if r.imagen_anotada is not None else None,
        }
    return jsonify({"success": True, "resultados": salida})


@app.route("/api/satelite", methods=["POST"])
def api_satelite():
    if "imagen" not in request.files:
        return jsonify({"error": "No se proporcionó ninguna imagen"}), 400
    from PIL import Image
    f = request.form
    img = Image.open(request.files["imagen"].stream).convert("RGB")
    s = settings.cargar()["satelite"]
    esquinas = satelite.Esquinas(
        float(f.get("lat_nw") or s["lat_nw"]), float(f.get("lon_nw") or s["lon_nw"]),
        float(f.get("lat_se") or s["lat_se"]), float(f.get("lon_se") or s["lon_se"]))
    conf = fachada.parsear_conf(f.get("conf"), 0.5)
    escala = float(f.get("escala") or s["escala_m_px"])
    tile_size = max(320, min(1280, int(f.get("tile_size") or s["tile_size"])))
    overlap = max(0, min(256, int(f.get("overlap") or s["overlap_px"])))

    log_manual("satelite_inicio", tam=f"{img.width}x{img.height}", conf=conf,
               tile=tile_size, overlap=overlap)
    r = vision.analizar_satelite(img, esquinas=esquinas, conf=conf,
                                 tile_size=tile_size, overlap=overlap,
                                 escala_m_px=escala)
    r.extra["escala"] = escala
    log_manual("satelite_fin", detecciones=r.num_detecciones,
               area_m2=round(r.area_total_m2, 1))

    return jsonify({
        "success": True,
        "num_detecciones": r.num_detecciones,
        "area_total_m2": round(r.area_total_m2, 2),
        "detecciones": r.detecciones[:200],
        "geojson": r.geojson,
        "centro": r.centro,
        "imagen_base64": np_a_base64(r.imagen_anotada),
    })


@app.route("/api/simular")
def simular():
    cvegeo = request.args.get("cvegeo")
    riesgos = {r["cvegeo"]: r for r in db.listar_riesgos()}
    if cvegeo not in riesgos:
        return jsonify({"error": "Manzana sin evaluación previa"}), 400
    base = riesgos[cvegeo]
    dia = request.args.get("dia", trafico.DIAS_ES[datetime.date.today().weekday()])
    hora = int(request.args.get("hora", datetime.datetime.now().hour))
    factor = float(request.args.get("factor", 1.0))

    def score_en(h):
        cong = min(1.0, trafico.congestion_historica(dia, h) * factor)
        s = riesgo.calcular_score(base["danos_ponderados"], cong,
                                  base["altura_promedio_pisos"], base["num_fotos"])
        return s["score_riesgo"], cong

    curva = [score_en(h) for h in range(24)]
    score_sim, cong_sim = score_en(hora)

    comparativa = []
    for cv, r in riesgos.items():
        cong = min(1.0, trafico.congestion_historica(dia, hora) * factor)
        s = riesgo.calcular_score(r["danos_ponderados"], cong,
                                  r["altura_promedio_pisos"], r["num_fotos"])
        comparativa.append({"cvegeo": cv, "simulado": s["score_riesgo"],
                            "actual": r["score_riesgo"]})
    comparativa.sort(key=lambda x: x["simulado"], reverse=True)

    return jsonify({
        "cvegeo": cvegeo, "dia": dia, "hora": hora, "factor": factor,
        "score_actual": base["score_riesgo"], "score_simulado": score_sim,
        "congestion_simulada": cong_sim,
        "curva_scores": [c[0] for c in curva],
        "curva_congestion": [c[1] for c in curva],
        "comparativa": comparativa[:20],
    })


@app.route("/api/importar-sam3", methods=["POST"])
def importar_sam3():
    """Importa el CSV de alturas estimadas por SAM3 (config/) a la tabla de
    detecciones y recalcula el riesgo de todas las manzanas afectadas."""
    from src.tools import gis_utils
    ruta = RAIZ / "config" / "edificios_altura_estimadaSam3.csv"
    if not ruta.exists():
        return jsonify({"error": f"No se encontró {ruta.name} en config/"}), 404
    log_manual("import_sam3_inicio", archivo=ruta.name)
    resumen = gis_utils.importar_csv_sam3(ruta, buscar_manzana=sig.localizar)
    resultados = orq.analizar_todas_sync()
    log_manual("import_sam3_fin", **resumen, manzanas_reevaluadas=len(resultados))
    return jsonify({"success": True, **resumen,
                    "manzanas_reevaluadas": len(resultados)})


@app.route("/api/trafico", methods=["POST"])
def api_trafico():
    log_manual("trafico_inicio")
    resultados = trafico.recolectar_trafico_sync(buscar_manzana=sig.localizar)
    log_manual("trafico_fin", calles=len(resultados),
               fuente=resultados[0]["fuente"] if resultados else "—")
    return jsonify({"success": True, "calles": resultados})


@app.route("/api/bitacora")
def api_bitacora():
    with _lock:
        return jsonify({"bitacora": bitacora[-BITACORA_MAX:]})


@app.route("/api/pdf")
def api_pdf():
    cvegeo = request.args.get("cvegeo")
    fila = next((r for r in db.listar_riesgos() if r["cvegeo"] == cvegeo), None)
    if fila is None:
        return jsonify({"error": "Manzana sin evaluación"}), 400
    from fpdf import FPDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "REPORTE TÉCNICO DE INTELIGENCIA URBANA", ln=True, align="C")
    pdf.set_font("Helvetica", "I", 10)
    pdf.cell(0, 8, f"Fecha: {datetime.date.today():%d/%m/%Y}  ·  Manzana: {cvegeo}",
             ln=True, align="C")
    pdf.ln(6)
    pdf.set_font("Helvetica", "", 11)
    for linea in [
        f"- Score de Riesgo Urbano Combinado: {fila['score_riesgo']:.3f}",
        f"- Daños ponderados por severidad: {fila['danos_ponderados']}",
        f"- Congestión vial ({fila['fuente_congestion']}): {(fila['congestion'] or 0):.0%}",
        f"- Altura promedio: {fila['altura_promedio_pisos']} pisos",
        f"- Área satelital: {fila['area_satelital_m2'] or 'sin dato'} m2",
        f"- Población expuesta estimada (35 m2/hab): {fila['poblacion_estimada'] or 'sin dato'}",
        f"- Confianza de la muestra: {fila['confianza']:.0%} ({fila['num_fotos']} fotos)",
    ]:
        pdf.cell(0, 8, linea, ln=True)
    pdf.ln(8)
    pdf.set_font("Helvetica", "I", 9)
    pdf.multi_cell(0, 5, "Nota: entregable automatizado del sistema multi-agente. "
                         "Estimaciones proyectuales basadas en visión por computadora.")
    buf = io.BytesIO(bytes(pdf.output()))
    return send_file(buf, as_attachment=True, mimetype="application/pdf",
                     download_name=f"reporte_{cvegeo}.pdf")


if __name__ == "__main__":
    puerto = int(sys.argv[1]) if len(sys.argv) > 1 else int(os.environ.get("PUERTO", 3005))
    print(f"🌐 http://127.0.0.1:{puerto}")
    app.run(host="127.0.0.1", port=puerto, debug=False, threaded=True)
