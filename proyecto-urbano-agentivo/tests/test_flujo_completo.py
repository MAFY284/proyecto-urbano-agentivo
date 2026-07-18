#!/usr/bin/env python3
"""Prueba end-to-end del flujo multi-agente (Fase 5).

Simula la petición de análisis de una clave `cvegeo` real y verifica la
cadena completa de datos:

    Agente SIG delimita → [Visión simulada inserta detecciones] →
    Agente Riesgo calcula con TomTom/histórico → BD actualiza →
    los datos quedan listos para que Streamlit los renderice.

La etapa de visión se SIMULA (se insertan detecciones sintéticas con el
mismo esquema que produce el Agente de Visión) para que la prueba corra en
segundos y sin GPU. Con --con-modelos también se ejercita la inferencia
YOLO11 real sobre una imagen sintética.

Uso:
    python tests/test_flujo_completo.py
    python tests/test_flujo_completo.py --con-modelos
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

FALLOS = []


def verificar(nombre: str, condicion: bool, detalle: str = "") -> None:
    estado = "✅" if condicion else "❌"
    print(f"  {estado} {nombre}" + (f" — {detalle}" if detalle else ""))
    if not condicion:
        FALLOS.append(nombre)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--con-modelos", action="store_true",
                        help="Además corre inferencia YOLO11 real (lento sin GPU)")
    args = parser.parse_args()

    # ── Base de datos temporal: NO tocar la BD real del proyecto ──
    import tempfile
    tmp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    from src import settings
    settings.cargar()          # calienta la caché…
    settings.cargar.cache_clear()
    cfg = settings.cargar()
    cfg["database"]["ruta"] = tmp_db.name   # …y se redirige a la BD temporal

    from src import db
    from src.agents import AgenteRiesgo, AgenteSIG, Orquestador
    from src.tools import trafico

    print("\n══ 1. Agente SIG delimita ══")
    sig = AgenteSIG()
    manzanas = sig.listar_manzanas()
    verificar("capa de manzanas cargada", len(manzanas) > 0, f"{len(manzanas)} manzanas")
    cvegeo = manzanas[0]
    delim = sig.delimitar(cvegeo)
    verificar("delimitación de la manzana", "centroide" in delim and "bbox" in delim,
              f"cvegeo={cvegeo}, centroide=({delim['centroide']['lat']:.4f}, "
              f"{delim['centroide']['lon']:.4f})")
    dentro = sig.localizar(delim["centroide"]["lat"], delim["centroide"]["lon"])
    verificar("el centroide cae dentro de su manzana", dentro == cvegeo)

    print("\n══ 2. Visión (simulada) inserta detecciones en la BD ══")
    db.init_db()
    detecciones_sinteticas = [
        {"pisos": 4, "ventanas": 12, "conteo": {"window": 12, "crack": 2, "dirty_mold": 3}},
        {"pisos": 5, "ventanas": 15, "conteo": {"window": 15, "concrete_spalling": 1}},
        {"pisos": 4, "ventanas": 10, "conteo": {"window": 10}},
    ]
    for i, d in enumerate(detecciones_sinteticas):
        db.guardar_deteccion(f"sintetica_{i}.jpg",
                             delim["centroide"]["lat"], delim["centroide"]["lon"],
                             cvegeo, d["pisos"], d["pisos"] * 3.0, d["ventanas"],
                             d["conteo"], sum(v for k, v in d["conteo"].items()
                                              if k != "window"))
    guardadas = db.detecciones_por_manzana(cvegeo)
    verificar("detecciones persistidas por cvegeo", len(guardadas) == 3)

    print("\n══ 3. Tráfico: TomTom → fallback histórico ══")
    lectura = trafico.obtener_congestion_sync(delim["centroide"]["lat"],
                                              delim["centroide"]["lon"])
    verificar("lectura de congestión obtenida", 0.0 <= lectura.congestion <= 1.0,
              f"congestion={lectura.congestion} fuente={lectura.fuente}")
    verificar("fallback histórico funciona sin API key",
              lectura.fuente in ("tomtom", "historico"))

    print("\n══ 4. Agente de Riesgo calcula y persiste (clave cvegeo) ══")
    riesgo = AgenteRiesgo()
    resultado = riesgo.evaluar_manzana(cvegeo, centroide=delim["centroide"],
                                       area_satelital_m2=1200.0)
    verificar("score en rango [0,1]", 0.0 <= resultado["score_riesgo"] <= 1.0,
              f"score={resultado['score_riesgo']}")
    # daños ponderados esperados: 2·1.0 + 3·0.1 + 1·1.0 = 3.3
    verificar("ponderación por severidad de daños",
              abs(resultado["danos_ponderados"] - 3.3) < 1e-6,
              f"danos_ponderados={resultado['danos_ponderados']} (esperado 3.3)")
    verificar("factor de confianza (3 fotos ≈ 78%)",
              abs(resultado["confianza"] - 0.78) < 0.01,
              f"confianza={resultado['confianza']}")
    verificar("población expuesta (área×pisos/35)",
              resultado["poblacion_estimada"] == round(1200.0 * (13 / 3) / 35),
              f"poblacion={resultado['poblacion_estimada']}")

    print("\n══ 5. BD actualizada → lista para renderizar en Streamlit ══")
    ranking = riesgo.ranking()
    verificar("riesgo consultable desde la BD", any(r["cvegeo"] == cvegeo for r in ranking))
    fila = next(r for r in ranking if r["cvegeo"] == cvegeo)
    verificar("upsert idempotente por cvegeo",
              len([r for r in ranking if r["cvegeo"] == cvegeo]) == 1)
    verificar("misma cifra persistida", fila["score_riesgo"] == resultado["score_riesgo"])

    print("\n══ 6. Orquestador: flujo reactivo completo ══")
    orq = Orquestador()
    orq.riesgo = riesgo      # reutiliza la instancia apuntada a la BD temporal
    etapas = []
    orq.suscribir(lambda e: etapas.append(e["etapa"]))
    salida = orq.analizar_manzana_sync(cvegeo)
    verificar("eventos emitidos en orden",
              etapas[0] == "sig_delimitacion" and etapas[-1] == "flujo_completo",
              " → ".join(etapas))
    verificar("resultado consolidado del pipeline",
              salida["riesgo"]["cvegeo"] == cvegeo)

    if args.con_modelos:
        print("\n══ 7. (opcional) Inferencia YOLO11 real ══")
        import numpy as np
        from PIL import Image
        from src.agents import AgenteVision
        vision = AgenteVision()
        img = Image.fromarray((np.random.rand(320, 320, 3) * 255).astype("uint8"))
        analisis = vision.analizar_fachada(img, tipos=["ventanas"])
        verificar("pipeline de visión ejecuta sin errores",
                  "confianza_ventanas" in analisis,
                  f"ventanas={analisis['ventanas']} "
                  f"conf={analisis['confianza_ventanas']} "
                  f"decisiones={[d['evento'] for d in vision.decisiones]}")

    print("\n" + "═" * 50)
    if FALLOS:
        print(f"❌ {len(FALLOS)} verificación(es) fallaron: {FALLOS}")
        sys.exit(1)
    print("✅ Flujo completo verificado: SIG → Visión → Riesgo → BD → render.")


if __name__ == "__main__":
    main()
