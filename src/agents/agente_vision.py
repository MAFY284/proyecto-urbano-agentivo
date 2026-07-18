"""Agente de Visión — control de todas las inferencias del sistema.

Tres frentes:
  - Satélite: segmentación de techos/edificios (tools/satelite, YOLOv8-XL).
  - Fachadas: pool de 7 modelos YOLO11 (tools/fachada).
  - Oráculo: lógica de decisión AUTÓNOMA sobre la detección de ventanas —
    evalúa la confianza promedio del YOLO11 y, si cae por debajo del umbral
    (65% por defecto, settings.yaml → vision.umbral_confianza_oraculo),
    invoca por su cuenta tools/comparador (Detectron2 y/o SAM3) para
    refinar el conteo antes de mandar los datos al módulo de fusión.
"""

from __future__ import annotations

import logging

from src import settings
from src.tools import comparador, fachada, satelite

log = logging.getLogger("agente_vision")


class AgenteVision:
    def __init__(self, umbral_oraculo: float | None = None):
        cfg = settings.cargar()
        self.umbral_oraculo = (umbral_oraculo if umbral_oraculo is not None
                               else cfg["vision"]["umbral_confianza_oraculo"])
        self.decisiones: list[dict] = []   # bitácora de decisiones autónomas

    # ── Satélite ──

    def analizar_satelite(self, imagen, esquinas=None, conf: float = 0.5,
                          tile_size: int | None = None, overlap: int | None = None,
                          escala_m_px: float | None = None,
                          progreso=None) -> satelite.ResultadoSatelite:
        """Segmenta edificios en una imagen satelital y retorna área (m²),
        polígonos y GeoJSON georreferenciado. `tile_size`/`overlap` permiten
        ajustar el tiling desde la interfaz."""
        return satelite.segmentar_imagen(imagen, esquinas=esquinas, conf=conf,
                                         tile_size=tile_size, overlap=overlap,
                                         escala_m_px=escala_m_px, progreso=progreso)

    # ── Fachadas + oráculo ──

    def _decidir(self, evento: str, detalle: dict) -> None:
        registro = {"evento": evento, **detalle}
        self.decisiones.append(registro)
        log.info("decisión autónoma: %s — %s", evento, detalle)

    def _corregir_con_oraculo(self, imagen, conf: float) -> dict | None:
        """Segunda opinión con los motores finos: Detectron2 primero (mismo
        proceso, barato), SAM3 después (subproceso aislado). Retorna el
        resultado del primer motor que responda, o None."""
        candidatos = []
        if comparador.detectron2_disponible():
            candidatos.append(comparador.MOTOR_DETECTRON2)
        if comparador.sam3_disponible():
            candidatos.append(comparador.MOTOR_SAM3)

        for motor in candidatos:
            try:
                r = comparador.inferir(motor, imagen, conf=conf)
                if r.error is None:
                    return {"motor": motor, "resultado": r}
            except Exception as e:
                log.warning("oráculo %s falló: %s", motor, e)
        return None

    def analizar_fachada(self, imagen, tipos: list[str] | None = None,
                         conf: float | None = None) -> dict:
        """Corre el pool sobre la imagen y aplica la lógica del oráculo.

        Si la confianza promedio de las ventanas YOLO11 es inferior al
        umbral, invoca de manera autónoma el comparador para activar la
        segmentación fina y corregir el conteo de ventanas/pisos antes de
        la fusión. Retorna:
          {resultados: {tipo: ResultadoFachada}, altura_estimada, ventanas,
           numero_pisos, confianza_ventanas, correccion: dict|None}
        """
        tipos = tipos or ["fachada", "ventanas"]
        conf = fachada.parsear_conf(conf)

        resultados = fachada.detectar_multiple(imagen, tipos, conf=conf)

        # Confianza de la detección de ventanas del pool YOLO11
        r_ventanas = resultados.get("ventanas") or resultados.get("fachada")
        confianza = r_ventanas.confianza_promedio if r_ventanas else 0.0
        sin_ventanas = r_ventanas is None or r_ventanas.ventanas == 0

        correccion = None
        if r_ventanas is not None and (confianza < self.umbral_oraculo or sin_ventanas):
            self._decidir("oraculo_invocado", {
                "confianza_yolo11": round(confianza, 3),
                "umbral": self.umbral_oraculo,
                "motivo": "sin ventanas detectadas" if sin_ventanas else "confianza baja",
            })
            fino = self._corregir_con_oraculo(imagen, conf)
            if fino is not None:
                r_fino = fino["resultado"]
                # Cajas del motor fino → (yc, h) normalizados → pisos
                import numpy as np
                from PIL import Image
                if isinstance(imagen, Image.Image):
                    alto = imagen.height
                else:
                    alto = Image.open(imagen).height
                boxes_norm = [(((y1 + y2) / 2) / alto, (y2 - y1) / alto)
                              for _, y1, _, y2 in np.array(r_fino.cajas).reshape(-1, 4)]
                pisos_corregidos = fachada.estimar_pisos_por_filas(boxes_norm)
                correccion = {
                    "motor": fino["motor"],
                    "ventanas": r_fino.n_detecciones,
                    "numero_pisos": pisos_corregidos,
                    "confianza": r_fino.confianza_promedio,
                }
                self._decidir("conteo_corregido", correccion)
            else:
                self._decidir("oraculo_sin_motores", {})

        altura = fachada.combinar_altura_estimada(resultados)

        # Fusión final: si el oráculo corrigió con mejor confianza, sus números mandan
        ventanas_final = r_ventanas.ventanas if r_ventanas else 0
        pisos_final = altura["numero_pisos"] if altura else 0
        if correccion and (correccion["confianza"] >= confianza or ventanas_final == 0):
            ventanas_final = correccion["ventanas"]
            if correccion["numero_pisos"]:
                pisos_final = correccion["numero_pisos"]

        altura_piso = settings.cargar()["inferencia"]["altura_piso_m"]
        return {
            "resultados": resultados,
            "altura_estimada": altura,
            "ventanas": ventanas_final,
            "numero_pisos": pisos_final,
            "altura_aproximada": round(pisos_final * altura_piso, 1),
            "confianza_ventanas": round(confianza, 3),
            "correccion": correccion,
        }

    # ── Comparador (acceso directo, para el Laboratorio del dashboard) ──

    def comparar_motores(self, imagen, motores=None, conf: float = 0.3,
                         prompt_sam3: str | None = None) -> dict:
        return comparador.comparar(imagen, motores=motores, conf=conf,
                                   prompt_sam3=prompt_sam3)
