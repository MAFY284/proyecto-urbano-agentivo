"""Orquestador — supervisor reactivo del sistema multi-agente.

Encadena el flujo completo de análisis de una manzana de forma asíncrona:

    Agente SIG delimita (cvegeo → bbox/centroide)
        → Agente Visión segmenta satélite e infiere fachadas
          (con corrección autónoma SAM3/Detectron2 si la confianza < 65%)
        → Agente Riesgo integra TomTom/histórico y calcula el score
        → la base de datos se actualiza (síncrono, clave cvegeo)

Es reactivo: emite eventos a los suscriptores conforme avanza el pipeline
(el dashboard y el CLI los consumen para mostrar progreso en tiempo real),
y las etapas pesadas de visión corren en executor para no bloquear el loop.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime

from src import db
from src.agents.agente_riesgo import AgenteRiesgo
from src.agents.agente_sig import AgenteSIG
from src.agents.agente_vision import AgenteVision

log = logging.getLogger("orquestador")


class Orquestador:
    def __init__(self):
        self.sig = AgenteSIG()
        self.vision = AgenteVision()
        self.riesgo = AgenteRiesgo()
        self._suscriptores = []
        self.eventos: list[dict] = []

    # ── Bus de eventos (programación reactiva) ──

    def suscribir(self, callback) -> None:
        """callback(evento: dict) — se invoca en cada transición del flujo."""
        self._suscriptores.append(callback)

    def _emitir(self, etapa: str, **datos) -> None:
        evento = {"etapa": etapa, "ts": datetime.now().isoformat(), **datos}
        self.eventos.append(evento)
        log.info("[%s] %s", etapa, datos)
        for cb in self._suscriptores:
            try:
                cb(evento)
            except Exception:
                pass

    # ── Flujo principal ──

    async def analizar_manzana(self, cvegeo: str,
                               imagenes_fachada: list | None = None,
                               imagen_satelital=None,
                               conf: float | None = None) -> dict:
        """Pipeline completo para una clave cvegeo.

        - `imagenes_fachada`: rutas de fotografías a nivel de calle (si el
          nombre codifica 'lat_lon.ext' se valida que caigan en la manzana).
        - `imagen_satelital`: imagen del área de la manzana (opcional).
        Sin imágenes nuevas, el riesgo se recalcula con lo acumulado en BD.
        """
        loop = asyncio.get_running_loop()

        # 1) SIG delimita
        delimitacion = self.sig.delimitar(cvegeo)
        self._emitir("sig_delimitacion", cvegeo=cvegeo,
                     centroide=delimitacion["centroide"])

        # 2) Visión — satélite (área en m² para población expuesta)
        area_satelital = None
        if imagen_satelital is not None:
            from src.tools.satelite import Esquinas
            bbox = delimitacion["bbox"]
            esquinas = Esquinas(bbox["lat_nw"], bbox["lon_nw"],
                                bbox["lat_se"], bbox["lon_se"])
            r_sat = await loop.run_in_executor(
                None, lambda: self.vision.analizar_satelite(imagen_satelital,
                                                            esquinas=esquinas))
            area_satelital = r_sat.area_total_m2
            self._emitir("vision_satelite", detecciones=r_sat.num_detecciones,
                         area_m2=round(area_satelital, 2))

        # 3) Visión — fachadas (con oráculo autónomo) + persistencia
        datos_vision = []
        for ruta in imagenes_fachada or []:
            geo = self.sig.localizar_archivo(str(ruta))
            analisis = await loop.run_in_executor(
                None, lambda r=ruta: self.vision.analizar_fachada(r, conf=conf))

            r_fachada = analisis["resultados"].get("fachada")
            conteo = r_fachada.conteo_clases if r_fachada else {}
            total_danos = r_fachada.total_danos if r_fachada else 0
            db.guardar_deteccion(str(ruta), geo["lat"], geo["lon"],
                                 geo["cvegeo"] or cvegeo,
                                 analisis["numero_pisos"],
                                 analisis["altura_aproximada"],
                                 analisis["ventanas"], conteo, total_danos)
            datos_vision.append({"conteo_clases": conteo,
                                 "numero_pisos": analisis["numero_pisos"]})
            self._emitir("vision_fachada", archivo=str(ruta),
                         ventanas=analisis["ventanas"],
                         pisos=analisis["numero_pisos"],
                         confianza=analisis["confianza_ventanas"],
                         correccion=(analisis["correccion"] or {}).get("motor"))

        # 4) Riesgo integra y guarda (síncrono, clave cvegeo)
        resultado = await loop.run_in_executor(
            None, lambda: self.riesgo.evaluar_manzana(
                cvegeo, centroide=delimitacion["centroide"],
                area_satelital_m2=area_satelital))
        self._emitir("riesgo_calculado", cvegeo=cvegeo,
                     score=resultado["score_riesgo"],
                     fuente_congestion=resultado["fuente_congestion"])

        self._emitir("flujo_completo", cvegeo=cvegeo)
        return {"delimitacion": delimitacion, "riesgo": resultado,
                "detecciones_nuevas": len(datos_vision),
                "area_satelital_m2": area_satelital}

    async def analizar_todas(self, con_imagenes: dict | None = None) -> list[dict]:
        """Recalcula el riesgo de todas las manzanas con datos en BD (y las
        que tengan imágenes nuevas en `con_imagenes: {cvegeo: [rutas]}`)."""
        con_imagenes = con_imagenes or {}
        cvegeos = set(con_imagenes)
        with db.conexion() as conn:
            cvegeos |= {r[0] for r in conn.execute(
                "SELECT DISTINCT cvegeo FROM detecciones WHERE cvegeo IS NOT NULL")}
            cvegeos |= {r[0] for r in conn.execute(
                "SELECT DISTINCT cvegeo FROM trafico_calles WHERE cvegeo IS NOT NULL")}
        resultados = []
        for cv in sorted(cvegeos):
            if not self.sig.existe(cv):
                continue
            resultados.append(await self.analizar_manzana(
                cv, imagenes_fachada=con_imagenes.get(cv)))
        return resultados

    # ── Envolturas síncronas ──

    def analizar_manzana_sync(self, cvegeo: str, **kwargs) -> dict:
        return asyncio.run(self.analizar_manzana(cvegeo, **kwargs))

    def analizar_todas_sync(self, **kwargs) -> list[dict]:
        return asyncio.run(self.analizar_todas(**kwargs))
