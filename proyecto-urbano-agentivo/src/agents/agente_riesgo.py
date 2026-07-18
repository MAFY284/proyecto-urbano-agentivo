"""Agente de Riesgo — fórmula matemática unificada de Riesgo Urbano Combinado.

Integra los outputs del Agente SIG (delimitación por cvegeo) y del Agente de
Visión (daños, pisos, área satelital) con la telemetría de tráfico
(TomTom → perfil histórico como fallback), y guarda el resultado de forma
SÍNCRONA en database/detecciones.db indexado por la clave única `cvegeo`.

Fórmula (heredada del proyecto 1, parametrizada en settings.yaml → riesgo):

    score = danos_norm·0.4 + congestion·0.3 + pisos_norm·0.3

    danos_norm  = min(1, danos_ponderados / 10) · confianza
    pisos_norm  = min(1, pisos_promedio / 10) · confianza
    confianza   = 1 - e^(-0.5 · num_fotos)     (amortigua muestras chicas)

con daños ponderados por severidad (estructural ×1.0, acabados ×0.5,
estético ×0.1) y congestión priorizando lecturas en horas pico.
"""

from __future__ import annotations

import logging
import math

from src import db, settings
from src.tools import fachada, trafico

log = logging.getLogger("agente_riesgo")


class AgenteRiesgo:
    def __init__(self):
        cfg = settings.cargar()["riesgo"]
        self.peso_danos = cfg["peso_danos"]
        self.peso_congestion = cfg["peso_congestion"]
        self.peso_altura = cfg["peso_altura"]
        self.danos_normalizacion = cfg["danos_normalizacion"]
        self.pisos_normalizacion = cfg["pisos_normalizacion"]
        self.factor_k = cfg["factor_confianza_k"]
        self.horas_pico = [tuple(h) for h in cfg["horas_pico"]]
        self.m2_por_habitante = settings.cargar()["poblacion"]["m2_por_habitante"]
        db.init_db()

    # ── Componentes de la fórmula ──

    def factor_confianza(self, num_fotos: int) -> float:
        """1 foto ≈ 39%, 3 ≈ 78%, 5 ≈ 92%, 10+ ≈ 100% — una sola foto con un
        hallazgo severo no debe pesar igual que 50 que confirman lo mismo."""
        return 1 - math.exp(-self.factor_k * num_fotos)

    def calcular_score(self, danos_ponderados: float, congestion: float,
                       pisos_promedio: float, num_fotos: int) -> dict:
        """Fórmula de Riesgo Urbano Combinado, pura y reutilizable (también
        la usa el Simulador de Crisis del dashboard con valores alterados)."""
        confianza = self.factor_confianza(num_fotos)
        danos_norm = min(1.0, danos_ponderados / self.danos_normalizacion) * confianza
        pisos_norm = min(1.0, pisos_promedio / self.pisos_normalizacion) * confianza
        score = round(danos_norm * self.peso_danos
                      + congestion * self.peso_congestion
                      + pisos_norm * self.peso_altura, 3)
        return {"score_riesgo": score, "danos_norm": round(danos_norm, 3),
                "pisos_norm": round(pisos_norm, 3), "confianza": round(confianza, 2),
                "congestion": round(congestion, 3)}

    def poblacion_estimada(self, area_m2: float | None, pisos: float) -> int | None:
        """Población expuesta por densidad normativa: 35 m² construidos por
        habitante (área de desplante × pisos / 35)."""
        if not area_m2 or not pisos:
            return None
        return round(area_m2 * max(1, pisos) / self.m2_por_habitante)

    # ── Congestión con fallback ──

    def congestion_de_manzana(self, cvegeo: str, centroide: dict | None = None) -> tuple[float, str]:
        """Prioridad: lecturas en BD en horas pico > lecturas en BD de todas
        las horas > consulta en vivo (TomTom → perfil histórico) sobre el
        centroide de la manzana. Retorna (congestion, fuente)."""
        valor, es_pico = db.congestion_por_manzana(cvegeo, self.horas_pico)
        if valor is not None:
            return round(valor, 3), "bd_hora_pico" if es_pico else "bd_todas_horas"
        if centroide is not None:
            lectura = trafico.obtener_congestion_sync(centroide["lat"], centroide["lon"])
            return lectura.congestion, lectura.fuente
        return trafico.congestion_historica(), "historico"

    # ── Evaluación completa de una manzana ──

    def evaluar_manzana(self, cvegeo: str, centroide: dict | None = None,
                        area_satelital_m2: float | None = None,
                        datos_vision: list[dict] | None = None) -> dict:
        """Aplica la fórmula unificada a una manzana y persiste el resultado.

        `datos_vision`: resultados frescos del Agente de Visión aún no
        guardados (opcional); se combinan con lo acumulado en la BD.
        """
        detecciones = db.detecciones_por_manzana(cvegeo)
        if datos_vision:
            detecciones = detecciones + datos_vision

        num_fotos = len(detecciones)
        danos_ponderados, danos_crudos, pisos = 0.0, 0, []
        for det in detecciones:
            conteo = det.get("conteo_clases", {})
            danos_ponderados += fachada.calcular_danos_ponderados(conteo)
            danos_crudos += sum(v for k, v in conteo.items() if k in fachada.CLASES_DANO)
            if det.get("numero_pisos"):
                pisos.append(det["numero_pisos"])
        pisos_promedio = sum(pisos) / len(pisos) if pisos else 0

        congestion, fuente_congestion = self.congestion_de_manzana(cvegeo, centroide)

        score = self.calcular_score(danos_ponderados, congestion, pisos_promedio, num_fotos)
        resultado = {
            "cvegeo": cvegeo,
            **score,
            "num_fotos": num_fotos,
            "total_danos": danos_crudos,
            "danos_ponderados": round(danos_ponderados, 1),
            "altura_promedio_pisos": round(pisos_promedio, 1),
            "area_satelital_m2": area_satelital_m2,
            "poblacion_estimada": self.poblacion_estimada(area_satelital_m2, pisos_promedio),
            "fuente_congestion": fuente_congestion,
        }

        # Persistencia SÍNCRONA, indexada por cvegeo (UPSERT)
        db.guardar_riesgo(resultado)
        log.info("riesgo guardado — cvegeo=%s score=%s (congestión: %s)",
                 cvegeo, resultado["score_riesgo"], fuente_congestion)
        return resultado

    def ranking(self) -> list[dict]:
        """Manzanas ordenadas de mayor a menor riesgo (desde la BD)."""
        return db.listar_riesgos()
