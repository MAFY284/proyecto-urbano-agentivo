"""Telemetría de tráfico: TomTom Traffic Flow API (asíncrona) + matriz de
fallback de curvas históricas de congestión de La Condesa.

Hereda trafico_tomtom.py del proyecto 1 y la metodología de índices de
movilidad de la Fase 1 del Proyecto-Delfin. Regla central: toda consulta a
la API vive dentro de un try-except — si TomTom falla, no hay API key o se
agotaron los créditos, la función retorna automáticamente el valor del
perfil histórico correspondiente al día/hora de la consulta, marcando la
fuente ('tomtom' o 'historico') para trazabilidad.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache

from src import settings

DIAS_ES = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]


@dataclass
class LecturaCongestion:
    congestion: float           # 0-1 (0 = flujo libre, 1 = detenido)
    fuente: str                 # 'tomtom' | 'historico'
    velocidad_actual: float | None = None
    velocidad_libre: float | None = None
    detalle: str = ""


def _cfg() -> dict:
    return settings.cargar()["trafico"]


def api_key() -> str | None:
    """La key SIEMPRE se lee de la variable de entorno (TOMTOM_API_KEY);
    nunca debe escribirse en un archivo del repositorio.

    ── Configurar la key (gratis en https://developer.tomtom.com/) ──
      bash/zsh:  export TOMTOM_API_KEY="tu_key"      # agrégalo a ~/.bashrc para que persista
      fish:      set -Ux TOMTOM_API_KEY "tu_key"     # variable universal, persiste sola

    ── Recolección automática cada 30 minutos ──
      python main.py trafico --loop                  # intervalo por defecto: 1800 s (30 min)
      python main.py trafico --loop --intervalo 900  # u otro intervalo en segundos
    O programado con cron (sin dejar el proceso abierto):
      */30 * * * * cd /ruta/al/repo && venv/bin/python main.py trafico
    Sin key configurada el sistema no falla: usa el perfil histórico 7×24.
    """
    return os.environ.get(_cfg()["api_key_env"])


# ── Perfil histórico (matriz 7 días × 24 horas) ──

@lru_cache(maxsize=1)
def cargar_curvas_historicas() -> dict:
    with open(settings.ruta(_cfg()["curvas_historicas"]), "r", encoding="utf-8") as f:
        return json.load(f)["curvas"]


def congestion_historica(dia_semana: str | None = None, hora: int | None = None) -> float:
    """Valor del perfil histórico para (día, hora); por defecto el momento
    actual de la consulta."""
    ahora = datetime.now()
    dia = dia_semana or DIAS_ES[ahora.weekday()]
    h = hora if hora is not None else ahora.hour
    return cargar_curvas_historicas()[dia][h % 24]


# ── Consulta asíncrona a TomTom con fallback automático ──

async def obtener_flujo_tomtom(lat: float, lon: float, key: str,
                               client=None) -> tuple[float, float]:
    """Velocidades (actual, flujo libre) en km/h del segmento más cercano al
    punto. Lanza excepción si la API falla — el manejo vive en
    obtener_congestion()."""
    import httpx
    cfg = _cfg()
    params = {"point": f"{lat},{lon}", "key": key}
    async def _consulta(c):
        r = await c.get(cfg["url"], params=params, timeout=cfg["timeout_s"])
        r.raise_for_status()
        data = r.json()["flowSegmentData"]
        return data["currentSpeed"], data["freeFlowSpeed"]
    if client is not None:
        return await _consulta(client)
    async with httpx.AsyncClient() as c:
        return await _consulta(c)


async def obtener_congestion(lat: float, lon: float, client=None,
                             dia_semana: str | None = None,
                             hora: int | None = None) -> LecturaCongestion:
    """Congestión (0-1) en un punto. Intenta TomTom; ante CUALQUIER fallo
    (sin key, sin créditos, timeout, respuesta corrupta) cae al perfil
    histórico del día/hora de la consulta — el sistema nunca se queda sin
    señal de tráfico."""
    key = api_key()
    if key:
        try:
            vel_actual, vel_libre = await obtener_flujo_tomtom(lat, lon, key, client=client)
            if vel_actual is not None and vel_libre:
                congestion = round(max(0.0, min(1.0, 1 - vel_actual / vel_libre)), 3)
                return LecturaCongestion(congestion=congestion, fuente="tomtom",
                                         velocidad_actual=vel_actual,
                                         velocidad_libre=vel_libre)
        except Exception as e:
            detalle = f"TomTom falló ({type(e).__name__}: {e}); usando perfil histórico"
        else:
            detalle = "TomTom regresó velocidades inválidas; usando perfil histórico"
    else:
        detalle = "Sin TOMTOM_API_KEY; usando perfil histórico"

    return LecturaCongestion(congestion=congestion_historica(dia_semana, hora),
                             fuente="historico", detalle=detalle)


def obtener_congestion_sync(lat: float, lon: float, **kwargs) -> LecturaCongestion:
    """Envoltura síncrona para llamadores no-async (CLI, Streamlit)."""
    return asyncio.run(obtener_congestion(lat, lon, **kwargs))


# ── Recolección por lote (calles de referencia del Excel) ──

def cargar_calles_referencia() -> list[dict]:
    """Calles del Excel heredado; el punto de consulta es el punto medio del
    segmento (aproximación razonable para calles cortas)."""
    import pandas as pd
    df = pd.read_excel(settings.ruta(settings.cargar()["geoespacial"]["coordenadas_trafico"]), header=1)
    calles = []
    for _, row in df.iterrows():
        calles.append({
            "vialidad": row["Vialidad"],
            "lat": (row["ini_lat"] + row["dest_lat"]) / 2,
            "lon": (row["ini_lon"] + row["dest_lon"]) / 2,
        })
    return calles


async def recolectar_trafico(buscar_manzana=None, pausa_s: float = 0.5) -> list[dict]:
    """Recolecta congestión para todas las calles de referencia y la persiste
    en trafico_calles (con purga automática de lecturas > retencion_dias).

    `buscar_manzana`: callable (lat, lon) -> cvegeo | None (lo aporta el
    Agente SIG); si no se pasa, las lecturas se guardan sin CVEGEO.
    """
    import httpx
    from src import db

    db.init_db()
    purgados = db.purgar_trafico_antiguo(_cfg()["retencion_dias"])
    if purgados:
        print(f"🗑️  Purgados {purgados} registros de tráfico antiguos")

    resultados = []
    async with httpx.AsyncClient() as client:
        for calle in cargar_calles_referencia():
            lectura = await obtener_congestion(calle["lat"], calle["lon"], client=client)
            cvegeo = buscar_manzana(calle["lat"], calle["lon"]) if buscar_manzana else None
            db.guardar_trafico(calle["vialidad"], calle["lat"], calle["lon"], cvegeo,
                               lectura.velocidad_actual, lectura.velocidad_libre,
                               lectura.congestion, fuente=lectura.fuente)
            resultados.append({**calle, "cvegeo": cvegeo,
                               "congestion": lectura.congestion, "fuente": lectura.fuente})
            # margen prudente frente al límite de tasa del nivel gratuito
            if lectura.fuente == "tomtom":
                await asyncio.sleep(pausa_s)
    return resultados


def recolectar_trafico_sync(**kwargs) -> list[dict]:
    return asyncio.run(recolectar_trafico(**kwargs))


def recolectar_en_loop(intervalo_s: int = 1800, **kwargs) -> None:
    """Recolecta indefinidamente cada `intervalo_s` (heredado de --loop)."""
    while True:
        print(f"\n🚦 Recolectando tráfico — {datetime.now().isoformat()}")
        recolectar_trafico_sync(**kwargs)
        time.sleep(intervalo_s)
