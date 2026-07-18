"""Acceso a la base de datos SQLite unificada (database/detecciones.db).

Hereda el esquema del proyecto 1 (tablas `detecciones` y `trafico_calles`) y
agrega `riesgo_manzanas`, donde el Agente de Riesgo persiste el score
combinado de forma síncrona, indexado por la clave espacial única `cvegeo`.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager

from src import settings


def ruta_db() -> str:
    return str(settings.ruta(settings.cargar()["database"]["ruta"]))


@contextmanager
def conexion():
    conn = sqlite3.connect(ruta_db())
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with conexion() as conn:
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
        c.execute('''CREATE TABLE IF NOT EXISTS trafico_calles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            vialidad TEXT,
            lat REAL,
            lon REAL,
            cvegeo TEXT,
            velocidad_actual REAL,
            velocidad_libre REAL,
            congestion REAL,
            fuente TEXT DEFAULT 'tomtom',
            fecha TEXT DEFAULT (datetime('now','localtime'))
        )''')
        # Resultado consolidado del Agente de Riesgo, una fila por manzana.
        c.execute('''CREATE TABLE IF NOT EXISTS riesgo_manzanas (
            cvegeo TEXT PRIMARY KEY,
            score_riesgo REAL,
            danos_norm REAL,
            congestion REAL,
            pisos_norm REAL,
            confianza REAL,
            num_fotos INTEGER,
            danos_ponderados REAL,
            altura_promedio_pisos REAL,
            area_satelital_m2 REAL,
            poblacion_estimada INTEGER,
            fuente_congestion TEXT,
            fecha TEXT DEFAULT (datetime('now','localtime'))
        )''')
        # Columna 'fuente' para bases heredadas del proyecto 1 (no la tenían).
        try:
            c.execute("ALTER TABLE trafico_calles ADD COLUMN fuente TEXT DEFAULT 'tomtom'")
        except sqlite3.OperationalError:
            pass
        c.execute('CREATE INDEX IF NOT EXISTS idx_detecciones_cvegeo ON detecciones(cvegeo)')
        c.execute('CREATE INDEX IF NOT EXISTS idx_detecciones_fecha ON detecciones(fecha)')
        c.execute('CREATE INDEX IF NOT EXISTS idx_trafico_cvegeo ON trafico_calles(cvegeo)')
        c.execute('CREATE INDEX IF NOT EXISTS idx_trafico_fecha ON trafico_calles(fecha)')


def guardar_deteccion(archivo, lat, lon, cvegeo, numero_pisos, altura_aproximada,
                      ventanas, conteo_clases, danos_total=0) -> None:
    with conexion() as conn:
        conn.execute(
            '''INSERT INTO detecciones (archivo, lat, lon, cvegeo, numero_pisos,
               altura_aproximada, ventanas, conteo_clases, danos_total)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
            (archivo, lat, lon, cvegeo, numero_pisos, altura_aproximada,
             ventanas, json.dumps(conteo_clases), danos_total))


def guardar_trafico(vialidad, lat, lon, cvegeo, velocidad_actual,
                    velocidad_libre, congestion, fuente='tomtom') -> None:
    with conexion() as conn:
        conn.execute(
            '''INSERT INTO trafico_calles (vialidad, lat, lon, cvegeo,
               velocidad_actual, velocidad_libre, congestion, fuente)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
            (vialidad, lat, lon, cvegeo, velocidad_actual, velocidad_libre,
             congestion, fuente))


def guardar_riesgo(resultado: dict) -> None:
    """UPSERT del score de una manzana (clave única: cvegeo)."""
    with conexion() as conn:
        conn.execute(
            '''INSERT INTO riesgo_manzanas (cvegeo, score_riesgo, danos_norm,
               congestion, pisos_norm, confianza, num_fotos, danos_ponderados,
               altura_promedio_pisos, area_satelital_m2, poblacion_estimada,
               fuente_congestion, fecha)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now','localtime'))
               ON CONFLICT(cvegeo) DO UPDATE SET
                 score_riesgo=excluded.score_riesgo,
                 danos_norm=excluded.danos_norm,
                 congestion=excluded.congestion,
                 pisos_norm=excluded.pisos_norm,
                 confianza=excluded.confianza,
                 num_fotos=excluded.num_fotos,
                 danos_ponderados=excluded.danos_ponderados,
                 altura_promedio_pisos=excluded.altura_promedio_pisos,
                 area_satelital_m2=excluded.area_satelital_m2,
                 poblacion_estimada=excluded.poblacion_estimada,
                 fuente_congestion=excluded.fuente_congestion,
                 fecha=datetime('now','localtime')''',
            (resultado["cvegeo"], resultado["score_riesgo"], resultado["danos_norm"],
             resultado["congestion"], resultado["pisos_norm"], resultado["confianza"],
             resultado["num_fotos"], resultado["danos_ponderados"],
             resultado["altura_promedio_pisos"], resultado.get("area_satelital_m2"),
             resultado.get("poblacion_estimada"), resultado.get("fuente_congestion")))


def detecciones_por_manzana(cvegeo: str) -> list[dict]:
    with conexion() as conn:
        filas = conn.execute(
            '''SELECT archivo, conteo_clases, numero_pisos, ventanas, danos_total, fecha
               FROM detecciones WHERE cvegeo = ?''', (cvegeo,)).fetchall()
    out = []
    for archivo, conteo_json, pisos, ventanas, danos, fecha in filas:
        try:
            conteo = json.loads(conteo_json) if conteo_json else {}
        except (TypeError, ValueError):
            conteo = {}
        out.append({"archivo": archivo, "conteo_clases": conteo, "numero_pisos": pisos,
                    "ventanas": ventanas, "danos_total": danos, "fecha": fecha})
    return out


def congestion_por_manzana(cvegeo: str, horas_pico: list) -> tuple[float | None, bool]:
    """Congestión promedio de la manzana priorizando lecturas en horas pico.

    Retorna (congestion, es_hora_pico) o (None, False) si no hay lecturas.
    """
    condiciones = ' OR '.join(
        f"(CAST(strftime('%H', fecha) AS INTEGER) BETWEEN {h1} AND {h2})"
        for h1, h2 in horas_pico)
    with conexion() as conn:
        pico = conn.execute(
            f'''SELECT AVG(congestion) FROM trafico_calles
                WHERE cvegeo = ? AND congestion IS NOT NULL AND ({condiciones})''',
            (cvegeo,)).fetchone()[0]
        if pico is not None:
            return pico, True
        todas = conn.execute(
            '''SELECT AVG(congestion) FROM trafico_calles
               WHERE cvegeo = ? AND congestion IS NOT NULL''', (cvegeo,)).fetchone()[0]
    return todas, False


def listar_riesgos() -> list[dict]:
    with conexion() as conn:
        conn.row_factory = sqlite3.Row
        filas = conn.execute(
            'SELECT * FROM riesgo_manzanas ORDER BY score_riesgo DESC').fetchall()
    return [dict(f) for f in filas]


def purgar_trafico_antiguo(dias: int) -> int:
    with conexion() as conn:
        cur = conn.execute(
            "DELETE FROM trafico_calles WHERE fecha < datetime('now','localtime', ?)",
            (f'-{dias} days',))
        return cur.rowcount
