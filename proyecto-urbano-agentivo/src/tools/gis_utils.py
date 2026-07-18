"""Utilidades geoespaciales.

Dos orígenes:
1. Operaciones de manzanas del proyecto 1 (cargar GeoJSON con CVEGEO,
   localizar un punto, extraer coordenadas del nombre de archivo).
2. Los pasos de limpieza y agregación espacial de los notebooks del
   Proyecto-Delfin (Limpieza_Escenarios.ipynb e
   Integracion_Espacial_Fase2.ipynb), traducidos a código Python plano:
   auditoría del CSV crudo, índice de movilidad, escenarios día+hora,
   tablas ancha/larga, agregación por hexágono y capa temporal para QGIS.

Cada función de limpieza recibe/retorna DataFrames y llena una bitácora de
decisiones (misma filosofía de auditoría citable de la tesis).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

DIAS_ES = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
COLUMNAS_HORAS = [f"{h:02d}h" for h in range(24)]


# ══════════════════════════════════════════════════════════════════════
# 1) Manzanas (CVEGEO) — heredado del proyecto 1
# ══════════════════════════════════════════════════════════════════════

def cargar_manzanas(ruta_geojson) -> tuple[dict, list]:
    """Retorna (geojson_crudo, lista de {'cvegeo', 'shape'}) con shapely."""
    from shapely.geometry import shape
    with open(ruta_geojson, "r", encoding="utf-8") as f:
        geojson = json.load(f)
    shapes = []
    for feat in geojson.get("features", []):
        try:
            shapes.append({"cvegeo": feat["properties"].get("CVEGEO", ""),
                           "shape": shape(feat["geometry"])})
        except Exception:
            pass
    return geojson, shapes


def buscar_manzana(shapes: list, lat: float, lon: float) -> str | None:
    """CVEGEO de la manzana que contiene el punto, o None."""
    from shapely.geometry import Point
    pt = Point(lon, lat)   # shapely usa (x=lon, y=lat)
    for m in shapes:
        if m["shape"].contains(pt):
            return m["cvegeo"]
    return None


def extraer_coordenadas_de_nombre(nombre_archivo: str) -> tuple[float | None, float | None]:
    """Extrae (lat, lon) de nombres tipo '19.4102_-99.1684.jpg'."""
    base = Path(nombre_archivo).stem
    match = re.match(r'(-?\d+\.\d+)_(-?\d+\.\d+)', base)
    if match:
        return float(match.group(1)), float(match.group(2))
    return None, None


def importar_csv_sam3(ruta_csv, buscar_manzana=None) -> dict:
    """Importa el CSV de alturas estimadas por SAM3 (columnas: archivo, lat,
    lon, ventanas_detectadas, pisos_detectados) a la tabla `detecciones`,
    para que el modelo de predicción de riesgo lo use como muestra de
    altura/exposición por manzana.

    Idempotente: borra las filas previas con los mismos nombres de archivo
    antes de insertar, así el CSV puede re-importarse sin duplicar.
    """
    import pandas as pd

    from src import db, settings

    df = pd.read_csv(ruta_csv)
    columnas = {"archivo", "lat", "lon", "ventanas_detectadas", "pisos_detectados"}
    faltan = columnas - set(df.columns)
    if faltan:
        raise ValueError(f"El CSV no tiene las columnas esperadas: faltan {sorted(faltan)}")

    db.init_db()
    altura_piso = settings.cargar()["inferencia"]["altura_piso_m"]

    filas, con_manzana = [], 0
    for _, fila in df.iterrows():
        lat, lon = float(fila["lat"]), float(fila["lon"])
        ventanas = int(fila["ventanas_detectadas"])
        pisos = int(fila["pisos_detectados"])
        cvegeo = buscar_manzana(lat, lon) if buscar_manzana else None
        if cvegeo:
            con_manzana += 1
        filas.append((str(fila["archivo"]), lat, lon, cvegeo, pisos,
                      round(pisos * altura_piso, 1), ventanas,
                      json.dumps({"window": ventanas}), 0))

    with db.conexion() as conn:
        conn.executemany("DELETE FROM detecciones WHERE archivo = ?",
                         [(f[0],) for f in filas])
        conn.executemany(
            '''INSERT INTO detecciones (archivo, lat, lon, cvegeo, numero_pisos,
               altura_aproximada, ventanas, conteo_clases, danos_total)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''', filas)

    return {"insertadas": len(filas), "con_manzana": con_manzana,
            "sin_manzana": len(filas) - con_manzana}


# ══════════════════════════════════════════════════════════════════════
# 2) Limpieza de escenarios (Fase 1 — Limpieza_Escenarios.ipynb)
# ══════════════════════════════════════════════════════════════════════

def auditar_csv_crudo(df, bitacora: list | None = None):
    """Paso 1: auditoría defensiva del CSV crudo de tráfico. Marca filas con
    'lat' no convertible a número (formato corrupto) sin corregirlas a mano
    — no se inventan coordenadas."""
    bitacora = bitacora if bitacora is not None else []

    def es_convertible(v):
        try:
            float(v)
            return True
        except (TypeError, ValueError):
            return False

    df = df.copy()
    df["lat_valida"] = df["lat"].apply(es_convertible)
    n_invalidas = int((~df["lat_valida"]).sum())
    bitacora.append(f"Filas con 'lat' no numérico (formato corrupto): {n_invalidas}. "
                    "Decisión: se excluyen del cálculo; no se corrigen a mano.")
    return df


def filtrar_fuente(df, fuente_valida: str = "google",
                   cols_numericas: tuple = ("travel_time_min", "traffic_delay_min",
                                            "distance_m", "current_speed_kmh",
                                            "free_speed_kmh"),
                   bitacora: list | None = None):
    """Paso 2: solo la fuente con cobertura temporal representativa entra al
    promedio histórico (mezclar una muestra grande y bien distribuida con
    una minúscula violaría el supuesto de 'comportamiento típico').
    Los nulos numéricos restantes se eliminan — no se imputan valores de
    congestión, porque inventarlos sesgaría el índice de riesgo."""
    bitacora = bitacora if bitacora is not None else []
    df = df[df["source"] == fuente_valida].copy()
    bitacora.append(f"Se conservan solo filas con source == '{fuente_valida}' ({len(df)} filas).")
    cols = [c for c in cols_numericas if c in df.columns]
    nulos = int(df[cols].isna().sum().sum())
    if nulos:
        df = df.dropna(subset=cols)
        bitacora.append(f"Eliminadas filas con nulos numéricos ({nulos} valores); no se imputa.")
    return df


def unificar_nombres_avenida(df, bitacora: list | None = None):
    """Paso 3: nombre canónico = street_qgis (ya validado contra OSM); si
    falta, se conserva el original marcado PENDIENTE_REVISION_MANUAL en vez
    de adivinar por similitud de texto."""
    import pandas as pd
    bitacora = bitacora if bitacora is not None else []

    def canonico(row):
        if pd.notna(row["street_qgis"]):
            return row["street_qgis"]
        return f"{row['street']} [PENDIENTE_REVISION_MANUAL]"

    df = df.copy()
    df["avenida"] = df.apply(canonico, axis=1)
    pendientes = sorted(a for a in df["avenida"].unique() if "PENDIENTE_REVISION_MANUAL" in a)
    bitacora.append(f"Avenidas marcadas para revisión manual: {pendientes or 'ninguna'}")
    return df


def parsear_timestamp_no_estandar(valor: str):
    """El timestamp crudo usa ':' entre año y mes ('2024:09-12T19:20:08.905383');
    se normaliza SOLO el primer ':' antes de parsear."""
    import pandas as pd
    return pd.to_datetime(valor.replace(":", "-", 1), format="%Y-%m-%dT%H:%M:%S.%f")


def construir_escenarios(df, bitacora: list | None = None):
    """Pasos 4-5: parsea fecha_hora y construye el escenario 'día + hora'
    (168 combinaciones posibles: 7 días × 24 horas)."""
    bitacora = bitacora if bitacora is not None else []
    df = df.copy()
    df["fecha_hora"] = df["timestamp"].apply(parsear_timestamp_no_estandar)
    n_malas = int(df["fecha_hora"].isna().sum())
    assert n_malas == 0, "Hay timestamps sin parsear: revisar formato antes de continuar."
    df["dia_semana"] = df["fecha_hora"].dt.weekday.map(lambda i: DIAS_ES[i])
    df["hora_dia"] = df["fecha_hora"].dt.hour
    df["escenario"] = df["dia_semana"] + "_" + df["hora_dia"].astype(str).str.zfill(2) + "h"
    bitacora.append(f"Rango de fechas cubierto: {df['fecha_hora'].min()} a {df['fecha_hora'].max()}")
    return df


def calcular_indice_movilidad(df, bitacora: list | None = None):
    """Paso 6: índice adimensional comparable entre avenidas de distinta
    longitud: current_speed_kmh / free_speed_kmh. Cercano a 1 = flujo libre;
    cercano a 0 = congestión. NO se invierte en esta fase — la relación con
    el riesgo se establece al formular el índice dinámico."""
    bitacora = bitacora if bitacora is not None else []
    df = df.copy()
    df["indice_movilidad"] = df["current_speed_kmh"] / df["free_speed_kmh"]
    bitacora.append("Índice de movilidad = current_speed / free_speed (sin invertir en esta fase).")
    return df


def agregar_por_escenario(df, umbral_n_minimo: int = 3, bitacora: list | None = None):
    """Pasos 7-8: tabla larga (avenida × escenario con media/σ/n y bandera de
    confiabilidad) y tabla ancha ordenada cronológicamente (Lunes_00h …
    Domingo_23h). Las celdas con pocas observaciones NO se eliminan; solo se
    marcan para usarse con cautela."""
    bitacora = bitacora if bitacora is not None else []
    tabla_larga = (df.groupby(["avenida", "dia_semana", "hora_dia", "escenario"])["indice_movilidad"]
                   .agg(media="mean", desviacion_estandar="std", n_observaciones="count")
                   .reset_index())
    tabla_larga["confiable"] = tabla_larga["n_observaciones"] >= umbral_n_minimo

    tabla_ancha = tabla_larga.pivot_table(index="avenida", columns="escenario", values="media")
    orden = [f"{d}_{h:02d}h" for d in DIAS_ES for h in range(24)]
    tabla_ancha = tabla_ancha[[c for c in orden if c in tabla_ancha.columns]]

    pct = 100 * tabla_larga["confiable"].mean()
    bitacora.append(f"Celdas con ≥{umbral_n_minimo} observaciones: {pct:.1f}% del total.")
    return tabla_larga, tabla_ancha


def tablas_por_dia(tabla_ancha) -> dict:
    """Paso 9.1: una tabla por día (24 columnas '00h'…'23h') para facilitar el
    join espacial en QGIS sin arrastrar 168 columnas."""
    export = tabla_ancha.reset_index()
    salidas = {}
    for dia in DIAS_ES:
        columnas = [c for c in export.columns if isinstance(c, str) and c.startswith(dia)]
        s = export[["avenida"] + columnas].copy()
        s.rename(columns={c: c.replace(f"{dia}_", "") for c in columnas}, inplace=True)
        salidas[dia] = s
    return salidas


def generar_curvas_desde_tabla_ancha(tabla_ancha) -> dict:
    """Convierte la tabla ancha de índices de movilidad en la matriz 7×24 de
    congestión histórica que usa tools/trafico.py como fallback de TomTom:
    congestion = 1 - promedio(indice_movilidad) por escenario, acotado [0,1].
    """
    curvas = {}
    for dia in DIAS_ES:
        fila = []
        for h in range(24):
            col = f"{dia}_{h:02d}h"
            if col in tabla_ancha.columns:
                valor = float(tabla_ancha[col].mean())
                fila.append(round(max(0.0, min(1.0, 1 - valor)), 3))
            else:
                fila.append(None)
        # celdas sin observación: se rellenan con el promedio del día
        con_dato = [v for v in fila if v is not None]
        respaldo = round(sum(con_dato) / len(con_dato), 3) if con_dato else 0.1
        curvas[dia] = [v if v is not None else respaldo for v in fila]
    return curvas


# ══════════════════════════════════════════════════════════════════════
# 3) Integración espacial (Fase 2 — Integracion_Espacial_Fase2.ipynb)
# ══════════════════════════════════════════════════════════════════════

def diagnostico_espacial(interseccion, hexagonos, campo_hex: str = "id",
                         campo_avenida: str = "Nombre", bitacora: list | None = None) -> dict:
    """Paso 10a: cuántos hexágonos son atravesados por alguna ruta, cuántos
    no, y cuántos por más de una. El criterio de agregación se define A
    PARTIR de este diagnóstico."""
    bitacora = bitacora if bitacora is not None else []
    total = hexagonos[campo_hex].nunique()
    con_ruta = interseccion[campo_hex].nunique()
    rutas_por_hex = interseccion.groupby(campo_hex)[campo_avenida].nunique()
    con_cruce = int((rutas_por_hex > 1).sum())
    bitacora.append(
        f"Hexágonos con ruta: {con_ruta}/{total}; con cruce: {con_cruce}. "
        "Decisión: promedio aritmético simple (los cruces son minoritarios); "
        "hexágonos sin ruta se conservan con movilidad NULL — no se interpola.")
    return {"total": total, "con_ruta": con_ruta,
            "sin_ruta": total - con_ruta, "con_cruce": con_cruce}


def validar_nombres(interseccion, movilidad, campo_interseccion: str = "Nombre",
                    campo_movilidad: str = "avenida") -> set:
    """Paso 10b: nombres sin coincidencia exacta quedarían como NaN y .mean()
    los ignoraría SIN avisar — por eso se valida explícitamente."""
    sin_match = set(interseccion[campo_interseccion].unique()) - set(movilidad[campo_movilidad].unique())
    assert not sin_match, f"Nombres sin coincidencia exacta: {sin_match} — revisa acentos/espacios."
    return sin_match


def agregar_movilidad_por_hexagono(interseccion, movilidad_dia,
                                   campo_hex: str = "id",
                                   campo_interseccion: str = "Nombre",
                                   campo_movilidad: str = "avenida"):
    """Paso 10c: promedio aritmético simple del índice de movilidad de las
    rutas que atraviesan cada hexágono, para las 24 horas de un día."""
    tabla = interseccion.merge(movilidad_dia, left_on=campo_interseccion,
                               right_on=campo_movilidad, how="left")
    return tabla.groupby(campo_hex)[COLUMNAS_HORAS].mean().reset_index()


def ancho_a_largo_temporal(hex_movilidad, fecha_base: str, campo_hex: str = "id"):
    """Paso 11: formato largo (una fila por hexágono-hora) con fecha_hora
    construida sobre una fecha ficticia — permite diferenciar los 7 días en
    el Controlador Temporal de QGIS."""
    import pandas as pd
    largo = hex_movilidad.melt(id_vars=campo_hex, value_vars=COLUMNAS_HORAS,
                               var_name="hora", value_name="movilidad")
    largo["fecha_hora"] = pd.to_datetime(fecha_base + " " + largo["hora"].str.replace("h", ":00"))
    return largo.sort_values(["fecha_hora", campo_hex]).reset_index(drop=True)


def generar_capa_temporal(hexagonos_gdf, movilidad_larga, campo_hex: str = "id"):
    """Paso 12: replica cada hexágono 24 veces (una por hora) y une el valor
    de movilidad por id + fecha_hora. Hexágonos sin ruta quedan NULL
    (visibles en gris en la animación de QGIS). Requiere geopandas."""
    import geopandas as gpd
    import pandas as pd

    piezas = []
    for fecha in sorted(movilidad_larga["fecha_hora"].unique()):
        temp = hexagonos_gdf.copy()
        temp["fecha_hora"] = fecha
        temp["hora"] = pd.to_datetime(fecha).strftime("%Hh")
        piezas.append(temp)
    base = pd.concat(piezas, ignore_index=True)
    resultado = base.merge(movilidad_larga[[campo_hex, "fecha_hora", "movilidad"]],
                           on=[campo_hex, "fecha_hora"], how="left")
    return gpd.GeoDataFrame(resultado, geometry="geometry", crs=hexagonos_gdf.crs)
