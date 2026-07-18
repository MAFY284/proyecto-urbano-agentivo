"""
Descarga una imagen satelital de la colonia Hipódromo (CDMX) desde el Copernicus
Data Space Ecosystem (CDSE) — la plataforma gratuita de ESA que desde 2023-2024
reemplazó al Sentinel Hub clásico para acceso gratuito a Sentinel-2, con una API
compatible con la Process API de Sentinel Hub (mismo formato de request/evalscript,
solo cambian las URLs de autenticación y de la API). Recorta la imagen al bounding
box de las manzanas en hipodromo_manzanas.geojson — sirve para tener una toma real
y actual de la zona de estudio (a diferencia de los datasets satelitales genéricos
de otras ciudades que ya usa el modelo de Techos, ver organizar_datasets.py).

Requiere una cuenta en https://dataspace.copernicus.eu/ y un "OAuth client" creado
en el dashboard (ícono de usuario > Settings > OAuth clients, o directo en
https://shapps.dataspace.copernicus.eu/dashboard/#/account/settings), con su
Client ID y Client Secret. NO los escribas en este archivo — expórtalos como
variables de entorno:
    export SENTINEL_CLIENT_ID="tu_client_id"
    export SENTINEL_CLIENT_SECRET="tu_client_secret"

Uso:
    python3 descargar_sentinel.py                        # color verdadero, últimos 60 días, la menos nubosa
    python3 descargar_sentinel.py --dias 90               # busca en una ventana más amplia
    python3 descargar_sentinel.py --fecha 2026-03-15      # imagen de un día específico (±3 días alrededor)
    python3 descargar_sentinel.py --resolucion 10         # metros/píxel de salida (10 = nativa de Sentinel-2 RGB)
    python3 descargar_sentinel.py --salida mi_imagen.jpg  # ruta de salida explícita

Nota sobre resolución: Sentinel-2 tiene 10 m/píxel en las bandas RGB — un edificio
chico puede ocupar menos de 1 píxel. Sirve para tener una vista actual/de referencia
de la zona, pero no esperes el nivel de detalle de los datasets de techos que ya usa
el proyecto (esos vienen de fuentes de mayor resolución). Pedir --resolucion menor a
10 solo interpola (no agrega detalle real). Si necesitas más detalle de verdad,
Sentinel Hub también da acceso a otras colecciones (Planet, Airbus, etc.) pero son
de pago.
"""

import argparse
import json
import math
import os
import sys
from datetime import datetime, timedelta

import requests

RAIZ = os.path.dirname(os.path.abspath(__file__))
GEOJSON_MANZANAS = os.path.join(RAIZ, 'hipodromo_manzanas.geojson')
SALIDA_DIR_DEFAULT = os.path.join(RAIZ, 'datasets_fuente', 'techos', 'Sentinel_Hipodromo')

TOKEN_URL = 'https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token'
PROCESS_URL = 'https://sh.dataspace.copernicus.eu/api/v1/process'
LIMITE_PX_PROCESS_API = 2500  # máximo de ancho/alto por request que acepta Sentinel Hub

# Color verdadero estándar de Sentinel-2 L2A (B04=rojo, B03=verde, B02=azul), con
# ganancia x2.5 para que no se vea oscuro (la reflectancia típica de superficie es baja).
EVALSCRIPT_TRUE_COLOR = """
//VERSION=3
function setup() {
  return {
    input: ["B02", "B03", "B04"],
    output: { bands: 3, sampleType: "AUTO" }
  };
}
function evaluatePixel(sample) {
  return [sample.B04 * 2.5, sample.B03 * 2.5, sample.B02 * 2.5];
}
"""


def obtener_credenciales():
    client_id = os.environ.get('SENTINEL_CLIENT_ID')
    client_secret = os.environ.get('SENTINEL_CLIENT_SECRET')
    if not client_id or not client_secret:
        print("Faltan credenciales: exporta SENTINEL_CLIENT_ID y SENTINEL_CLIENT_SECRET.")
        print("Se generan en https://shapps.dataspace.copernicus.eu/dashboard/#/account/settings (OAuth clients).")
        sys.exit(1)
    return client_id, client_secret


def obtener_token(client_id, client_secret):
    resp = requests.post(TOKEN_URL, data={
        'grant_type': 'client_credentials',
        'client_id': client_id,
        'client_secret': client_secret,
    }, timeout=30)
    resp.raise_for_status()
    return resp.json()['access_token']


def extraer_coords(geometria):
    """Aplana las coordenadas de una geometría GeoJSON (Polygon o MultiPolygon, con
    o sin huecos, cualquier nivel de anidamiento) a una lista plana de (lon, lat)."""
    def recorrer(nodo):
        if isinstance(nodo[0], (int, float)):
            yield nodo
        else:
            for hijo in nodo:
                yield from recorrer(hijo)
    return list(recorrer(geometria['coordinates']))


def calcular_bbox(geojson_path, margen_m=100):
    """[min_lon, min_lat, max_lon, max_lat] que cubre todas las manzanas del geojson,
    con un margen extra en metros (convertido a grados según la latitud del sitio)."""
    with open(geojson_path, encoding='utf-8') as f:
        data = json.load(f)

    lons, lats = [], []
    for feat in data['features']:
        for lon, lat in extraer_coords(feat['geometry']):
            lons.append(lon)
            lats.append(lat)
    if not lons:
        raise ValueError(f"No se encontraron coordenadas en {geojson_path}")

    lat_centro = (min(lats) + max(lats)) / 2
    m_por_grado_lon = 111_320 * abs(math.cos(math.radians(lat_centro)))
    m_por_grado_lat = 111_320
    margen_lon = margen_m / m_por_grado_lon
    margen_lat = margen_m / m_por_grado_lat

    return [min(lons) - margen_lon, min(lats) - margen_lat, max(lons) + margen_lon, max(lats) + margen_lat]


def bbox_a_dimensiones(bbox, resolucion_m):
    """Ancho/alto en píxeles para cubrir el bbox (WGS84, grados) a la resolución
    pedida (metros/píxel) — recortado al límite de 2500px de la Process API."""
    min_lon, min_lat, max_lon, max_lat = bbox
    lat_centro = (min_lat + max_lat) / 2
    ancho_m = (max_lon - min_lon) * 111_320 * abs(math.cos(math.radians(lat_centro)))
    alto_m = (max_lat - min_lat) * 111_320
    ancho_px = max(1, round(ancho_m / resolucion_m))
    alto_px = max(1, round(alto_m / resolucion_m))

    if ancho_px > LIMITE_PX_PROCESS_API or alto_px > LIMITE_PX_PROCESS_API:
        print(f"Advertencia: {ancho_px}x{alto_px}px excede el límite de {LIMITE_PX_PROCESS_API}px "
              "por request de la Process API — se recorta al límite (baja la resolución con --resolucion "
              "si quieres cubrir toda el área sin recortar).")
        ancho_px = min(ancho_px, LIMITE_PX_PROCESS_API)
        alto_px = min(alto_px, LIMITE_PX_PROCESS_API)
    return ancho_px, alto_px


def descargar_imagen(token, bbox, ancho_px, alto_px, fecha_desde, fecha_hasta, salida_path):
    body = {
        "input": {
            "bounds": {
                "bbox": bbox,
                "properties": {"crs": "http://www.opengis.net/def/crs/OGC/1.3/CRS84"},
            },
            "data": [{
                "type": "sentinel-2-l2a",
                "dataFilter": {
                    "timeRange": {
                        "from": f"{fecha_desde}T00:00:00Z",
                        "to": f"{fecha_hasta}T23:59:59Z",
                    },
                    "mosaickingOrder": "leastCC",  # prioriza la escena con menos nubes en el rango
                    "maxCloudCoverage": 30,
                },
            }],
        },
        "output": {
            "width": ancho_px,
            "height": alto_px,
            "responses": [{"identifier": "default", "format": {"type": "image/jpeg"}}],
        },
        "evalscript": EVALSCRIPT_TRUE_COLOR,
    }

    resp = requests.post(
        PROCESS_URL,
        headers={"Authorization": f"Bearer {token}"},
        json=body,
        timeout=60,
    )
    if resp.status_code != 200:
        print(f"Error {resp.status_code} de Sentinel Hub: {resp.text[:800]}")
        resp.raise_for_status()

    os.makedirs(os.path.dirname(salida_path) or '.', exist_ok=True)
    with open(salida_path, 'wb') as f:
        f.write(resp.content)
    return salida_path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dias', type=int, default=60,
                         help="ventana de búsqueda hacia atrás en días desde hoy (default 60)")
    parser.add_argument('--fecha', type=str, default=None,
                         help="fecha específica YYYY-MM-DD (busca ±3 días alrededor en vez de --dias)")
    parser.add_argument('--resolucion', type=float, default=10.0,
                         help="metros/píxel de salida (default 10, la nativa de Sentinel-2 RGB)")
    parser.add_argument('--margen', type=float, default=100.0,
                         help="margen extra alrededor del bbox de las manzanas, en metros (default 100)")
    parser.add_argument('--salida', type=str, default=None,
                         help="ruta de archivo de salida (default: datasets_fuente/techos/Sentinel_Hipodromo/hipodromo_<fecha>.jpg)")
    args = parser.parse_args()

    client_id, client_secret = obtener_credenciales()
    print("Autenticando con Sentinel Hub...")
    token = obtener_token(client_id, client_secret)

    print(f"Calculando bounding box desde {GEOJSON_MANZANAS}...")
    bbox = calcular_bbox(GEOJSON_MANZANAS, margen_m=args.margen)
    print(f"  bbox (lon/lat): {[round(v, 6) for v in bbox]}")

    ancho_px, alto_px = bbox_a_dimensiones(bbox, args.resolucion)
    print(f"  tamaño de salida: {ancho_px}x{alto_px} px (~{args.resolucion} m/píxel)")

    if args.fecha:
        centro = datetime.strptime(args.fecha, '%Y-%m-%d')
        desde = (centro - timedelta(days=3)).strftime('%Y-%m-%d')
        hasta = (centro + timedelta(days=3)).strftime('%Y-%m-%d')
    else:
        hoy = datetime.utcnow()
        hasta = hoy.strftime('%Y-%m-%d')
        desde = (hoy - timedelta(days=args.dias)).strftime('%Y-%m-%d')
    print(f"  buscando la escena menos nubosa entre {desde} y {hasta}...")

    salida_path = args.salida or os.path.join(SALIDA_DIR_DEFAULT, f"hipodromo_{hasta}.jpg")

    print("Descargando...")
    ruta = descargar_imagen(token, bbox, ancho_px, alto_px, desde, hasta, salida_path)
    print(f"Listo: {ruta}")


if __name__ == '__main__':
    main()