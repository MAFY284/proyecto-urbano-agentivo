"""
Requiere una API key gratuita de TomTom:
  1. Crea una cuenta en https://developer.tomtom.com/
  2. Genera una API key del producto "Traffic API"
  3. Expórtala como variable de entorno (NO la escribas en este archivo):
       export TOMTOM_API_KEY="tu_key_aqui"

Uso:
    python3 trafico_tomtom.py              # recolecta una vez y termina
    python3 trafico_tomtom.py --loop       # recolecta cada 30 min indefinidamente
"""

import argparse
import json
import os
import sqlite3
import time
from datetime import datetime

import pandas as pd
import requests
from shapely.geometry import Point, shape

RAIZ = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(RAIZ, 'detecciones.db')
COORDS_PATH = os.path.join(RAIZ, 'coordenadas_trafico.xlsx')
GEOJSON_MANZANAS = os.path.join(RAIZ, 'hipodromo_manzanas.geojson')

TOMTOM_URL = 'https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json'

# Retención de datos: no tiene sentido acumular tráfico indefinidamente para un
# análisis de riesgo actual — se purgan registros más viejos que esto en cada corrida.
RETENCION_DIAS = 90  # 3 meses


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute('''CREATE TABLE IF NOT EXISTS trafico_calles (
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
    conn.execute('CREATE INDEX IF NOT EXISTS idx_trafico_cvegeo ON trafico_calles(cvegeo)')
    conn.execute('CREATE INDEX IF NOT EXISTS idx_trafico_fecha ON trafico_calles(fecha)')
    conn.commit()
    conn.close()


def purgar_antiguos(conn):
    """Borra lecturas de tráfico más viejas que RETENCION_DIAS. Se corre en cada
    recolección para que la tabla nunca crezca indefinidamente."""
    c = conn.cursor()
    c.execute(
        "DELETE FROM trafico_calles WHERE fecha < datetime('now', 'localtime', ?)",
        (f'-{RETENCION_DIAS} days',)
    )
    borrados = c.rowcount
    conn.commit()
    if borrados > 0:
        print(f"🗑️  Purgados {borrados} registros de tráfico con más de {RETENCION_DIAS} días")


def cargar_manzanas():
    if not os.path.exists(GEOJSON_MANZANAS):
        print(f"⚠️ No se encontró {GEOJSON_MANZANAS}; las calles se guardarán sin CVEGEO")
        return []
    with open(GEOJSON_MANZANAS, 'r') as f:
        geojson = json.load(f)
    manzanas = []
    for feat in geojson['features']:
        try:
            manzanas.append({'cvegeo': feat['properties'].get('CVEGEO', ''), 'shape': shape(feat['geometry'])})
        except Exception:
            pass
    return manzanas


def buscar_manzana(manzanas, lat, lon):
    pt = Point(lon, lat)
    for m in manzanas:
        if m['shape'].contains(pt):
            return m['cvegeo']
    return None


def obtener_flujo_tomtom(lat, lon, api_key):
    """Consulta TomTom Flow Segment Data para el punto (lat, lon).
    Retorna (velocidad_actual, velocidad_libre) en km/h, o (None, None) si falla."""
    params = {'point': f'{lat},{lon}', 'key': api_key}
    try:
        r = requests.get(TOMTOM_URL, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()['flowSegmentData']
        return data['currentSpeed'], data['freeFlowSpeed']
    except Exception as e:
        print(f"  ⚠️ Error consultando TomTom para ({lat},{lon}): {e}")
        return None, None


def recolectar_una_vez(api_key, vacuum=False):
    df = pd.read_excel(COORDS_PATH, header=1)
    manzanas = cargar_manzanas()
    init_db()

    conn = sqlite3.connect(DB_PATH)
    purgar_antiguos(conn)
    c = conn.cursor()

    for _, row in df.iterrows():
        vialidad = row['Vialidad']
        # Punto medio del segmento (aproximación razonable para calles cortas)
        lat = (row['ini_lat'] + row['dest_lat']) / 2
        lon = (row['ini_lon'] + row['dest_lon']) / 2

        vel_actual, vel_libre = obtener_flujo_tomtom(lat, lon, api_key)
        congestion = None
        if vel_actual is not None and vel_libre:
            congestion = round(max(0.0, min(1.0, 1 - vel_actual / vel_libre)), 3)

        cvegeo = buscar_manzana(manzanas, lat, lon)

        c.execute('''INSERT INTO trafico_calles (vialidad, lat, lon, cvegeo, velocidad_actual, velocidad_libre, congestion)
                     VALUES (?, ?, ?, ?, ?, ?, ?)''',
                  (vialidad, lat, lon, cvegeo, vel_actual, vel_libre, congestion))

        estado = f"{congestion*100:.0f}% congestión" if congestion is not None else "sin datos"
        print(f"  {vialidad}: {estado} (manzana: {cvegeo or 'no identificada'})")

        time.sleep(0.5)  # margen prudente frente al límite de tasa gratuito

    conn.commit()

    if vacuum:
        print("🧹 Compactando detecciones.db (VACUUM)...")
        conn.execute('VACUUM')

    conn.close()
    print(f"✅ Recolección completada — {len(df)} calles guardadas en {DB_PATH}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--loop', action='store_true', help='Recolecta cada 30 minutos indefinidamente')
    parser.add_argument('--vacuum', action='store_true', help='Compacta la base de datos tras recolectar (libera espacio de filas purgadas)')
    args = parser.parse_args()

    api_key = os.environ.get('TOMTOM_API_KEY')
    if not api_key:
        print("❌ Falta la variable de entorno TOMTOM_API_KEY.")
        print("   Consigue una key gratuita en https://developer.tomtom.com/ y expórtala:")
        print('   export TOMTOM_API_KEY="tu_key_aqui"')
        return

    if args.loop:
        while True:
            print(f"\n🚦 Recolectando tráfico — {datetime.now().isoformat()}")
            recolectar_una_vez(api_key, vacuum=args.vacuum)
            time.sleep(1800)
    else:
        print(f"🚦 Recolectando tráfico — {datetime.now().isoformat()}")
        recolectar_una_vez(api_key, vacuum=args.vacuum)


if __name__ == '__main__':
    main()
