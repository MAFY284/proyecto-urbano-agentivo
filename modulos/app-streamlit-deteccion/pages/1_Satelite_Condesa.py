import os
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import pandas as pd
import json
import folium
import gc
import torch
from streamlit_folium import st_folium

# Configuración de la página
st.set_page_config(page_title="IA Mapeo Urbano Analítico", layout="wide")

st.title("🏙️ IA Mapeo Analítico: La Condesa")
st.write("Sube una imagen satelital, define sus coordenadas geográficas y genera un reporte + capa GeoJSON lista para OSM/uMap.")

# El peso vive en el almacén único del proyecto (config/checkpoints/, raíz del repo)
_MODELO_DEFAULT = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "..", "config", "checkpoints", "satelite_yolov8xl_seg.pt"))

# --- BARRA LATERAL (SIDEBAR) PARA CONTROLES ---
st.sidebar.header("⚙️ Configuración del Modelo")

nombre_modelo = st.sidebar.text_input(
    "Archivo del modelo (.pt):",
    value=_MODELO_DEFAULT,
    help="Ruta al peso YOLO. Por defecto: config/checkpoints/satelite_yolov8xl_seg.pt"
)

# Cargar el modelo usando caché
@st.cache_resource(show_spinner="Cargando modelo YOLO en memoria...")
def load_model(ruta_modelo):
    return YOLO(ruta_modelo, task='segment')

try:
    model = load_model(nombre_modelo)
except Exception as e:
    st.error(f"❌ No se pudo cargar el modelo '{nombre_modelo}': {e}")
    st.stop()

confianza_slider = st.sidebar.slider(
    "Umbral de Confianza (Confidence)",
    min_value=0.1, max_value=1.0, value=0.5, step=0.05,
    help="Valores más altos muestran solo detecciones muy seguras."
)

st.sidebar.subheader("🧠 Uso de Memoria (GPU)")
usar_half = st.sidebar.checkbox("Precisión media (half/FP16)", value=True)
dispositivo = st.sidebar.selectbox("Dispositivo", options=["GPU (cuda)", "CPU"], index=0)
device_param = "0" if dispositivo == "GPU (cuda)" else "cpu"
half_param = usar_half if dispositivo == "GPU (cuda)" else False

st.sidebar.subheader("🧩 Procesamiento por Tiles")
tile_size = st.sidebar.number_input("Tamaño de cada tile (px):", min_value=320, max_value=1280, value=1280, step=32)
imgsz_input = tile_size 
overlap_px = st.sidebar.number_input("Traslape entre tiles (px):", min_value=0, max_value=256, value=64, step=16)

st.sidebar.subheader("📐 Escala de la Imagen")
escala_pixel = st.sidebar.number_input("Resolución (Metros por Píxel):", min_value=0.01, max_value=5.00, value=0.30, step=0.05)

st.sidebar.subheader("🌍 Georreferenciación")
lat_nw = st.sidebar.number_input("Latitud Superior-Izq", value=19.4130, format="%.6f")
lon_nw = st.sidebar.number_input("Longitud Superior-Izq", value=-99.1780, format="%.6f")
lat_se = st.sidebar.number_input("Latitud Inferior-Der", value=19.4080, format="%.6f")
lon_se = st.sidebar.number_input("Longitud Inferior-Der", value=-99.1720, format="%.6f")

# --- FUNCIONES GEO-ESPACIALES ---
def pixel_a_geo(px, py, w, h):
    lon = lon_nw + (px / w) * (lon_se - lon_nw)
    lat = lat_nw + (py / h) * (lat_se - lat_nw)
    return lon, lat

def generar_tiles(w, h, tile_size, overlap):
    paso = max(tile_size - overlap, 1)
    xs = list(range(0, max(w - tile_size, 0) + 1, paso))
    if not xs or xs[-1] + tile_size < w: xs.append(max(w - tile_size, 0))
    ys = list(range(0, max(h - tile_size, 0) + 1, paso))
    if not ys or ys[-1] + tile_size < h: ys.append(max(h - tile_size, 0))
    tiles = []
    for y0 in ys:
        for x0 in xs:
            x1, y1 = min(x0 + tile_size, w), min(y0 + tile_size, h)
            tiles.append((x0, y0, x1, y1))
    return tiles

def esta_en_zona_nucleo(cx, cy, x0, y0, x1, y1, overlap, w, h):
    m = overlap // 2
    nx0, nx1 = x0 + (m if x0 > 0 else 0), x1 - (m if x1 < w else 0)
    ny0, ny1 = y0 + (m if y0 > 0 else 0), y1 - (m if y1 < h else 0)
    return nx0 <= cx <= nx1 and ny0 <= cy <= ny1

# --- MEMORIA GLOBAL ---
if "resultado_satelite" not in st.session_state:
    st.session_state.resultado_satelite = None
if 'area_satelital' not in st.session_state:
    st.session_state['area_satelital'] = 0.0

# --- CUERPO PRINCIPAL ---
imagen_subida = st.file_uploader("Sube la imagen satelital:", type=["jpg", "jpeg", "png", "tif", "tiff"])

if imagen_subida is not None:
    # Mostramos botón solo si hay imagen subida
    boton_analizar = st.button("Analizar Entorno y Calcular Métricas 🚀")
    
    if boton_analizar:
        # MEJORA UX: Envolver todo en un spinner
        with st.spinner("🛰️ Ejecutando escaneo aéreo con YOLOv8. Procesando tiles espaciales..."):
            try:
                image = Image.open(imagen_subida).convert("RGB")
                w, h = image.size
                tiles = generar_tiles(w, h, tile_size, overlap_px)
                
                datos_reporte, geojson_features, polys_para_dibujar = [], [], []
                area_total_m2 = 0.0
                contador_id = 0

                barra_progreso = st.progress(0)

                for i, (x0, y0, x1, y1) in enumerate(tiles):
                    tile_img = image.crop((x0, y0, x1, y1))
                    results = model.predict(tile_img, conf=confianza_slider, imgsz=imgsz_input, half=half_param, device=device_param, retina_masks=False, verbose=False)

                    if results[0].masks is not None:
                        poligonos_px = results[0].masks.xy
                        confianzas = results[0].boxes.conf.cpu().numpy()
                        clases = results[0].boxes.cls.cpu().numpy() if results[0].boxes.cls is not None else None

                        for idx, (poly, conf) in enumerate(zip(poligonos_px, confianzas)):
                            if len(poly) < 3: continue
                            
                            poly_global = poly.copy()
                            poly_global[:, 0] += x0
                            poly_global[:, 1] += y0
                            cx, cy = poly_global[:, 0].mean(), poly_global[:, 1].mean()

                            if not esta_en_zona_nucleo(cx, cy, x0, y0, x1, y1, overlap_px, w, h): continue

                            xs_, ys_ = poly_global[:, 0], poly_global[:, 1]
                            area_px = 0.5 * abs(np.dot(xs_, np.roll(ys_, 1)) - np.dot(ys_, np.roll(xs_, 1)))
                            area_m2 = area_px * (escala_pixel ** 2)
                            area_total_m2 += area_m2

                            nombre_clase = model.names[int(clases[idx])] if clases is not None else "building"
                            contador_id += 1

                            # Centroide del edificio convertido a lat/lon
                            lon_centro, lat_centro = pixel_a_geo(cx, cy, w, h)

                            datos_reporte.append({
                                "Edificio ID": contador_id, "Clase": nombre_clase,
                                "Confianza (%)": round(float(conf) * 100, 2),
                                "Tamaño (Píxeles²)": int(area_px), "Área Estimada (m²)": round(area_m2, 2),
                                "Latitud": round(lat_centro, 6), "Longitud": round(lon_centro, 6)
                            })

                            coords_geo = [list(pixel_a_geo(x, y, w, h)) for x, y in poly_global]
                            coords_geo.append(coords_geo[0])
                            
                            geojson_features.append({
                                "type": "Feature", "properties": {"id": contador_id, "class": nombre_clase, "confidence": round(float(conf), 3), "area_m2": round(area_m2, 2), "lat": round(lat_centro, 6), "lon": round(lon_centro, 6)},
                                "geometry": {"type": "Polygon", "coordinates": [coords_geo]}
                            })
                            polys_para_dibujar.append(poly_global)

                    del results
                    gc.collect()
                    if torch.cuda.is_available(): torch.cuda.empty_cache()
                    barra_progreso.progress((i + 1) / len(tiles))

                # Guardamos para la Fusión
                st.session_state['area_satelital'] = area_total_m2
                st.session_state['coordenadas'] = f"{lat_nw:.4f}, {lon_nw:.4f}"

                from PIL import ImageDraw
                imagen_dibujada = image.copy()
                draw = ImageDraw.Draw(imagen_dibujada, "RGBA")
                for poly_global in polys_para_dibujar:
                    puntos = [tuple(p) for p in poly_global]
                    draw.polygon(puntos, outline=(0, 255, 100, 255), fill=(0, 255, 100, 60))

                # Guardar TODO en la memoria central
                st.session_state.resultado_satelite = {
                    "num_detecciones": len(datos_reporte),
                    "area_total_m2": area_total_m2,
                    "confianza_usada": confianza_slider,
                    "imagen_plotted": imagen_dibujada,
                    "datos_reporte": datos_reporte,
                    "geojson_data": {"type": "FeatureCollection", "features": geojson_features},
                    "centro_lat": (lat_nw + lat_se) / 2,
                    "centro_lon": (lon_nw + lon_se) / 2,
                }
                
                # MEJORA UX: Notificación de éxito
                st.toast("¡Mapeo Satelital Completado! 🗺️", icon="✅")

            except Exception as e:
                st.error(f"❌ Error inesperado: {e}")
                st.session_state.resultado_satelite = None
            finally:
                gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()

# --- BLOQUE DE VISUALIZACIÓN ---
if st.session_state.resultado_satelite is not None:
    r = st.session_state.resultado_satelite
    st.info("ℹ️ Mostrando análisis satelital activo (Los datos están listos para la pestaña de Fusión)")

    col1, col2, col3 = st.columns(3)
    col1.metric("🏢 Edificios Detectados", f"{r['num_detecciones']}")
    col2.metric("📐 Área Total Construida", f"{r['area_total_m2']:.2f} m²")
    col3.metric("🎯 Confianza Mínima", f"{r['confianza_usada']:.2f}")

    st.image(r["imagen_plotted"], caption='Polígonos de Edificios', use_container_width=True)

    if r["num_detecciones"] > 0:
        df = pd.DataFrame(r["datos_reporte"])
        st.dataframe(df, use_container_width=True)

        col_a, col_b = st.columns(2)
        with col_a:
            st.download_button("📥 Descargar Reporte (CSV)", data=df.to_csv(index=False).encode('utf-8'), file_name='reporte.csv', mime='text/csv')
        with col_b:
            st.download_button("🌍 Descargar GeoJSON", data=json.dumps(r["geojson_data"]).encode('utf-8'), file_name='deteccion.geojson', mime='application/geo+json')

        st.subheader("🗺️ Vista en Mapa (Folium)")
        mapa = folium.Map(location=[r["centro_lat"], r["centro_lon"]], zoom_start=17)
        for feature in r["geojson_data"]["features"]:
            folium.GeoJson(feature, tooltip=f"ID {feature['properties']['id']} · {feature['properties']['area_m2']} m² · ({feature['properties']['lat']}, {feature['properties']['lon']})").add_to(mapa)
        st_folium(mapa, width=900, height=500, key="mapa_folium")