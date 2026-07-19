"""Segmentación de edificios en imágenes satelitales (YOLOv8-XL seg).

Lógica heredada de App_Streamlit_Deteccion/pages/1_Satelite_Condesa.py,
convertida en funciones puras: tiling con traslape y zona núcleo (evita
duplicados en las costuras), área por fórmula del polígono (shoelace),
georreferenciación por esquinas NW/SE y exportación GeoJSON compatible
con QGIS/uMap/JOSM.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass, field

import numpy as np

from src import settings

_modelo_cache = None


@dataclass
class Esquinas:
    """Georreferenciación de la imagen por sus esquinas NW y SE."""
    lat_nw: float
    lon_nw: float
    lat_se: float
    lon_se: float

    @classmethod
    def por_defecto(cls) -> "Esquinas":
        s = settings.cargar()["satelite"]
        return cls(s["lat_nw"], s["lon_nw"], s["lat_se"], s["lon_se"])

    def pixel_a_geo(self, px: float, py: float, w: int, h: int) -> tuple[float, float]:
        """(px, py) en píxeles → (lon, lat) por interpolación lineal."""
        lon = self.lon_nw + (px / w) * (self.lon_se - self.lon_nw)
        lat = self.lat_nw + (py / h) * (self.lat_se - self.lat_nw)
        return lon, lat

    @property
    def centro(self) -> tuple[float, float]:
        return (self.lat_nw + self.lat_se) / 2, (self.lon_nw + self.lon_se) / 2


@dataclass
class ResultadoSatelite:
    num_detecciones: int
    area_total_m2: float
    detecciones: list           # dicts con id, clase, confianza, área, lat/lon
    geojson: dict               # FeatureCollection lista para QGIS/uMap
    poligonos_px: list          # polígonos en píxeles globales (para dibujar)
    centro: tuple               # (lat, lon) del centro del área
    imagen_anotada: object = None   # PIL.Image con polígonos dibujados
    extra: dict = field(default_factory=dict)


def cargar_modelo():
    global _modelo_cache
    if _modelo_cache is None:
        from ultralytics import YOLO
        # verificar_pesos da un error claro si faltan o son punteros LFS,
        # en vez del críptico 'not found' del framework
        ruta = settings.verificar_pesos(
            settings.ruta(settings.cargar()["satelite"]["pesos"]))
        _modelo_cache = YOLO(str(ruta), task='segment')
    return _modelo_cache


def generar_tiles(w: int, h: int, tile_size: int, overlap: int) -> list[tuple]:
    """Rejilla de tiles con traslape; garantiza cubrir los bordes derecho e
    inferior aunque no caigan en el paso exacto."""
    paso = max(tile_size - overlap, 1)
    xs = list(range(0, max(w - tile_size, 0) + 1, paso))
    if not xs or xs[-1] + tile_size < w:
        xs.append(max(w - tile_size, 0))
    ys = list(range(0, max(h - tile_size, 0) + 1, paso))
    if not ys or ys[-1] + tile_size < h:
        ys.append(max(h - tile_size, 0))
    return [(x0, y0, min(x0 + tile_size, w), min(y0 + tile_size, h))
            for y0 in ys for x0 in xs]


def esta_en_zona_nucleo(cx, cy, x0, y0, x1, y1, overlap, w, h) -> bool:
    """Un edificio solo se acepta si su centroide cae en la zona núcleo del
    tile (excluye media franja de traslape hacia tiles vecinos) — así cada
    edificio se cuenta exactamente una vez aunque aparezca en varios tiles."""
    m = overlap // 2
    nx0, nx1 = x0 + (m if x0 > 0 else 0), x1 - (m if x1 < w else 0)
    ny0, ny1 = y0 + (m if y0 > 0 else 0), y1 - (m if y1 < h else 0)
    return nx0 <= cx <= nx1 and ny0 <= cy <= ny1


def area_shoelace_px(poly: np.ndarray) -> float:
    xs, ys = poly[:, 0], poly[:, 1]
    return 0.5 * abs(np.dot(xs, np.roll(ys, 1)) - np.dot(ys, np.roll(xs, 1)))


def segmentar_imagen(imagen, esquinas: Esquinas | None = None,
                     escala_m_px: float | None = None, conf: float = 0.5,
                     tile_size: int | None = None, overlap: int | None = None,
                     device: str | None = None, half: bool = False,
                     progreso=None, dibujar: bool = True) -> ResultadoSatelite:
    """Pipeline completo: imagen satelital → tiles → YOLOv8-seg → polígonos
    globales → m² → GeoJSON georreferenciado.

    `progreso`: callback opcional f(fraccion_0_1) para barras de progreso.
    """
    import torch
    from PIL import Image, ImageDraw

    s = settings.cargar()["satelite"]
    esquinas = esquinas or Esquinas.por_defecto()
    escala = escala_m_px if escala_m_px is not None else s["escala_m_px"]
    tile_size = tile_size or s["tile_size"]
    overlap = overlap if overlap is not None else s["overlap_px"]

    modelo = cargar_modelo()
    if not isinstance(imagen, Image.Image):
        imagen = Image.open(imagen)
    imagen = imagen.convert("RGB")
    w, h = imagen.size
    tiles = generar_tiles(w, h, tile_size, overlap)

    detecciones, features, polys_global = [], [], []
    area_total_m2 = 0.0
    contador = 0

    device_efectivo = device or ("0" if torch.cuda.is_available() else "cpu")

    for i, (x0, y0, x1, y1) in enumerate(tiles):
        tile_img = imagen.crop((x0, y0, x1, y1))
        try:
            results = modelo.predict(tile_img, conf=conf, imgsz=tile_size, half=half,
                                     device=device_efectivo,
                                     retina_masks=False, verbose=False)
        except RuntimeError as e:
            from src.tools.fachada import es_error_de_memoria
            if device_efectivo == "cpu" or not es_error_de_memoria(e):
                raise
            # GPU sin memoria para este tamaño de tile: seguir todo el lote en
            # CPU — más lento, pero el análisis termina en cualquier equipo
            torch.cuda.empty_cache()
            device_efectivo = "cpu"
            results = modelo.predict(tile_img, conf=conf, imgsz=tile_size, half=half,
                                     device=device_efectivo,
                                     retina_masks=False, verbose=False)

        if results[0].masks is not None:
            poligonos_px = results[0].masks.xy
            confianzas = results[0].boxes.conf.cpu().numpy()
            clases = results[0].boxes.cls.cpu().numpy() if results[0].boxes.cls is not None else None

            for idx, (poly, cf) in enumerate(zip(poligonos_px, confianzas)):
                if len(poly) < 3:
                    continue
                poly_global = poly.copy()
                poly_global[:, 0] += x0
                poly_global[:, 1] += y0
                cx, cy = poly_global[:, 0].mean(), poly_global[:, 1].mean()

                if not esta_en_zona_nucleo(cx, cy, x0, y0, x1, y1, overlap, w, h):
                    continue

                area_px = area_shoelace_px(poly_global)
                area_m2 = area_px * (escala ** 2)
                area_total_m2 += area_m2

                clase = modelo.names[int(clases[idx])] if clases is not None else "building"
                contador += 1
                lon_c, lat_c = esquinas.pixel_a_geo(cx, cy, w, h)

                detecciones.append({
                    "id": contador, "clase": clase,
                    "confianza": round(float(cf), 3),
                    "area_px": int(area_px), "area_m2": round(area_m2, 2),
                    "lat": round(lat_c, 6), "lon": round(lon_c, 6),
                })

                coords_geo = [list(esquinas.pixel_a_geo(x, y, w, h)) for x, y in poly_global]
                coords_geo.append(coords_geo[0])
                features.append({
                    "type": "Feature",
                    "properties": {"id": contador, "class": clase,
                                   "confidence": round(float(cf), 3),
                                   "area_m2": round(area_m2, 2),
                                   "lat": round(lat_c, 6), "lon": round(lon_c, 6)},
                    "geometry": {"type": "Polygon", "coordinates": [coords_geo]},
                })
                polys_global.append(poly_global)

        del results
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if progreso:
            progreso((i + 1) / len(tiles))

    imagen_anotada = None
    if dibujar:
        imagen_anotada = imagen.copy()
        draw = ImageDraw.Draw(imagen_anotada, "RGBA")
        for poly in polys_global:
            draw.polygon([tuple(p) for p in poly],
                         outline=(0, 255, 100, 255), fill=(0, 255, 100, 60))

    return ResultadoSatelite(
        num_detecciones=len(detecciones),
        area_total_m2=area_total_m2,
        detecciones=detecciones,
        geojson={"type": "FeatureCollection", "features": features},
        poligonos_px=polys_global,
        centro=esquinas.centro,
        imagen_anotada=imagen_anotada,
    )
