"""Agente SIG — abstracción de todas las operaciones geoespaciales.

Es la única autoridad sobre la clave espacial unificada `cvegeo`: carga los
polígonos de manzanas, localiza puntos y fotografías, delimita el área de
una manzana y exporta capas GeoJSON listas para QGIS/uMap.
"""

from __future__ import annotations

import json

from src import settings
from src.tools import gis_utils


class AgenteSIG:
    def __init__(self, ruta_geojson=None):
        ruta = ruta_geojson or settings.ruta(
            settings.cargar()["geoespacial"]["manzanas_geojson"])
        self.geojson, self._shapes = gis_utils.cargar_manzanas(ruta)
        self._por_cvegeo = {m["cvegeo"]: m["shape"] for m in self._shapes}

    # ── Localización ──

    def localizar(self, lat: float, lon: float) -> str | None:
        """CVEGEO de la manzana que contiene el punto, o None."""
        return gis_utils.buscar_manzana(self._shapes, lat, lon)

    def localizar_archivo(self, nombre_archivo: str) -> dict:
        """Geolocaliza una fotografía por su nombre (patrón 'lat_lon.ext')."""
        lat, lon = gis_utils.extraer_coordenadas_de_nombre(nombre_archivo)
        cvegeo = self.localizar(lat, lon) if lat is not None else None
        return {"lat": lat, "lon": lon, "cvegeo": cvegeo}

    # ── Delimitación ──

    def listar_manzanas(self) -> list[str]:
        return sorted(self._por_cvegeo)

    def existe(self, cvegeo: str) -> bool:
        return cvegeo in self._por_cvegeo

    def delimitar(self, cvegeo: str) -> dict:
        """Delimita el área de análisis de una manzana: bbox, centroide y
        área del polígono (insumo del pipeline del Orquestador)."""
        shp = self._por_cvegeo.get(cvegeo)
        if shp is None:
            raise KeyError(f"CVEGEO '{cvegeo}' no existe en la capa de manzanas")
        minx, miny, maxx, maxy = shp.bounds
        c = shp.centroid
        return {
            "cvegeo": cvegeo,
            "bbox": {"lat_nw": maxy, "lon_nw": minx, "lat_se": miny, "lon_se": maxx},
            "centroide": {"lat": c.y, "lon": c.x},
            "area_grados2": shp.area,
        }

    # ── Exportación ──

    def feature(self, cvegeo: str) -> dict | None:
        for feat in self.geojson.get("features", []):
            if feat["properties"].get("CVEGEO") == cvegeo:
                return feat
        return None

    def exportar_geojson(self, cvegeos: list[str] | None = None,
                         propiedades_extra: dict | None = None) -> str:
        """FeatureCollection (str JSON) de las manzanas pedidas — o todas —
        opcionalmente enriquecida con propiedades por manzana (por ejemplo el
        score de riesgo), lista para arrastrar a QGIS."""
        propiedades_extra = propiedades_extra or {}
        features = []
        for feat in self.geojson.get("features", []):
            cv = feat["properties"].get("CVEGEO")
            if cvegeos is not None and cv not in cvegeos:
                continue
            f = json.loads(json.dumps(feat))   # copia profunda
            f["properties"].update(propiedades_extra.get(cv, {}))
            features.append(f)
        return json.dumps({"type": "FeatureCollection", "features": features},
                          ensure_ascii=False)
