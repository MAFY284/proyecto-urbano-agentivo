"""Herramientas puras del sistema — lógica heredada de los 4 repositorios,
convertida en funciones de Python invocables por los agentes:

    satelite.py   — tiling satelital + segmentación YOLOv8-XL + GeoJSON
    fachada.py    — pool de 7 modelos YOLO11 (fachadas, techos, daños, …)
    comparador.py — 5 motores de ventanas, incl. SAM3 vía subproceso aislado
    trafico.py    — TomTom asíncrono + fallback de curvas históricas
    gis_utils.py  — limpieza espacial (notebooks del Proyecto-Delfin)
"""
