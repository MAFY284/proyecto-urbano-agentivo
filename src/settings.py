"""Carga de configuración central (config/settings.yaml).

Todas las rutas relativas del YAML se resuelven contra la raíz del proyecto,
así los módulos funcionan igual sin importar desde dónde se invoquen (CLI,
Streamlit, tests).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

RAIZ_PROYECTO = Path(__file__).resolve().parents[1]
RUTA_SETTINGS = RAIZ_PROYECTO / "config" / "settings.yaml"


@lru_cache(maxsize=1)
def cargar() -> dict:
    with open(RUTA_SETTINGS, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ruta(relativa: str) -> Path:
    """Resuelve una ruta del settings.yaml contra la raíz del proyecto."""
    p = Path(relativa)
    return p if p.is_absolute() else RAIZ_PROYECTO / p
