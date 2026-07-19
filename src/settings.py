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


def verificar_pesos(p: Path) -> Path:
    """Valida un archivo de pesos antes de cargarlo, con errores accionables:
    distingue entre "no existe" y "es un puntero de Git LFS sin descargar"
    (clon hecho sin git-lfs instalado) — ambos se veían como un críptico
    'not found' del framework al intentar cargar el modelo."""
    if not p.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo de pesos '{p.name}' en {p.parent}. "
            "Verifica que clonaste el repositorio completo y corre `git lfs pull` "
            "en la raíz para descargar los modelos.")
    if p.stat().st_size < 2048:  # los .pt/.pth reales pesan MB; un puntero LFS ~130 bytes
        raise FileNotFoundError(
            f"'{p.name}' es un puntero de Git LFS, no el modelo real (el clon se hizo "
            "sin git-lfs). Instala Git LFS y corre `git lfs pull` en la raíz del repositorio.")
    return p
