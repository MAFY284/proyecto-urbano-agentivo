"""Capa de agentes autónomos del sistema.

    AgenteSIG     — abstracción de operaciones geoespaciales (manzanas CVEGEO)
    AgenteVision  — control de inferencias (satélite + fachadas + oráculo SAM3/D2)
    AgenteRiesgo  — TomTom/perfil histórico + score de Riesgo Urbano Combinado
    Orquestador   — supervisor reactivo que encadena el flujo completo
"""

from src.agents.agente_sig import AgenteSIG
from src.agents.agente_vision import AgenteVision
from src.agents.agente_riesgo import AgenteRiesgo
from src.agents.orquestador import Orquestador

__all__ = ["AgenteSIG", "AgenteVision", "AgenteRiesgo", "Orquestador"]
