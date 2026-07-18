"""proyecto-urbano-agentivo — sistema multi-agente de análisis urbano.

Consolida cuatro proyectos independientes (detección de fachadas/techos con
YOLO11, mapeo satelital YOLOv8-seg, comparador de 5 modelos de ventanas y
análisis espacial de riesgo sísmico) en una sola arquitectura:

    src/tools/   — funciones puras (visión, tráfico, SIG)
    src/agents/  — agentes autónomos (Orquestador, SIG, Visión, Riesgo)
    src/dashboard/ — Streamlit multipágina unificado
"""
