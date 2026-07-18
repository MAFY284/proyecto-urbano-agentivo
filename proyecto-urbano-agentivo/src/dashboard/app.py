"""Dashboard unificado de proyecto-urbano-agentivo (Streamlit multipágina).

Unifica y sustituye las apps individuales heredadas:
  1. Centro de Mando Urbano        — mapa de riesgo por manzana (cvegeo),
     KPIs de población expuesta (35 m²/hab) y descarga GeoJSON para QGIS.
  2. Laboratorio de Benchmarking   — comparador lado a lado de los 5 modelos
     de ventanas (YOLOv8/-seg, YOLOv11-seg, Detectron2, SAM3).
  3. Simulador Dinámico de Crisis  — proyección Plotly del score de riesgo
     al alterar hora del día y densidad vial.

Ejecutar:  streamlit run src/dashboard/app.py
"""

from __future__ import annotations

import base64
import datetime
import json
import sys
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

# La app corre como script suelto: asegurar la raíz del proyecto en sys.path
RAIZ = Path(__file__).resolve().parents[2]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from src import db, settings                       # noqa: E402
from src.agents.agente_riesgo import AgenteRiesgo  # noqa: E402
from src.agents.agente_sig import AgenteSIG        # noqa: E402
from src.tools import comparador, trafico          # noqa: E402


# ── Recursos compartidos (una sola instancia por sesión de servidor) ──

@st.cache_resource(show_spinner="Cargando capa de manzanas...")
def sig() -> AgenteSIG:
    return AgenteSIG()


@st.cache_resource(show_spinner="Inicializando Agente de Riesgo...")
def riesgo() -> AgenteRiesgo:
    return AgenteRiesgo()


def color_por_score(score: float) -> str:
    if score >= 0.5:
        return "#d62728"   # alto
    if score >= 0.25:
        return "#ff9f1c"   # medio
    return "#2ca02c"       # bajo


def descargar_pdf_con_selector(pdf_bytes: bytes, nombre_sugerido: str, label: str) -> None:
    """Botón de descarga que abre el diálogo nativo "Guardar como…" del sistema
    operativo (File System Access API) en Chrome/Edge, para elegir carpeta y nombre
    en vez de descargar siempre a la carpeta de descargas por defecto del navegador.
    En Firefox/Safari, donde esa API no existe, cae a la descarga normal del navegador.
    """
    pdf_b64 = base64.b64encode(pdf_bytes).decode()
    html = f"""
    <button id="btnDescargarPdf" style="
        background-color:#0f766e;color:white;border:none;border-radius:6px;
        padding:0.5rem 1rem;font-size:1rem;cursor:pointer;">{label}</button>
    <script>
    document.getElementById('btnDescargarPdf').onclick = async () => {{
        const b64 = "{pdf_b64}";
        const bytes = Uint8Array.from(atob(b64), c => c.charCodeAt(0));
        const blob = new Blob([bytes], {{ type: 'application/pdf' }});
        const nombreSugerido = "{nombre_sugerido}";
        try {{
            if (window.showSaveFilePicker) {{
                const handle = await window.showSaveFilePicker({{
                    suggestedName: nombreSugerido,
                    types: [{{ description: 'Documento PDF', accept: {{ 'application/pdf': ['.pdf'] }} }}],
                }});
                const writable = await handle.createWritable();
                await writable.write(blob);
                await writable.close();
            }} else {{
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url; a.download = nombreSugerido;
                document.body.appendChild(a); a.click(); a.remove();
                URL.revokeObjectURL(url);
            }}
        }} catch (err) {{
            if (err.name !== 'AbortError') alert('No se pudo guardar el PDF: ' + err.message);
        }}
    }};
    </script>
    """
    components.html(html, height=50)


# ══════════════════════════════════════════════════════════════════════
# Página 1 — Centro de Mando Urbano
# ══════════════════════════════════════════════════════════════════════

def pagina_centro_mando():
    import folium
    from streamlit_folium import st_folium

    st.title("🏙️ Centro de Mando Urbano")
    st.caption("Riesgo Urbano Combinado por manzana (clave cvegeo) — "
               "daños 40% · congestión 30% · altura 30%")

    agente_sig = sig()
    riesgos = {r["cvegeo"]: r for r in db.listar_riesgos()}

    # ── Selección de manzana ──
    opciones = agente_sig.listar_manzanas()
    etiqueta = lambda cv: (f"{cv} — score {riesgos[cv]['score_riesgo']:.3f}"
                           if cv in riesgos else f"{cv} — sin evaluar")
    cvegeo = st.selectbox("Manzana (CVEGEO):", opciones, format_func=etiqueta)

    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        if st.button("♻️ Recalcular riesgo de esta manzana"):
            with st.spinner("Agente de Riesgo trabajando (TomTom → histórico)..."):
                delim = agente_sig.delimitar(cvegeo)
                riesgos[cvegeo] = riesgo().evaluar_manzana(
                    cvegeo, centroide=delim["centroide"])
            st.toast("Score actualizado", icon="✅")
    with col_info:
        if cvegeo in riesgos and riesgos[cvegeo].get("fuente_congestion"):
            st.caption(f"Fuente de congestión: `{riesgos[cvegeo]['fuente_congestion']}` · "
                       f"última evaluación: {riesgos[cvegeo].get('fecha', '—')}")

    # ── KPIs ──
    r = riesgos.get(cvegeo)
    c1, c2, c3, c4 = st.columns(4)
    if r:
        c1.metric("Score de riesgo", f"{r['score_riesgo']:.3f}")
        c2.metric("Congestión", f"{(r['congestion'] or 0) * 100:.0f}%")
        c3.metric("Altura promedio", f"{r['altura_promedio_pisos'] or 0:.1f} pisos")
        m2hab = settings.cargar()["poblacion"]["m2_por_habitante"]
        c4.metric("Población expuesta", f"{r['poblacion_estimada'] or '—'} hab",
                  help=f"Área satelital × pisos / {m2hab} m² por habitante")
        st.progress(min(1.0, r["score_riesgo"]),
                    text=f"Confianza de la muestra: {r['confianza']:.0%} ({r['num_fotos']} fotos)")
    else:
        c1.metric("Score de riesgo", "—")
        st.info("Esta manzana aún no tiene evaluación. Usa «Recalcular» o corre "
                "`python main.py analizar --cvegeo " + cvegeo + "`.")

    # ── Mapa Folium ──
    st.subheader("🗺️ Mapa de riesgo")
    delim = agente_sig.delimitar(cvegeo)
    centro = delim["centroide"]
    mapa = folium.Map(location=[centro["lat"], centro["lon"]], zoom_start=16)

    for feat in agente_sig.geojson.get("features", []):
        cv = feat["properties"].get("CVEGEO")
        info = riesgos.get(cv)
        color = color_por_score(info["score_riesgo"]) if info else "#9e9e9e"
        tooltip = (f"{cv} · score {info['score_riesgo']:.3f}" if info
                   else f"{cv} · sin datos")
        folium.GeoJson(
            feat,
            style_function=lambda _f, c=color, sel=(cv == cvegeo): {
                "fillColor": c, "color": "#1f77b4" if sel else c,
                "weight": 3 if sel else 1, "fillOpacity": 0.55},
            tooltip=tooltip,
        ).add_to(mapa)
    st_folium(mapa, width=900, height=520, key="mapa_riesgo")

    # ── Tabla de ranking ──
    if riesgos:
        import pandas as pd
        st.subheader("📋 Ranking de manzanas por riesgo")
        df = pd.DataFrame(riesgos.values())[
            ["cvegeo", "score_riesgo", "danos_norm", "congestion", "pisos_norm",
             "confianza", "num_fotos", "poblacion_estimada", "fuente_congestion"]]
        st.dataframe(df.sort_values("score_riesgo", ascending=False),
                     use_container_width=True, hide_index=True)

    # ── Descargas GeoJSON para QGIS ──
    st.subheader("💾 Capas GeoJSON listas para QGIS")
    extra = {cv: {"score_riesgo": i["score_riesgo"], "congestion": i["congestion"],
                  "poblacion_estimada": i["poblacion_estimada"]}
             for cv, i in riesgos.items()}
    col_a, col_b = st.columns(2)
    with col_a:
        st.download_button("🌍 Todas las manzanas + scores",
                           data=agente_sig.exportar_geojson(propiedades_extra=extra),
                           file_name="riesgo_manzanas.geojson",
                           mime="application/geo+json")
    with col_b:
        st.download_button(f"📍 Solo {cvegeo}",
                           data=agente_sig.exportar_geojson([cvegeo], propiedades_extra=extra),
                           file_name=f"manzana_{cvegeo}.geojson",
                           mime="application/geo+json")

    # ── Reporte PDF ejecutivo (lógica de fusión heredada) ──
    if r:
        from fpdf import FPDF
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 10, "REPORTE TÉCNICO DE INTELIGENCIA URBANA", ln=True, align="C")
        pdf.set_font("Helvetica", "I", 10)
        pdf.cell(0, 8, f"Fecha: {datetime.date.today():%d/%m/%Y}  ·  Manzana: {cvegeo}",
                 ln=True, align="C")
        pdf.ln(6)
        pdf.set_font("Helvetica", "", 11)
        for linea in [
            f"- Score de Riesgo Urbano Combinado: {r['score_riesgo']:.3f}",
            f"- Daños ponderados por severidad: {r['danos_ponderados']}",
            f"- Congestión vial ({r['fuente_congestion']}): {(r['congestion'] or 0):.0%}",
            f"- Altura promedio: {r['altura_promedio_pisos']} pisos",
            f"- Área satelital: {r['area_satelital_m2'] or 'sin dato'} m2",
            f"- Población expuesta estimada (35 m2/hab): {r['poblacion_estimada'] or 'sin dato'}",
            f"- Confianza de la muestra: {r['confianza']:.0%} ({r['num_fotos']} fotos)",
        ]:
            pdf.cell(0, 8, linea, ln=True)
        pdf.ln(8)
        pdf.set_font("Helvetica", "I", 9)
        pdf.multi_cell(0, 5, "Nota: entregable automatizado generado por el sistema "
                             "multi-agente. Los datos son estimaciones proyectuales "
                             "basadas en modelos de Visión por Computadora y densidades promedio.")
        descargar_pdf_con_selector(
            pdf_bytes=bytes(pdf.output()),
            nombre_sugerido=f"reporte_{cvegeo}_{datetime.date.today()}.pdf",
            label="📥 Descargar Reporte Ejecutivo (PDF)",
        )


# ══════════════════════════════════════════════════════════════════════
# Página 2 — Laboratorio de Benchmarking de Modelos
# ══════════════════════════════════════════════════════════════════════

def pagina_laboratorio():
    st.title("🧪 Laboratorio de Benchmarking de Modelos")
    st.caption("Evalúa lado a lado los 5 motores de detección de ventanas "
               "sobre imágenes de fachada cargadas por el usuario.")

    disponibles = comparador.motores_disponibles()
    if not disponibles:
        st.error("No hay motores disponibles: verifica los pesos en config/checkpoints/comparador/.")
        return

    with st.sidebar:
        st.header("Configuración")
        motores = st.multiselect("Motores a comparar", list(disponibles),
                                 default=list(disponibles)[:1])
        conf = st.slider("Umbral de confianza", 0.0, 1.0, 0.30, 0.05)
        prompt_sam3 = "window"
        if comparador.MOTOR_SAM3 in motores:
            prompt_sam3 = st.text_input("Prompt de texto (SAM3)", value="window")
            st.caption("ℹ️ SAM3 corre como subproceso en su venv aislado (env_sam3).")
        no_disponibles = {comparador.MOTOR_DETECTRON2: comparador.detectron2_disponible(),
                          comparador.MOTOR_SAM3: comparador.sam3_disponible()}
        for motor, ok in no_disponibles.items():
            if not ok:
                st.caption(f"ℹ️ {motor} no está disponible en este entorno (se oculta).")

    archivos = st.file_uploader("Sube una o varias imágenes de fachada",
                                type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    if not archivos:
        st.info("Sube imágenes para comenzar el benchmarking.")
        return
    if not motores:
        st.warning("Elige al menos un motor en la barra lateral.")
        return

    from PIL import Image
    for archivo in archivos:
        st.subheader(archivo.name)
        imagen = Image.open(archivo).convert("RGB")
        cols = st.columns(len(motores) + 1)
        cols[0].image(imagen, caption="Original", use_container_width=True)
        for i, motor in enumerate(motores, start=1):
            with st.spinner(f"{motor}..."):
                r = comparador.inferir(motor, imagen, conf=conf, prompt_sam3=prompt_sam3)
            if r.error:
                cols[i].error(f"{motor}: {r.error}")
            else:
                cols[i].image(r.imagen_anotada,
                              caption=f"{motor} — {r.n_detecciones} ventanas "
                                      f"(conf. prom. {r.confianza_promedio:.0%})",
                              use_container_width=True)
    st.toast("Benchmarking completado", icon="✅")


# ══════════════════════════════════════════════════════════════════════
# Página 3 — Simulador Dinámico de Crisis Sísmica
# ══════════════════════════════════════════════════════════════════════

def pagina_simulador():
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go

    st.title("🌋 Simulador Dinámico de Crisis Sísmica")
    st.caption("Proyecta el impacto en el score de riesgo al alterar "
               "artificialmente la hora del día y la densidad vial "
               "(perfil histórico de congestión de La Condesa).")

    agente = riesgo()
    riesgos = {r["cvegeo"]: r for r in db.listar_riesgos()}
    if not riesgos:
        st.warning("No hay manzanas evaluadas todavía — corre primero un análisis "
                   "en el Centro de Mando o con `python main.py analizar`.")
        return

    cvegeo = st.selectbox("Manzana a simular:", sorted(riesgos))
    base = riesgos[cvegeo]

    col1, col2, col3 = st.columns(3)
    with col1:
        dia = st.selectbox("Día de la semana", trafico.DIAS_ES,
                           index=datetime.date.today().weekday())
    with col2:
        hora = st.slider("Hora del día", 0, 23, datetime.datetime.now().hour)
    with col3:
        factor_densidad = st.slider("Factor de densidad vial", 0.0, 2.0, 1.0, 0.1,
                                    help="1.0 = congestión histórica normal; "
                                         "2.0 = colapso vial; 0.0 = calles vacías")

    # ── Score simulado en el escenario elegido ──
    congestion_sim = min(1.0, trafico.congestion_historica(dia, hora) * factor_densidad)
    sim = agente.calcular_score(base["danos_ponderados"], congestion_sim,
                                base["altura_promedio_pisos"], base["num_fotos"])

    c1, c2, c3 = st.columns(3)
    c1.metric("Score actual (BD)", f"{base['score_riesgo']:.3f}")
    c2.metric("Score simulado", f"{sim['score_riesgo']:.3f}",
              delta=f"{sim['score_riesgo'] - base['score_riesgo']:+.3f}",
              delta_color="inverse")
    c3.metric("Congestión simulada", f"{congestion_sim:.0%}")

    # ── Curva del día completo (24 h) ──
    horas = list(range(24))
    scores_dia = [agente.calcular_score(
        base["danos_ponderados"],
        min(1.0, trafico.congestion_historica(dia, h) * factor_densidad),
        base["altura_promedio_pisos"], base["num_fotos"])["score_riesgo"]
        for h in horas]
    congestiones = [min(1.0, trafico.congestion_historica(dia, h) * factor_densidad)
                    for h in horas]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=horas, y=scores_dia, mode="lines+markers",
                             name="Score de riesgo", line=dict(width=3)))
    fig.add_trace(go.Scatter(x=horas, y=congestiones, mode="lines",
                             name="Congestión (perfil histórico)",
                             line=dict(dash="dot")))
    fig.add_vline(x=hora, line_dash="dash", line_color="red",
                  annotation_text=f"{hora:02d}h")
    fig.update_layout(title=f"Evolución del riesgo — {cvegeo} · {dia} "
                            f"(densidad ×{factor_densidad:.1f})",
                      xaxis_title="Hora del día", yaxis_title="Valor (0–1)",
                      yaxis_range=[0, 1], hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    # ── Comparativa entre manzanas en el escenario simulado ──
    st.subheader("Comparativa entre manzanas en este escenario")
    filas = []
    for cv, r in riesgos.items():
        s = agente.calcular_score(r["danos_ponderados"],
                                  min(1.0, trafico.congestion_historica(dia, hora) * factor_densidad),
                                  r["altura_promedio_pisos"], r["num_fotos"])
        filas.append({"cvegeo": cv, "score_simulado": s["score_riesgo"],
                      "score_actual": r["score_riesgo"]})
    df = pd.DataFrame(filas).sort_values("score_simulado", ascending=False).head(20)
    fig2 = px.bar(df, x="cvegeo", y=["score_simulado", "score_actual"],
                  barmode="group",
                  color_discrete_sequence=["#d62728", "#9e9e9e"],
                  labels={"value": "Score", "variable": ""})
    fig2.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig2, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════
# Navegación
# ══════════════════════════════════════════════════════════════════════

st.set_page_config(page_title="Proyecto Urbano Agentivo", page_icon="🏙️", layout="wide")

paginas = st.navigation([
    st.Page(pagina_centro_mando, title="Centro de Mando Urbano", icon="🏙️", default=True),
    st.Page(pagina_laboratorio, title="Laboratorio de Modelos", icon="🧪"),
    st.Page(pagina_simulador, title="Simulador de Crisis", icon="🌋"),
])
paginas.run()
