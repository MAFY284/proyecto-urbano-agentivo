"""CDD — Centro de Control Dinámico de la Plataforma de Análisis Urbano.

Dashboard operativo estilo "mission control" (Streamlit multipágina):
  1. Centro de Mando Urbano        — diagnóstico agentivo por manzana (cvegeo),
     carga física / fragilidad estructural / carga humana expuesta, mapa
     inmersivo de riesgo y descargas GeoJSON/PDF.
  2. Laboratorio de Benchmarking   — grid comparativo de los 5 motores de
     ventanas con telemetría de inferencia en milisegundos.
  3. Simulador Dinámico de Crisis  — proyección del score al inyectar
     escenarios sísmicos (escala Mercalli) sobre el perfil vial histórico.

Ejecutar:  streamlit run src/dashboard/app.py
"""

from __future__ import annotations

import base64
import datetime
import sys
import time
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


# ══════════════════════════════════════════════════════════════════════
# Dirección de arte — paleta táctica y tipografías
# ══════════════════════════════════════════════════════════════════════

FONDO = "#0A0F1D"          # negro profundo operativo
SUPERFICIE = "#101827"     # paneles
LINEA = "#1e2a3f"          # bordes delgados
CIAN = "#00E5FF"           # acento visión/IA
VERDE_IA = "#00E676"       # verde matriz (alta confianza)
RIESGO_BAJO = "#81C784"    # verde menta
RIESGO_MEDIO = "#FFD54F"   # amarillo polvo
RIESGO_ALTO = "#FF1744"    # rojo alerta/neón
TINTA = "#E6EDF6"
TINTA_SUAVE = "#8494a7"

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Ubuntu:wght@400;500;700&family=JetBrains+Mono:wght@400;500;700&display=swap');

/* ── lienzo general ── */
[data-testid="stAppViewContainer"], [data-testid="stHeader"] {{
  background: {FONDO}; color: {TINTA};
}}
html, body, [class*="css"] {{ font-family: 'Ubuntu', sans-serif; }}
h1, h2, h3 {{ color: {TINTA}; letter-spacing: .01em; }}
[data-testid="stCaptionContainer"], .stCaption {{ color: {TINTA_SUAVE}; }}

/* ── sidebar técnico ── */
[data-testid="stSidebar"] {{
  background: {SUPERFICIE}; border-right: 1px solid {LINEA};
}}
[data-testid="stSidebar"] * {{ color: {TINTA}; }}

/* ── métricas nativas: valor gigante monoespaciado ── */
[data-testid="stMetricValue"] {{
  font-family: 'JetBrains Mono', monospace; font-weight: 700;
  color: {CIAN};
}}
[data-testid="stMetricLabel"] {{
  color: {TINTA_SUAVE}; text-transform: uppercase; letter-spacing: .08em;
  font-size: 11px;
}}

/* ── botón principal: alto contraste + pulso de telemetría activa ── */
.stButton > button[kind="primary"] {{
  background: linear-gradient(120deg, #d50000, {RIESGO_ALTO});
  color: #fff; font-weight: 700; letter-spacing: .06em;
  border: 1px solid {RIESGO_ALTO}; border-radius: 6px;
  box-shadow: 0 0 0 0 rgba(255, 23, 68, .55);
  animation: pulso-critico 1.8s infinite;
}}
.stButton > button[kind="primary"]:hover {{
  filter: brightness(1.15);
}}
@keyframes pulso-critico {{
  0%   {{ box-shadow: 0 0 0 0 rgba(255, 23, 68, .55); }}
  70%  {{ box-shadow: 0 0 0 12px rgba(255, 23, 68, 0); }}
  100% {{ box-shadow: 0 0 0 0 rgba(255, 23, 68, 0); }}
}}
.stButton > button:not([kind="primary"]),
.stDownloadButton > button {{
  background: {SUPERFICIE}; color: {CIAN}; border: 1px solid {LINEA};
  border-radius: 6px; font-family: 'JetBrains Mono', monospace; font-size: 13px;
}}
.stButton > button:not([kind="primary"]):hover,
.stDownloadButton > button:hover {{
  border-color: {CIAN}; box-shadow: 0 0 10px rgba(0, 229, 255, .25);
}}

/* ── controles ── */
[data-testid="stSlider"] [role="slider"] {{ background: {CIAN}; box-shadow: 0 0 8px {CIAN}; }}
[data-testid="stSlider"] [data-testid="stSliderTrack"] > div {{ background: {CIAN}; }}
[data-baseweb="select"] > div, .stTextInput input {{
  background: {FONDO}; border-color: {LINEA}; color: {TINTA};
  font-family: 'JetBrains Mono', monospace; font-size: 13px;
}}

/* ── tarjetas del CDD: borde delgado + sombra quirúrgica ── */
.cdd-card {{
  background: {SUPERFICIE}; border: 1px solid {LINEA}; border-radius: 8px;
  padding: 18px 20px 16px; height: 100%;
  box-shadow: 0 1px 0 rgba(0,229,255,.06), 0 8px 24px rgba(0,0,0,.35);
}}
.cdd-card.alerta {{ border-top: 2px solid {RIESGO_ALTO}; }}
.cdd-card.vision {{ border-top: 2px solid {CIAN}; }}
.cdd-card.estructura {{ border-top: 2px solid {VERDE_IA}; }}
.cdd-l {{
  font-size: 10.5px; font-weight: 700; letter-spacing: .14em;
  text-transform: uppercase; color: {TINTA_SUAVE}; margin-bottom: 6px;
}}
.cdd-v {{
  font-family: 'JetBrains Mono', monospace; font-weight: 700;
  font-size: 42px; line-height: 1.05; color: {TINTA};
}}
.cdd-v.cian {{ color: {CIAN}; text-shadow: 0 0 18px rgba(0,229,255,.35); }}
.cdd-v.verde {{ color: {VERDE_IA}; }}
.cdd-sub {{ font-size: 12px; color: {TINTA_SUAVE}; margin-top: 6px; }}
.cdd-sub .mono {{ font-family: 'JetBrains Mono', monospace; }}

/* ── chips de estado / nivel de riesgo ── */
.chip {{
  display: inline-block; font-family: 'JetBrains Mono', monospace;
  font-size: 11px; font-weight: 700; letter-spacing: .08em;
  padding: 3px 10px; border-radius: 999px; border: 1px solid;
}}
.chip.bajo   {{ color: {RIESGO_BAJO};  border-color: {RIESGO_BAJO};  background: rgba(129,199,132,.08); }}
.chip.medio  {{ color: {RIESGO_MEDIO}; border-color: {RIESGO_MEDIO}; background: rgba(255,213,79,.08); }}
.chip.alto   {{ color: {RIESGO_ALTO};  border-color: {RIESGO_ALTO};  background: rgba(255,23,68,.10);
               animation: pulso-critico 1.8s infinite; }}
.chip.tomtom {{ color: {VERDE_IA};     border-color: {VERDE_IA};     background: rgba(0,230,118,.08); }}
.chip.hist   {{ color: {RIESGO_MEDIO}; border-color: {RIESGO_MEDIO}; background: rgba(255,213,79,.08); }}

/* ── etiqueta flotante de motor (laboratorio) ── */
.tag-motor {{
  position: relative; z-index: 5; display: inline-block;
  margin-bottom: -34px; transform: translate(8px, 8px);
  font-family: 'JetBrains Mono', monospace; font-size: 11px; font-weight: 700;
  letter-spacing: .06em; color: {CIAN};
  background: rgba(10,15,29,.85); border: 1px solid {CIAN};
  border-radius: 4px; padding: 3px 8px;
  box-shadow: 0 0 12px rgba(0,229,255,.3);
}}
.tag-motor .ms {{ color: {VERDE_IA}; }}

/* ── dataframes / tablas ── */
[data-testid="stDataFrame"] {{ border: 1px solid {LINEA}; border-radius: 8px; }}

/* ── barra de progreso de confianza ── */
[data-testid="stProgress"] > div > div {{ background: {CIAN}; }}
</style>
"""


def nivel_riesgo(score: float | None) -> tuple[str, str]:
    """(clase css, texto) según el semáforo estricto del S_RU."""
    if score is None:
        return "medio", "SIN DATOS"
    if score >= 0.5:
        return "alto", "CRÍTICO"
    if score >= 0.25:
        return "medio", "MEDIO"
    return "bajo", "BAJO"


def color_por_score(score: float) -> str:
    if score >= 0.5:
        return RIESGO_ALTO
    if score >= 0.25:
        return RIESGO_MEDIO
    return RIESGO_BAJO


def tarjeta(col, clase: str, etiqueta: str, valor: str, sub: str = "",
            tono: str = "") -> None:
    """Tarjeta KPI del CDD: borde delgado, valor monoespaciado gigante."""
    col.markdown(
        f'<div class="cdd-card {clase}"><div class="cdd-l">{etiqueta}</div>'
        f'<div class="cdd-v {tono}">{valor}</div>'
        f'<div class="cdd-sub">{sub}</div></div>',
        unsafe_allow_html=True)


# ── Recursos compartidos (una sola instancia por sesión de servidor) ──

@st.cache_resource(show_spinner="Cargando capa de manzanas...")
def sig() -> AgenteSIG:
    return AgenteSIG()


@st.cache_resource(show_spinner="Inicializando Agente de Riesgo...")
def riesgo() -> AgenteRiesgo:
    return AgenteRiesgo()


def descargar_pdf_con_selector(pdf_bytes: bytes, nombre_sugerido: str, label: str) -> None:
    """Botón de descarga que abre el diálogo nativo "Guardar como…" del sistema
    operativo (File System Access API) en Chrome/Edge, para elegir carpeta y nombre
    en vez de descargar siempre a la carpeta de descargas por defecto del navegador.
    En Firefox/Safari, donde esa API no existe, cae a la descarga normal del navegador.
    """
    pdf_b64 = base64.b64encode(pdf_bytes).decode()
    html = f"""
    <button id="btnDescargarPdf" style="
        background:#101827;color:#00E5FF;border:1px solid #00E5FF;border-radius:6px;
        padding:0.55rem 1.1rem;font-size:0.95rem;cursor:pointer;
        font-family:'JetBrains Mono',monospace;">{label}</button>
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
    components.html(html, height=54)


# ══════════════════════════════════════════════════════════════════════
# Página 1 — Centro de Mando Urbano (CDD)
# ══════════════════════════════════════════════════════════════════════

def pagina_centro_mando():
    import folium
    from streamlit_folium import st_folium

    agente_sig = sig()
    riesgos = {r["cvegeo"]: r for r in db.listar_riesgos()}

    # ── Sidebar técnico: selector cvegeo + coordenadas + acción principal ──
    with st.sidebar:
        st.markdown("### ⬡ CDD · CONTROL")
        opciones = agente_sig.listar_manzanas()
        etiqueta = lambda cv: (f"{cv} · {riesgos[cv]['score_riesgo']:.3f}"
                               if cv in riesgos else f"{cv} · s/eval")
        cvegeo = st.selectbox("Clave CVEGEO (Agente SIG)", opciones, format_func=etiqueta)

        delim = agente_sig.delimitar(cvegeo)
        centro = delim["centroide"]
        st.text_input("LAT (centroide)", value=f"{centro['lat']:.6f}", disabled=True)
        st.text_input("LON (centroide)", value=f"{centro['lon']:.6f}", disabled=True)

        diagnosticar = st.button("🔴 INICIAR DIAGNÓSTICO URBANO AGENTIVO",
                                 type="primary", use_container_width=True)

    if diagnosticar:
        with st.status("TELEMETRÍA DE AGENTES — diagnóstico en curso", expanded=True) as estado:
            t0 = time.perf_counter()
            st.write(f"`[SIG]` Delimitación de manzana `{cvegeo}` · "
                     f"centroide `{centro['lat']:.5f}, {centro['lon']:.5f}`")
            st.write("`[TRÁFICO]` Consultando TomTom → fallback a perfil histórico 7×24 si falla…")
            riesgos[cvegeo] = riesgo().evaluar_manzana(cvegeo, centroide=centro)
            ms = (time.perf_counter() - t0) * 1000
            st.write(f"`[RIESGO]` S_RU computado: "
                     f"`{riesgos[cvegeo]['score_riesgo']:.3f}` · fuente congestión: "
                     f"`{riesgos[cvegeo]['fuente_congestion']}` · `{ms:.0f} ms`")
            estado.update(label=f"DIAGNÓSTICO COMPLETADO · {ms:.0f} ms", state="complete")

    # ── Encabezado ──
    r = riesgos.get(cvegeo)
    clase, texto = nivel_riesgo(r["score_riesgo"] if r else None)
    fuente = (r or {}).get("fuente_congestion")
    chip_fuente = ('<span class="chip tomtom">TELEMETRÍA TOMTOM · LIVE</span>'
                   if fuente == "tomtom" else
                   '<span class="chip hist">FALLBACK · PERFIL HISTÓRICO 7×24</span>'
                   if fuente else "")
    st.markdown(
        f"## ⬡ Centro de Mando Urbano &nbsp; "
        f'<span class="chip {clase}">S_RU {texto}</span> &nbsp; {chip_fuente}',
        unsafe_allow_html=True)
    st.caption(f"Manzana `{cvegeo}` — Score de Riesgo Urbano Combinado: "
               "daños 40 % · congestión 30 % · altura 30 %")

    # ── Panel central: tres columnas (física / estructural / humana) ──
    m2hab = settings.cargar()["poblacion"]["m2_por_habitante"]
    c1, c2, c3 = st.columns(3)
    if r:
        area = r.get("area_satelital_m2")
        pisos = r.get("altura_promedio_pisos") or 0
        area_total = area * pisos if (area and pisos) else None
        tarjeta(c1, "vision", "Carga física · YOLOv8-XL",
                f"{area:,.0f} m²" if area else "—",
                (f"Área total construida proyectada: "
                 f"<span class='mono'>{area_total:,.0f} m²</span>" if area_total
                 else "Sin segmentación satelital para esta manzana"),
                tono="cian")
        tarjeta(c2, "estructura", "Fragilidad estructural",
                f"{pisos:.1f}",
                f"Pisos estimados por correlación de ventanas · daños ponderados "
                f"<span class='mono'>{r['danos_ponderados']}</span>",
                tono="verde")
        tarjeta(c3, "alerta", "Carga humana expuesta",
                f"{r['poblacion_estimada'] or '—'}",
                f"Habitantes proyectados (estándar {m2hab} m²/hab) · nivel "
                f"<span class='chip {clase}'>{texto}</span>")
        st.progress(min(1.0, r["confianza"]),
                    text=f"Confianza de la muestra: {r['confianza']:.0%} "
                         f"({r['num_fotos']} fotos)")
    else:
        tarjeta(c1, "vision", "Carga física · YOLOv8-XL", "—", "Sin diagnóstico")
        tarjeta(c2, "estructura", "Fragilidad estructural", "—", "Sin diagnóstico")
        tarjeta(c3, "alerta", "Carga humana expuesta", "—",
                "Ejecuta 🔴 INICIAR DIAGNÓSTICO en el panel lateral")

    st.markdown("")

    # ── Mapa inmersivo (100 % del contenedor, base oscura) ──
    mapa = folium.Map(location=[centro["lat"], centro["lon"]], zoom_start=16,
                      tiles="CartoDB dark_matter")
    for feat in agente_sig.geojson.get("features", []):
        cv = feat["properties"].get("CVEGEO")
        info = riesgos.get(cv)
        color = color_por_score(info["score_riesgo"]) if info else "#37474F"
        tooltip = (f"{cv} · S_RU {info['score_riesgo']:.3f}" if info
                   else f"{cv} · sin datos")
        folium.GeoJson(
            feat,
            style_function=lambda _f, c=color, sel=(cv == cvegeo): {
                "fillColor": c, "color": CIAN if sel else c,
                "weight": 3 if sel else 1, "fillOpacity": 0.55},
            tooltip=tooltip,
        ).add_to(mapa)
    st_folium(mapa, height=540, use_container_width=True, key="mapa_riesgo")

    # ── Ranking ──
    if riesgos:
        import pandas as pd
        st.markdown("#### ▤ Ranking de manzanas por S_RU")
        df = pd.DataFrame(riesgos.values())[
            ["cvegeo", "score_riesgo", "danos_norm", "congestion", "pisos_norm",
             "confianza", "num_fotos", "poblacion_estimada", "fuente_congestion"]]
        st.dataframe(df.sort_values("score_riesgo", ascending=False),
                     use_container_width=True, hide_index=True)

    # ── Descargas ──
    st.markdown("#### ⇩ Exportación operativa")
    extra = {cv: {"score_riesgo": i["score_riesgo"], "congestion": i["congestion"],
                  "poblacion_estimada": i["poblacion_estimada"]}
             for cv, i in riesgos.items()}
    col_a, col_b = st.columns(2)
    with col_a:
        st.download_button("GEOJSON · TODAS LAS MANZANAS",
                           data=agente_sig.exportar_geojson(propiedades_extra=extra),
                           file_name="riesgo_manzanas.geojson",
                           mime="application/geo+json")
    with col_b:
        st.download_button(f"GEOJSON · {cvegeo}",
                           data=agente_sig.exportar_geojson([cvegeo], propiedades_extra=extra),
                           file_name=f"manzana_{cvegeo}.geojson",
                           mime="application/geo+json")

    # ── Reporte PDF ejecutivo ──
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
            f"- Población expuesta estimada ({m2hab} m2/hab): {r['poblacion_estimada'] or 'sin dato'}",
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
            label="⇩ REPORTE EJECUTIVO (PDF)",
        )


# ══════════════════════════════════════════════════════════════════════
# Página 2 — Laboratorio de Benchmarking de Modelos
# ══════════════════════════════════════════════════════════════════════

def pagina_laboratorio():
    st.markdown("## ⬡ Laboratorio de Benchmarking")
    st.caption("Grid comparativo de los 5 motores de detección de ventanas — "
               "telemetría de inferencia por arquitectura en milisegundos.")

    disponibles = comparador.motores_disponibles()
    if not disponibles:
        st.error("No hay motores disponibles: verifica los pesos en config/checkpoints/comparador/.")
        return

    with st.sidebar:
        st.markdown("### ⬡ CDD · BENCHMARK")
        motores = st.multiselect("Motores a comparar", list(disponibles),
                                 default=list(disponibles)[:2])
        conf = st.slider("Umbral de confianza", 0.0, 1.0, 0.30, 0.05)
        prompt_sam3 = "window"
        if comparador.MOTOR_SAM3 in motores:
            prompt_sam3 = st.text_input("Prompt de texto (SAM3)", value="window")
            st.caption("SAM3 corre como subproceso en su venv aislado (env_sam3).")
        for motor, ok in {comparador.MOTOR_DETECTRON2: comparador.detectron2_disponible(),
                          comparador.MOTOR_SAM3: comparador.sam3_disponible()}.items():
            if not ok:
                st.caption(f"· {motor} no disponible en este entorno (oculto).")

    archivos = st.file_uploader("Imágenes de fachada (una o varias)",
                                type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    if not archivos:
        st.info("Sube imágenes para iniciar el benchmarking.")
        return
    if not motores:
        st.warning("Elige al menos un motor en el panel lateral.")
        return

    from PIL import Image
    for archivo in archivos:
        st.markdown(f"#### `{archivo.name}`")
        imagen = Image.open(archivo).convert("RGB")
        cols = st.columns(len(motores) + 1)
        with cols[0]:
            st.markdown('<span class="tag-motor">ORIGINAL</span>', unsafe_allow_html=True)
            st.image(imagen, use_container_width=True)
        for i, motor in enumerate(motores, start=1):
            with cols[i]:
                with st.spinner(f"{motor}…"):
                    t0 = time.perf_counter()
                    r = comparador.inferir(motor, imagen, conf=conf, prompt_sam3=prompt_sam3)
                    ms = (time.perf_counter() - t0) * 1000
                if r.error:
                    st.markdown(f'<span class="tag-motor">{motor.upper()}</span>',
                                unsafe_allow_html=True)
                    st.error(r.error)
                else:
                    st.markdown(
                        f'<span class="tag-motor">{motor.upper()} · '
                        f'<span class="ms">{ms:,.0f} ms</span></span>',
                        unsafe_allow_html=True)
                    st.image(r.imagen_anotada, use_container_width=True)
                    st.caption(f"`{r.n_detecciones}` ventanas · confianza promedio "
                               f"`{r.confianza_promedio:.0%}`")
    st.toast("Benchmarking completado", icon="✅")


# ══════════════════════════════════════════════════════════════════════
# Página 3 — Simulador Dinámico de Crisis Sísmica
# ══════════════════════════════════════════════════════════════════════

# Escenarios predefinidos: intensidad Mercalli → factor de densidad vial
ESCENARIOS = {
    "Manual (slider)": None,
    "Mercalli V — evacuación parcial (×1.3)": 1.3,
    "Mercalli VII — pánico vial (×1.6)": 1.6,
    "Mercalli IX — colapso de vialidades (×2.0)": 2.0,
}

PLOTLY_OSCURO = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Ubuntu, sans-serif", color=TINTA),
    xaxis=dict(gridcolor="rgba(255,255,255,.07)", zerolinecolor="rgba(255,255,255,.12)"),
    yaxis=dict(gridcolor="rgba(255,255,255,.07)", zerolinecolor="rgba(255,255,255,.12)"),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
)


def pagina_simulador():
    import pandas as pd
    import plotly.graph_objects as go

    st.markdown("## ⬡ Simulador Dinámico de Crisis")
    st.caption("Proyección del S_RU al inyectar un escenario sísmico sobre el "
               "perfil histórico de congestión 7×24 de La Condesa.")

    agente = riesgo()
    riesgos = {r["cvegeo"]: r for r in db.listar_riesgos()}
    if not riesgos:
        st.warning("No hay manzanas evaluadas — ejecuta primero un diagnóstico "
                   "en el Centro de Mando o con `python main.py analizar`.")
        return

    with st.sidebar:
        st.markdown("### ⬡ CDD · ESCENARIO")
        cvegeo = st.selectbox("Manzana a simular", sorted(riesgos))
        dia = st.selectbox("Día de la semana", trafico.DIAS_ES,
                           index=datetime.date.today().weekday())
        hora = st.slider("Hora del día", 0, 23, datetime.datetime.now().hour)
        escenario = st.selectbox("Escenario sísmico", list(ESCENARIOS))
        if ESCENARIOS[escenario] is None:
            factor = st.slider("Factor de densidad vial", 0.0, 2.0, 1.0, 0.1,
                               help="1.0 = congestión histórica normal · "
                                    "2.0 = colapso vial · 0.0 = calles vacías")
        else:
            factor = ESCENARIOS[escenario]
            st.caption(f"Factor fijado por escenario: `×{factor:.1f}`")

    base = riesgos[cvegeo]
    fuente = base.get("fuente_congestion")
    chip_fuente = ('<span class="chip tomtom">TELEMETRÍA TOMTOM · LIVE</span>'
                   if fuente == "tomtom" else
                   '<span class="chip hist">FALLBACK ACTIVO · PERFIL HISTÓRICO 7×24</span>')
    st.markdown(f"Manzana `{cvegeo}` &nbsp; {chip_fuente}", unsafe_allow_html=True)

    # ── Score simulado en el escenario elegido ──
    congestion_sim = min(1.0, trafico.congestion_historica(dia, hora) * factor)
    sim = agente.calcular_score(base["danos_ponderados"], congestion_sim,
                                base["altura_promedio_pisos"], base["num_fotos"])

    c1, c2, c3 = st.columns(3)
    clase_sim, texto_sim = nivel_riesgo(sim["score_riesgo"])
    tarjeta(c1, "vision", "S_RU actual (BD)", f"{base['score_riesgo']:.3f}")
    tarjeta(c2, "alerta" if clase_sim == "alto" else "estructura",
            "S_RU simulado", f"{sim['score_riesgo']:.3f}",
            f"Δ <span class='mono'>{sim['score_riesgo'] - base['score_riesgo']:+.3f}</span> · "
            f"nivel <span class='chip {clase_sim}'>{texto_sim}</span>",
            tono="cian")
    tarjeta(c3, "vision", "Congestión simulada", f"{congestion_sim:.0%}",
            f"{escenario if ESCENARIOS[escenario] else f'Factor manual ×{factor:.1f}'} · "
            f"{dia} {hora:02d}:00")
    st.markdown("")

    # ── Curva del día completo (24 h) — líneas neón, fondo transparente ──
    horas = list(range(24))
    scores_dia = [agente.calcular_score(
        base["danos_ponderados"],
        min(1.0, trafico.congestion_historica(dia, h) * factor),
        base["altura_promedio_pisos"], base["num_fotos"])["score_riesgo"]
        for h in horas]
    congestiones = [min(1.0, trafico.congestion_historica(dia, h) * factor)
                    for h in horas]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=horas, y=scores_dia, mode="lines+markers",
                             name="S_RU (score de riesgo)",
                             line=dict(width=3, color=CIAN),
                             marker=dict(size=5, color=CIAN)))
    fig.add_trace(go.Scatter(x=horas, y=congestiones, mode="lines",
                             name=f"Congestión — {'TomTom live' if fuente == 'tomtom' else 'fallback histórico 7×24'}",
                             line=dict(dash="dot", width=2, color=RIESGO_MEDIO)))
    fig.add_vline(x=hora, line_dash="dash", line_color=RIESGO_ALTO,
                  annotation_text=f"{hora:02d}h",
                  annotation_font_color=RIESGO_ALTO)
    fig.update_layout(
        title=f"Evolución del riesgo en 24 h — {cvegeo} · {dia} (densidad ×{factor:.1f})",
        xaxis_title="Hora del día", yaxis_title="Valor (0–1)",
        yaxis_range=[0, 1], hovermode="x unified", **PLOTLY_OSCURO)
    st.plotly_chart(fig, use_container_width=True)

    # ── Comparativa entre manzanas en el escenario simulado ──
    st.markdown("#### ▤ Comparativa entre manzanas en este escenario")
    filas = []
    for cv, r in riesgos.items():
        s = agente.calcular_score(r["danos_ponderados"],
                                  min(1.0, trafico.congestion_historica(dia, hora) * factor),
                                  r["altura_promedio_pisos"], r["num_fotos"])
        filas.append({"cvegeo": cv, "score_simulado": s["score_riesgo"],
                      "score_actual": r["score_riesgo"]})
    df = pd.DataFrame(filas).sort_values("score_simulado", ascending=False).head(20)

    fig2 = go.Figure()
    fig2.add_trace(go.Bar(x=df["cvegeo"], y=df["score_simulado"],
                          name="Score simulado", marker_color=RIESGO_ALTO))
    fig2.add_trace(go.Bar(x=df["cvegeo"], y=df["score_actual"],
                          name="Score actual", marker_color="#546E7A"))
    fig2.update_layout(barmode="group", xaxis_tickangle=-45,
                       yaxis_title="Score", **PLOTLY_OSCURO)
    st.plotly_chart(fig2, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════
# Navegación
# ══════════════════════════════════════════════════════════════════════

st.set_page_config(page_title="CDD · Plataforma de Análisis Urbano",
                   page_icon="⬡", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)

paginas = st.navigation([
    st.Page(pagina_centro_mando, title="Centro de Mando Urbano", icon="🏙️", default=True),
    st.Page(pagina_laboratorio, title="Laboratorio de Modelos", icon="🧪"),
    st.Page(pagina_simulador, title="Simulador de Crisis", icon="🌋"),
])
paginas.run()
