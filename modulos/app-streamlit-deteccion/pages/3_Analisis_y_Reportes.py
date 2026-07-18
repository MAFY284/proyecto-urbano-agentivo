import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import plotly.express as px
from fpdf import FPDF
import base64
import datetime


def descargar_pdf_con_selector(pdf_bytes: bytes, nombre_sugerido: str, label: str):
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

st.set_page_config(page_title="Fusión Analítica Urbana", page_icon="📊", layout="wide")
st.title("📊 Dashboard de Fusión Analítica Urbana")
st.write("Esta pestaña cruza los datos del espacio aéreo (YOLOv8) con la fachada terrestre (Detectron2).")

# 1. VALIDACIÓN: Verificar si el usuario ya procesó las imágenes en las otras pestañas
if 'area_satelital' not in st.session_state or 'total_ventanas' not in st.session_state:
    st.warning("⚠️ **Falta información para la fusión.** Por favor, procesa primero una imagen en la pestaña de 'Satélite' y otra en la pestaña de 'Ventanas'.")
else:
    # Recuperar datos guardados en la memoria
    a_sat = st.session_state['area_satelital']
    ventanas = st.session_state['total_ventanas']
    coords = st.session_state.get('coordenadas', 'No especificadas')
    
    # 2. ALGORITMO DE FUSIÓN REAL (Cálculos de investigación)
    # Asumimos una constante de 4 ventanas promedio por piso en esa fachada
    pisos_est = max(1, round(ventanas / 4)) 
    area_construida_total = a_sat * pisos_est
    # Asumimos la norma urbana de 35m² por habitante para calcular población
    poblacion_est = round(area_construida_total / 35)

    st.success("✅ ¡Datos cruzados con éxito! Modelos sincronizados.")

    # 3. COMPONENTE DASHBOARD: Tarjetas de Métricas (KPIs)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Área de Techo (YOLOv8)", f"{a_sat:.2f} m²")
    col2.metric("Ventanas (Detectron2)", f"{ventanas} unidades")
    col3.metric("Niveles Estimados (Fusión)", f"{pisos_est} pisos")
    col4.metric("Población Proyectada", f"{poblacion_est} hab")

    st.markdown("---")

    # 4. COMPONENTE DASHBOARD: Gráfico Interactivo (Plotly)
    st.subheader("Impacto de la Densificación Volumétrica")
    df_chart = pd.DataFrame({
        "Concepto": ["Área Ocupada en Suelo", "Superficie Total Construida (Pisos)"],
        "Metros Cuadrados": [a_sat, area_construida_total]
    })
    
    fig = px.bar(df_chart, x="Concepto", y="Metros Cuadrados", color="Concepto", 
                 text_auto='.2f', color_discrete_sequence=["#0f766e", "#38bdf8"])
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # 5. COMPONENTE REPORTE: Función para armar el PDF en memoria
    def generar_pdf_bytes(area_techo, num_ventanas, num_pisos, area_tot, pob, coordenadas):
        pdf = FPDF()
        pdf.add_page()
        
        # Encabezado del reporte
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 10, "REPORTE TÉCNICO DE INTELIGENCIA URBANA", ln=True, align="C")
        pdf.set_font("Helvetica", "I", 10)
        pdf.cell(0, 10, f"Fecha de emisión: {datetime.date.today().strftime('%d/%m/%Y')}", ln=True, align="C")
        pdf.cell(0, 5, f"Coordenadas del análisis: {coordenadas}", ln=True, align="C")
        pdf.ln(10)
        
        # Sección 1: Resultados individuales
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 10, "1. Métricas de Modelos de Inteligencia Artificial", ln=True)
        pdf.set_font("Helvetica", "", 11)
        pdf.cell(0, 8, f"- Superficie de desplante predial (Techo - YOLOv8 XL): {area_techo:.2f} m2", ln=True)
        pdf.cell(0, 8, f"- Conteo de vanos en fachada (Ventanas - Detectron2): {num_ventanas} unidades", ln=True)
        pdf.ln(5)
        
        # Sección 2: Resultados de la Fusión
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 10, "2. Analíticas de Fusión de Datos (Data Fusion)", ln=True)
        pdf.set_font("Helvetica", "", 11)
        pdf.cell(0, 8, f"- Estimación del número de niveles del inmueble: {num_pisos} pisos", ln=True)
        pdf.cell(0, 8, f"- Superficie total construida proyectada: {area_tot:.2f} m2", ln=True)
        pdf.cell(0, 8, f"- Carga poblacional estimada por densidad normativa: {pob} habitantes", ln=True)
        pdf.ln(15)
        
        # Nota legal/científica al pie
        pdf.set_font("Helvetica", "I", 9)
        pdf.multi_cell(0, 5, "Nota: Este documento es un entregable científico automatizado generado mediante modelos de Visión por Computadora. Los datos son estimaciones proyectuales basadas en densidades promedio.")
        
        return bytes(pdf.output())

    # Botón de descarga del PDF en Streamlit
    st.subheader("💾 Exportación de Resultados")
    st.write("Genera un documento PDF oficial con las gráficas y analíticas listas para tu entrega del Verano Científico:")
    
    # Ejecutar la función y obtener los bytes del PDF
    pdf_output = generar_pdf_bytes(a_sat, ventanas, pisos_est, area_construida_total, poblacion_est, coords)
    
    descargar_pdf_con_selector(
        pdf_bytes=pdf_output,
        nombre_sugerido=f"Reporte_Inteligencia_Urbana_{datetime.date.today()}.pdf",
        label="📥 Descargar Reporte Ejecutivo en PDF",
    )