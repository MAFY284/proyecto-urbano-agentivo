import streamlit as st

st.set_page_config(
    page_title="Herramientas de IA Urbana",
    page_icon="🏙️",
)

st.title("Bienvenido al Analizador Urbano 🏙️")
st.write("Selecciona una herramienta en el menú de la izquierda:")
st.info("🛰️ **Satélite Condesa:** Detecta metros cuadrados de edificios desde el espacio usando YOLOv8.")
st.info("🪟 **Ventanas Edificios:** Detecta e inspecciona ventanas en fachadas usando Detectron2.")