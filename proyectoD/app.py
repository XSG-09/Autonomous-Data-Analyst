import streamlit as st
import pandas as pd
import plotly.express as px 
from motor.perfil import infoDataSet
from motor.limpieza import limpieza_estandarizar
from motor.decisionTrata import decisionGrafica, mostrarVarImportantes


st.set_page_config(
    page_title="Autonomous Data Analyst",
    page_icon="📊",
    layout="wide",
)

st.title("Autonomous Data Analyst")
st.write("Carga un dataset para comenzar el análisis.")

archivo = st.file_uploader(
    "Selecciona un archivo CSV",
    type=["csv"],
)

if archivo is not None:
    try:
        df, df_procesado = limpieza_estandarizar(archivo)

        st.session_state.df_original = df.copy()
        st.session_state.df_procesado = df_procesado
        st.session_state.varImportantes = mostrarVarImportantes(df)

        if st.button("Ir a la segunda página"):
            st.switch_page("pages/paginaDos.py")
    
        decisionGrafica(df)
       

    except (ValueError, UnicodeDecodeError, OSError) as exc:
        st.error(f"No se pudo procesar el archivo: {exc}")
