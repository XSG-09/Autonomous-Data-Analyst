import os
import sys
import streamlit as st

if "df_original" not in st.session_state:
    st.warning("Primero carga un dataset en la página principal.")
    st.stop()

df_original = st.session_state.df_original
var_importantes = list(st.session_state.varImportantes)

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from predicion import entrenar_modelo_y_predecir, inputs

st.set_page_config(layout="wide")

st.markdown(
    """
    <style>
    div.stButton > button {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: #ffffff;
        border-radius: 12px;
        border: none;
        font-weight: bold;
        padding: 10px 20px;
        width: 100%;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
        transition: all 0.3s ease;
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 242, 254, 0.4);
        background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%);
        color: #ffffff;
    }
    div.stButton > button:disabled {
        background: #2a2e39;
        color: #6c757d;
        box-shadow: none;
        transform: none;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Variables")
st.write("Selecciona la variable que quieres predecir.")

if "varImportantes" not in st.session_state or "df_procesado" not in st.session_state:
    st.warning(
        "Por favor, sube y procesa un archivo primero en la página principal."
    )
    st.stop()

df_procesado = st.session_state.df_procesado
var_importantes = list(st.session_state.varImportantes)

if df_procesado is None or len(var_importantes) == 0:
    st.warning("No hay datos procesados o no se encontraron variables importantes.")
    st.stop()

if "boton_activo" not in st.session_state:
    st.session_state.boton_activo = None

columnas = st.columns(min(len(var_importantes), 5))
for i, nombre in enumerate(var_importantes):
    with columnas[i % len(columnas)]:
        if st.button(
            str(nombre),
            key=f"btn_objetivo_{i}_{nombre}",
            use_container_width=True,
        ):
            st.session_state.boton_activo = nombre
            st.session_state.pop("modelo_entrenado", None)
            st.rerun()

if st.session_state.boton_activo is not None:
    columna_y = st.session_state.boton_activo
    st.markdown("---")
    st.success(f"Has seleccionado **{columna_y}** como variable a predecir.")

    if st.button("Reiniciar selección", use_container_width=False):
        st.session_state.boton_activo = None
        st.session_state.pop("modelo_entrenado", None)
        st.rerun()

    clave_modelo = f"modelo_entrenado::{columna_y}"

    if clave_modelo not in st.session_state:
        try:
            with st.spinner("Entrenando la red neuronal; esto puede tardar..."):
                st.session_state[clave_modelo] = entrenar_modelo_y_predecir(
                    columna_y, df_original, var_importantes
                )
            st.success("¡Modelo entrenado con éxito!")
        except Exception as error:
            st.error(f"No se pudo entrenar el modelo: {error}")
            st.info(
                "Comprueba que la variable objetivo exista en el DataFrame y que "
                "contenga valores numéricos si quieres predecir una cantidad."
            )
            st.stop()

    st.subheader("Ingresa los datos para predecir")
    inputs(var_importantes, st.session_state[clave_modelo])