
import pandas as pd
import numpy as np
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier


def entrenar_modelo_y_predecir(columnaY, df_original, topVariables):
    df = df_original.copy()

    if df.empty:
        raise ValueError("El dataset está vacío.")

    df.columns = df.columns.map(lambda c: str(c).strip())

    if df.columns.duplicated().any():
        raise ValueError(
            "Hay nombres de columnas duplicados. "
            "Corrígelos antes de entrenar."
        )

    if columnaY not in df.columns:
        raise ValueError(
            f"No existe la variable '{columnaY}'. "
            f"Columnas disponibles: {list(df.columns)}"
        )

    df = df.dropna(subset=[columnaY]).copy()

    if len(df) < 2:
        raise ValueError(
            "Se necesitan al menos dos filas con objetivo válido."
        )

    y = df[columnaY]

    caracteristicas = [
        c for c in topVariables
        if c in df.columns and c != columnaY
    ]

    if not caracteristicas:
        caracteristicas = [
            c for c in df.columns if c != columnaY
        ]

    if not caracteristicas:
        raise ValueError(
            "No existen variables de entrada para predecir."
        )

    X = df[caracteristicas].copy()

    numericas = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categoricas = [
        c for c in X.columns if c not in numericas
    ]

    for c in categoricas:
        X[c] = X[c].map(
            lambda v: str(v) if pd.notna(v) else np.nan
        )

    transformadores = []

    if numericas:
        transformadores.append((
            "num",
            SimpleImputer(strategy="median"),
            numericas
        ))

    if categoricas:
        transformadores.append((
            "cat",
            Pipeline([
                (
                    "imputar",
                    SimpleImputer(strategy="constant",
                                  fill_value="Desconocido")
                ),
                (
                    "codificar",
                    OneHotEncoder(handle_unknown="ignore")
                )
            ]),
            categoricas
        ))

    preprocesador = ColumnTransformer(
        transformers=transformadores
    )

    es_regresion = pd.api.types.is_numeric_dtype(y)

    if es_regresion:
        estimador = RandomForestRegressor(
            n_estimators=150,
            random_state=42,
            n_jobs=-1
        )
        tipo = "regresion"
    else:
        estimador = RandomForestClassifier(
            n_estimators=150,
            random_state=42,
            n_jobs=-1
        )
        tipo = "clasificacion"

    modelo = Pipeline([
        ("preprocesamiento", preprocesador),
        ("modelo", estimador)
    ])

    modelo.fit(X, y)

    return {
        "pipeline": modelo,
        "caracteristicas": caracteristicas,
        "referencia": df[caracteristicas].copy(),
        "tipo": tipo,
        "objetivo": columnaY
    }


def inputs(topVariables, modelo):
    caracteristicas = modelo["caracteristicas"]
    referencia = modelo["referencia"]

    st.write(
        "Introduce los valores de las variables "
        "para obtener una predicción."
    )

    with st.form("form_prediccion"):
        datos = {}

        for nombre in caracteristicas:
            serie = referencia[nombre]

            if pd.api.types.is_numeric_dtype(serie):
                valores = pd.to_numeric(
                    serie, errors="coerce"
                ).dropna()

                valor_inicial = (
                    float(valores.median())
                    if not valores.empty else 0.0
                )

                datos[nombre] = st.number_input(
                    f"Valor para {nombre}",
                    value=valor_inicial,
                    key=f"pred_{nombre}"
                )

            else:
                opciones = sorted(
                    serie.dropna().astype(str).unique().tolist()
                )

                opciones = ["Desconocido"] + [
                    x for x in opciones
                    if x != "Desconocido"
                ]

                datos[nombre] = st.selectbox(
                    f"Valor para {nombre}",
                    options=opciones,
                    key=f"pred_{nombre}"
                )

        boton = st.form_submit_button(
            "Calcular predicción"
        )

    if boton:
        entrada = pd.DataFrame(
            [datos], columns=caracteristicas
        )

        prediccion = modelo["pipeline"].predict(entrada)[0]

        st.success("¡Predicción completada!")

        if modelo["tipo"] == "regresion":
            st.metric(
                "Resultado de la predicción",
                f"{float(prediccion):,.4f}"
            )
        else:
            st.metric(
                "Clase predicha",
                str(prediccion)
            )

            pipeline = modelo["pipeline"]
            clasificador = pipeline.named_steps["modelo"]

            if hasattr(clasificador, "predict_proba"):
                probabilidades = pipeline.predict_proba(
                    entrada
                )[0]

                indice = list(clasificador.classes_).index(
                    prediccion
                )

                st.write(
                    f"Probabilidad estimada de esa clase: "
                    f"{probabilidades[indice] * 100:.2f}%"
                )

