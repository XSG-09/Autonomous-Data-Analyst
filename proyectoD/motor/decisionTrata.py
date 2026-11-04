import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
from io import BytesIO
from fpdf import FPDF

class PDFReport(FPDF):
    def header(self):
        self.set_font('helvetica', 'B', 14)
        self.cell(0, 10, 'Reporte de Análisis Exploratorio de Datos', align='C', new_x='LMARGIN', new_y='NEXT')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('helvetica', 'I', 8)
        self.cell(0, 10, f'Página {self.page_no()}', align='C')

def crear_pdf_reporte(elementos_reporte):
    pdf = PDFReport()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    
    pdf.set_font('helvetica', 'B', 16)
    pdf.cell(0, 10, 'Resumen de Gráficas y Conclusiones', new_x='LMARGIN', new_y='NEXT')
    pdf.ln(5)

    for variable_nombre, conclusion, fig in elementos_reporte:
        pdf.set_font('helvetica', 'B', 12)
        pdf.cell(0, 8, f'Variable / Análisis: {variable_nombre}', new_x='LMARGIN', new_y='NEXT')
        
        texto_limpio = conclusion.replace("**", "").replace("### ", "").replace("## ", "")
        
        pdf.set_font('helvetica', '', 10)
        pdf.multi_cell(0, 6, texto_limpio)
        pdf.ln(4)
        

        if fig is not None:
            try:
                img_bytes = fig.to_image(format="png", width=700, height=350, scale=2)
                img_buffer = BytesIO(img_bytes)
                
                if pdf.get_y() > 200:
                    pdf.add_page()

                pdf.image(img_buffer, w=160)
                pdf.ln(8)
            except Exception as e:
                pdf.set_font('helvetica', 'I', 9)
                pdf.cell(0, 6, f"(No se pudo exportar la imagen: {str(e)})", new_x='LMARGIN', new_y='NEXT')
                pdf.ln(4)
                
        pdf.set_draw_color(200, 200, 200)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(6)

    return bytes(pdf.output())



def valoresFaltantes(df: pd.DataFrame, column, tipoC=None):
    cantidad = int(df[column].isna().sum())

    if cantidad == 0:
        return {
            "column": column,
            "problem": "missing_values",
            "action": "review",
            "reason": "No hay valores faltantes."
        }

    if tipoC is None:
        return {
            "column": column,
            "problem": "missing_values",
            "action": "review",
            "reason": "Se desconoce el significado de la variable."
        }

    if tipoC == "identifier":
        return {
            "column": column,
            "problem": "missing_values",
            "action": "review",
            "reason": "Un identificador faltante no se debe imputar con una estadística."
        }

    if tipoC == "numeric":
        if not pd.api.types.is_numeric_dtype(df[column]):
            return {
                "column": column,
                "problem": "missing_values",
                "action": "review",
                "reason": "El tipo declarado no coincide con los datos."
            }

        values = df[column].dropna()

        if len(values) >= 3 and abs(values.skew()) > 1:
            metodo = "median"
        else:
            metodo = "mean"

        return {
            "column": column,
            "problem": "missing_values",
            "action": "impute",
            "metodo": metodo
        }

    return {
        "column": column,
        "problem": "missing_values",
        "action": "review",
        "reason": "Tipo de variable no reconocido."
    }

def personalizacionGraficas(df, tipoGraf: str, varImportante, varMayorC=None):
    if tipoGraf == "histograma":
        fig = px.histogram(
            df,
            x=varImportante,
            nbins=50,
            title=f"Distribución de {varImportante}",
            color_discrete_sequence=["#00CC96"]
        )

        fig.update_traces(
            marker=dict(line=dict(color="#111111", width=1))
        )

    elif tipoGraf == "barras":
        df_counts = (
            df[varImportante]
            .dropna()
            .value_counts()
            .rename_axis(varImportante)
            .reset_index(name="conteo")
        )

        fig = px.bar(
            df_counts,
            x=varImportante,
            y="conteo",
            color="conteo",
            color_continuous_scale="Viridis",
            title=f"Frecuencias de {varImportante}"
        )

    elif tipoGraf == "scatter":
        datos = df[[varMayorC, varImportante]].dropna()

        fig = px.scatter(
            datos,
            x=varMayorC,
            y=varImportante,
            opacity=0.8,
            title=f"{varImportante} vs {varMayorC}",
            color_discrete_sequence=["#AB63FA"]
        )

        fig.update_traces(
            marker=dict(line=dict(width=0.5, color="white"))
        )

    else:
        raise ValueError(f"Tipo de gráfica desconocido: {tipoGraf}")

    fig.update_layout(
        height=320,
        autosize=True,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white", family="sans-serif", size=11),
        title_font=dict(size=16, color="#00CC96"),
        margin=dict(l=35, r=20, t=55, b=40),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.1)")
    )

    return fig


def concluir_histograma(df, variable):
    datos = df[variable].dropna()

    if datos.empty:
        return "No hay datos disponibles para analizar esta variable."

    media = datos.mean()
    mediana = datos.median()
    desviacion = datos.std()

    if len(datos) >= 3:
        asimetria = datos.skew()

        if asimetria > 0.5:
            forma = "sesgo hacia la derecha"
        elif asimetria < -0.5:
            forma = "sesgo hacia la izquierda"
        else:
            forma = "distribución aproximadamente simétrica"
    else:
        forma = "insuficientes observaciones para evaluar la asimetría"

    return (
        f"**Distribución de {variable}**\n\n"
        f"- Media: {media:.2f}\n"
        f"- Mediana: {mediana:.2f}\n"
        f"- Desviación estándar: "
        f"{desviacion:.2f}\n\n"
        f"La distribución presenta {forma}. "
        f"La desviación estándar resume la dispersión "
        f"de los datos respecto a su media."
    )


def concluir_barras(df, variable):
    datos = df[variable].dropna()

    if datos.empty:
        return "No hay datos disponibles para analizar."

    frecuencias = datos.value_counts()
    categoria = frecuencias.index[0]
    cantidad = int(frecuencias.iloc[0])
    porcentaje = cantidad / len(datos) * 100

    if porcentaje >= 70:
        interpretacion = "un claro predominio de una categoría"
    elif porcentaje >= 40:
        interpretacion = "una concentración importante en una categoría"
    else:
        interpretacion = "ninguna categoría alcanza el 40 % de la muestra"

    return (
        f"**Frecuencias de {variable}**\n\n"
        f"- Categorías distintas: {datos.nunique()}\n"
        f"- Categoría más frecuente: {categoria}\n"
        f"- Frecuencia: {cantidad} ({porcentaje:.1f} %)\n\n"
        f"Los datos muestran {interpretacion}. "
        f"Esta conclusión describe las frecuencias observadas."
    )


def concluir_scatter(df, x, y):
    datos = df[[x, y]].dropna()

    if len(datos) < 3:
        return "No hay suficientes observaciones para analizar la relación."

    correlacion = datos[x].corr(datos[y])

    if pd.isna(correlacion):
        return (
            f"No se pudo calcular la correlación entre {x} y {y}. "
            "Comprueba si alguna variable es constante."
        )

    magnitud = abs(correlacion)

    if magnitud < 0.2:
        fuerza = "muy débil"
    elif magnitud < 0.4:
        fuerza = "débil"
    elif magnitud < 0.7:
        fuerza = "moderada"
    else:
        fuerza = "fuerte"

    if correlacion > 0:
        direccion = "positiva"
    elif correlacion < 0:
        direccion = "negativa"
    else:
        direccion = "nula"

    return (
        f"**Relación entre {x} y {y}**\n\n"
        f"- Correlación de Pearson: {correlacion:.3f}\n\n"
        f"Se observa una asociación lineal {fuerza} y {direccion}. "
        "La correlación no demuestra causalidad y puede no "
        "representar adecuadamente una relación no lineal."
    )


def concluir_caja(df, variable, grupo=None):
    columnas = [variable] if grupo is None else [grupo, variable]
    datos = df[columnas].dropna()

    if datos.empty:
        return "No hay datos disponibles para analizar."

    serie = datos[variable]
    q1 = serie.quantile(0.25)
    mediana = serie.median()
    q3 = serie.quantile(0.75)
    ric = q3 - q1

    atipicos = (
        (serie < q1 - 1.5 * ric) |
        (serie > q3 + 1.5 * ric)
    ).sum()

    texto = (
        f"**Distribución de {variable}**\n\n"
        f"- Mediana: {mediana:.2f}\n"
        f"- Primer cuartil: {q1:.2f}\n"
        f"- Tercer cuartil: {q3:.2f}\n"
        f"- Rango intercuartílico: {ric:.2f}\n"
        f"- Posibles valores atípicos: {atipicos}\n\n"
        "Los valores atípicos se identifican mediante "
        "la regla de 1,5 veces el rango intercuartílico."
    )

    if grupo is not None:
        medianas = datos.groupby(
            grupo, observed=True
        )[variable].median()

        if not medianas.empty:
            texto += (
                f"\n\nEl grupo con mayor mediana es "
                f"**{medianas.idxmax()}** "
                f"({medianas.max():.2f})."
            )

    return texto


def concluir_mapa_calor(df, x, y):
    datos = df[[x, y]].dropna()

    if datos.empty:
        return "No hay datos disponibles para analizar."

    frecuencias = pd.crosstab(datos[x], datos[y])

    if frecuencias.empty:
        return "No se encontraron combinaciones para comparar."

    fila, columna = np.unravel_index(
        np.argmax(frecuencias.to_numpy()),
        frecuencias.shape
    )

    categoria_x = frecuencias.index[fila]
    categoria_y = frecuencias.columns[columna]
    cantidad = int(frecuencias.iloc[fila, columna])

    return (
        f"**Relación entre {x} y {y}**\n\n"
        f"- Combinación más frecuente: "
        f"{categoria_x} / {categoria_y}\n"
        f"- Observaciones en esa combinación: {cantidad}\n"
        f"- Total de observaciones analizadas: {len(datos)}\n\n"
        "El mapa representa frecuencias conjuntas. Para determinar "
        "si existe asociación estadística, se requiere un análisis "
        "adicional, por ejemplo, una prueba chi-cuadrado."
    )


def mostrar_grafica_conclusion(fig, conclusion):
    col_grafica, col_conclusiones = st.columns([3, 2])

    with col_grafica:
        st.plotly_chart(
            fig,
            use_container_width=True,
            config={"responsive": True}
        )

    with col_conclusiones:
        st.markdown("### Conclusiones")
        st.markdown(conclusion)



def decisionGrafica(df: pd.DataFrame):
    
    if df.empty:
        st.warning("El DataFrame está vacío.")
        return

    numericas = df.select_dtypes(include=np.number).columns.tolist()

    categoricas = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    if not numericas and not categoricas:
        st.warning("No hay variables compatibles para analizar.")
        return

    if len(numericas) >= 2:
        matriz_corr = df[numericas].corr().abs()

        promedio_corr = (matriz_corr.sum() - 1) / (len(matriz_corr) - 1)

        top_variables = promedio_corr.sort_values(ascending=False).head(5).index.tolist()

    else:
        top_variables = numericas.copy()

    for columna in categoricas:
        if columna not in top_variables and len(top_variables) < 5:
            top_variables.append(columna)

    elementos_pdf = []

    for variable in top_variables:
        st.markdown("---")
        st.markdown(f"Análisis para la variable: **{variable}**")

        es_numerica = pd.api.types.is_numeric_dtype(df[variable])

        varMayorC = None

        if es_numerica:
            otras_numericas = [c for c in numericas if c != variable]

            if otras_numericas:
                correlaciones = (df[otras_numericas + [variable]].corr(numeric_only=True)[variable].drop(variable, errors="ignore").abs().dropna() )

                if not correlaciones.empty:
                    varMayorC = correlaciones.idxmax()

        elif numericas:
            varMayorC = numericas[0]

        if es_numerica:
            fig = personalizacionGraficas(df, "histograma", variable, varMayorC)
            conclusion = concluir_histograma(df, variable)
        else:
            fig = personalizacionGraficas(df, "barras", variable, varMayorC)
            conclusion = concluir_barras(df, variable)

        mostrar_grafica_conclusion(fig, conclusion)
        elementos_pdf.append((f"{variable} (Univariado)", conclusion, fig))

        if varMayorC is None:
            continue

        if es_numerica and pd.api.types.is_numeric_dtype(df[varMayorC]):
            fig = personalizacionGraficas(df, "scatter", variable, varMayorC)
            conclusion = concluir_scatter(df, varMayorC, variable)

        elif es_numerica:
            fig = px.box(df, x=varMayorC, y=variable, title=f"{variable} por {varMayorC}")
            fig.update_layout(
                height=320,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white"),
                margin=dict(l=35, r=20, t=55, b=40)
            )
            conclusion = concluir_caja(df, variable, varMayorC)

        elif pd.api.types.is_numeric_dtype(df[varMayorC]):
            fig = px.box(df, x=variable, y=varMayorC, title=f"{varMayorC} agrupado por {variable}")
            fig.update_layout(
                height=320,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white"),
                margin=dict(l=35, r=20, t=55, b=40)
            )
            conclusion = concluir_caja(df, varMayorC, variable)

        else:
            fig = px.density_heatmap(df, x=variable, y=varMayorC, title=f"{variable} vs {varMayorC}", text_auto=True)
            fig.update_layout(
                height=320,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white"),
                margin=dict(l=35, r=20, t=55, b=40)
            )
            conclusion = concluir_mapa_calor(df, variable, varMayorC)

        mostrar_grafica_conclusion(fig, conclusion)
        elementos_pdf.append((f"{variable} vs {varMayorC}", conclusion, fig))


    if elementos_pdf:
        st.markdown("---")
        st.markdown("Exportar Análisis Completo")
        
        with st.spinner("Preparando reporte en PDF..."):
            pdf_bytes = crear_pdf_reporte(elementos_pdf)
            
        st.download_button(
            label="📥 Descargar Reporte en PDF (Gráficas y Conclusiones)",
            data=pdf_bytes,
            file_name="reporte_analisis_dashboard.pdf",
            mime="application/pdf",
            use_container_width=True
        )


def mostrarVarImportantes(df: pd.DataFrame):
    numericas = df.select_dtypes(include=np.number).columns.tolist()
    categoricas = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    if not numericas and not categoricas:
        st.warning("No hay variables compatibles para analizar.")
        return

    if len(numericas) >= 2:
        matriz_corr = df[numericas].corr().abs()
        promedio_corr = (matriz_corr.sum() - 1) / (len(matriz_corr) - 1)

        global top_variables
        top_variables = promedio_corr.sort_values(ascending=False).head(5).index.tolist()

    else:
        top_variables = numericas.copy()

    for columna in categoricas:
        if columna not in top_variables and len(top_variables) < 5:
            top_variables.append(columna)

    return top_variables