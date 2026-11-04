import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from motor.decisionTrata import valoresFaltantes

def aplicarMetodo(df: pd.DataFrame, decision: dict):
    df = df.copy()
    columna = decision["column"]
    action = decision["action"]

    if action == "impute":

        if decision["metodo"] == "median":
            valor = df[columna].median()
        elif decision["metodo"] == "mean":
            valor = df[columna].mean()
        else:
            raise ValueError("Método no soportado")

        df[columna] = df[columna].fillna(valor)

    return df

def limpieza_estandarizar(archivo):
    if archivo.name.lower().endswith(".csv"):
        df = pd.read_csv(archivo)
    else:
        df = pd.read_excel(archivo)

    TNumerico = list(df.select_dtypes(include=["int64", "float64"]).columns)
    TCategorical = list(df.select_dtypes(
            include=["object", "category", "bool", "string"]
        ).columns)
    #scaler = OneHotEncoder(sparse_output=False, drop='first')


    columnas = list(df.columns)
    varCategoricas = ["object", "string"]
    for columna in columnas:
        tipoValor = df[columna].dtype
        
        if pd.api.types.is_numeric_dtype(df[columna]):
            tipo_conceptual = "numeric"
        else:
            tipo_conceptual = "category"
        
        decision = valoresFaltantes(df, columna, tipoC=tipo_conceptual)
        
        if decision and decision.get("action") == "impute":
            df = aplicarMetodo(df, decision=decision)
            
        if str(tipoValor) in varCategoricas:
            df[columna] = df[columna].astype(str).str.strip().str.lower()

    preprocesador = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(), TCategorical),
            ('num', StandardScaler(), TNumerico)
    ])

    df_procesado = preprocesador.fit_transform(df)

    df_procesado = pd.DataFrame(
        df_procesado,
        columns=preprocesador.get_feature_names_out(),
        index=df.index
    ).copy()


    return df, df_procesado
    