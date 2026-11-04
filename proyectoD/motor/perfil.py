import pandas as pd

def infoDataSet(df: pd.DataFrame):
  TNumerico = df.select_dtypes(include="number")
  TCategorical = df.select_dtypes(
        include=["object", "category", "bool", "string"]
    )

  perfilSet =  {
      "filas": len(df),
      "columnas": len(df.columns),
      "duplicados": int(df.duplicated().sum()),

      "numerico_resumen": TNumerico.describe().to_dict(),
      "categorico_resumen": TCategorical.describe().to_dict(),

      "dtypes": df.dtypes.astype(str).to_dict(),
      
  }

  return perfilSet


