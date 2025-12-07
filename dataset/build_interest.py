from pathlib import Path
import pandas as pd

RAW_INTEREST = Path("Dataset/tipo_interes_hipotecas.csv")               # tu CSV del INE
OUT_INTEREST = Path("Dataset/tipo_interes_hipotecas_final.csv")


def to_float_percent(x):
    # '3,16' -> 3.16
    s = str(x).strip().replace(".", "").replace(",", ".")
    # en este caso no hay miles, pero por si acaso
    return float(s)


def main():
    # sep=None = que pandas detecte ; o , automáticamente
    df = pd.read_csv(RAW_INTEREST, sep=None, engine="python", encoding="utf-8-sig")

    # Ajusta estos nombres si tus columnas son un pelín distintas
    df = df.rename(columns={
        "Periodo": "periodo",
        "Total": "tipo"
    })

    # Convertimos a float
    df["tipo"] = df["tipo"].apply(to_float_percent)

    # Año = primeros 4 caracteres del periodo (2025M09 -> 2025)
    df["anio"] = df["periodo"].astype(str).str.slice(0, 4).astype(int)

    # Media anual del tipo de interés
    df_year = (
        df.groupby("anio", as_index=False)["tipo"]
          .mean()
          .rename(columns={"tipo": "tipo_interes_hipoteca"})
    )

    OUT_INTEREST.parent.mkdir(parents=True, exist_ok=True)
    df_year.to_csv(OUT_INTEREST, index=False, float_format="%.4f", encoding="utf-8-sig")

    print(f"Serie anual generada: {OUT_INTEREST}")
    print(df_year.head())


if __name__ == "__main__":
    main()
