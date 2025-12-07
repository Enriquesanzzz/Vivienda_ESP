from pathlib import Path
import numpy as np
import pandas as pd
import re

RENTA_RAW = Path("Dataset/renta_ccaa.csv")           # tu fichero limpio
RENTA_OUT = Path("data/renta_provincia_2015_2025.csv")

MAX_YEAR_TARGET = 2025
N_YEARS_GROWTH = 3   # años recientes para la tasa media


def to_float_euros(x):
    """
    Convierte '11,543' o '11.543' o '11 543' -> 11543.0

    Como ya no hay nulls y todas son 5 cifras, simplemente
    nos quedamos con los dígitos.
    """
    s = str(x)

    # extraemos solo dígitos
    digits = re.sub(r"[^\d]", "", s)
    if digits == "":
        return np.nan

    return float(digits)


def load_renta(path: Path) -> pd.DataFrame:
    """
    Espera columnas:
      - 'Comunidades'
      - 'Provincias'
      - 'Periodo'
      - 'Total'
    """
    # sep=None deja que pandas detecte si el separador es ; o ,
    df = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")

    df = df.rename(columns={
        "Comunidades": "ccaa",
        "Provincias": "provincia",
        "Periodo": "anio",
        "Total": "renta_neta_anual",
    })

    df = df[["ccaa", "provincia", "anio", "renta_neta_anual"]].copy()

    df["anio"] = df["anio"].astype(int)
    df["renta_neta_anual"] = df["renta_neta_anual"].apply(to_float_euros)

    # por si quedara alguna fila rara
    df = df.dropna(subset=["renta_neta_anual"])

    return df


def project_renta(df_renta: pd.DataFrame) -> pd.DataFrame:
    """
    Proyecta renta_neta_anual a 2024 y 2025 por provincia
    usando la tasa media de crecimiento de los últimos N_YEARS_GROWTH años.
    """
    filas = []

    for (ccaa, provincia), grp in df_renta.groupby(["ccaa", "provincia"]):
        g = grp.sort_values("anio").reset_index(drop=True)

        # crecimientos interanuales
        g["growth"] = g["renta_neta_anual"].pct_change()

        growth_recent = g["growth"].dropna().tail(N_YEARS_GROWTH)
        if len(growth_recent) == 0:
            g_med = 0.0
        else:
            g_med = growth_recent.mean()

        last_year = g["anio"].max()
        last_value = g.loc[g["anio"].idxmax(), "renta_neta_anual"]

        projections = []
        current_year = last_year
        current_value = last_value

        while current_year < MAX_YEAR_TARGET:
            next_year = current_year + 1
            current_value = current_value * (1 + g_med)
            projections.append({
                "ccaa": ccaa,
                "provincia": provincia,
                "anio": next_year,
                "renta_neta_anual": current_value,
                "is_projection": True,
            })
            current_year = next_year

        g["is_projection"] = False
        filas.append(g[["ccaa", "provincia", "anio",
                        "renta_neta_anual", "is_projection"]])
        if projections:
            filas.append(pd.DataFrame(projections))

    df_full = pd.concat(filas, ignore_index=True)
    df_full = df_full.sort_values(["ccaa", "provincia", "anio"]).reset_index(drop=True)

    # renta mensual
    df_full["renta_mensual_neta"] = df_full["renta_neta_anual"] / 12.0

    return df_full


def main():
    df_renta_raw = load_renta(RENTA_RAW)
    df_renta_full = project_renta(df_renta_raw)

    RENTA_OUT.parent.mkdir(parents=True, exist_ok=True)
    df_renta_full.to_csv(RENTA_OUT, index=False,
                         float_format="%.4f", encoding="utf-8-sig")

    print(f"Fichero intermedio generado: {RENTA_OUT}")
    print(df_renta_full.head(10))


if __name__ == "__main__":
    main()
