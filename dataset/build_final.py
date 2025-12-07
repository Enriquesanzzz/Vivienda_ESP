from pathlib import Path
import unicodedata
import pandas as pd

# --- RUTAS DE ENTRADA / SALIDA ---

PRECIOS_CSV   = Path("data/housing_precios_provincia.csv")
RENTA_CSV     = Path("data/renta_provincia_2015_2025.csv")
TIPO_INT_CSV  = Path("Dataset/tipo_interes_hipotecas_final.csv")

OUTPUT_CSV    = Path("data/housing_final.csv")


# --- FUNCIONES AUXILIARES ---

def normalize_name(s: str) -> str:
    s = str(s).strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return " ".join(s.split())


def standardize_geo_cols(df: pd.DataFrame, origen: str) -> pd.DataFrame:
    df.columns = [c.strip() for c in df.columns]
    lower_map = {c.lower(): c for c in df.columns}

    ccaa_candidates = ["ccaa", "comunidades", "comunidad", "comunidad_autonoma"]
    prov_candidates = ["provincia", "provincias", "prov"]

    ccaa_col = None
    prov_col = None

    for cand in ccaa_candidates:
        if cand in lower_map:
            ccaa_col = lower_map[cand]
            break

    for cand in prov_candidates:
        if cand in lower_map:
            prov_col = lower_map[cand]
            break

    if ccaa_col is None or prov_col is None:
        raise ValueError(
            f"No encuentro columnas de CCAA/provincia en {origen}. "
            f"Columnas disponibles: {list(df.columns)}"
        )

    if "ccaa" not in df.columns:
        df = df.rename(columns={ccaa_col: "ccaa"})
    if "provincia" not in df.columns:
        df = df.rename(columns={prov_col: "provincia"})

    return df


def main():
    # 1) Precios por provincia  (auto-detectar , o ;)
    df_pre = pd.read_csv(PRECIOS_CSV, sep=None, engine="python", encoding="utf-8-sig")
    df_pre = standardize_geo_cols(df_pre, "housing_precios_provincia.csv")

    # 2) Renta por provincia  (auto-detectar , o ;)
    df_renta = pd.read_csv(RENTA_CSV, sep=None, engine="python", encoding="utf-8-sig")
    df_renta = standardize_geo_cols(df_renta, "renta_provincia_2015_2025.csv")

    # 3) Tipos de interés anuales  (ya lo teníamos con sep=None)
    df_int = pd.read_csv(TIPO_INT_CSV, sep=None, engine="python", encoding="utf-8-sig")
    df_int = df_int.rename(columns={
        "anio": "anio",
        "tipo_interes_hipoteca": "tipo_interes_hipoteca"
    })
    df_int["anio"] = df_int["anio"].astype(int)
    df_int["tipo_interes_hipoteca"] = df_int["tipo_interes_hipoteca"].astype(float)

    # 4) Normalizar nombres y tipos
    for df in (df_pre, df_renta):
        df["ccaa_norm"] = df["ccaa"].apply(normalize_name)
        df["prov_norm"] = df["provincia"].apply(normalize_name)
        df["anio"] = df["anio"].astype(int)

    # 5) Merge precios + renta
    df_merge = df_pre.merge(
        df_renta[["ccaa_norm", "prov_norm", "anio",
                  "renta_neta_anual", "renta_mensual_neta", "is_projection"]],
        on=["ccaa_norm", "prov_norm", "anio"],
        how="left",
        suffixes=("", "_renta")
    )
    df_merge = df_merge.rename(columns={"is_projection": "renta_es_proyeccion"})

    # 6) Añadir tipos de interés
    df_full = df_merge.merge(
        df_int[["anio", "tipo_interes_hipoteca"]],
        on="anio",
        how="left",
    )

    # 7) Limpiar columnas auxiliares
    df_full = df_full.drop(columns=["ccaa_norm", "prov_norm"])

    # 8) Ordenar columnas
    cols_order = [
        "ccaa", "provincia", "anio",
        "precio_compra_m2", "precio_alquiler_m2",
        "renta_neta_anual", "renta_mensual_neta", "renta_es_proyeccion",
        "tipo_interes_hipoteca",
    ]
    cols_order = [c for c in cols_order if c in df_full.columns]
    df_full = df_full[cols_order]

    # 9) Guardar
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_full.to_csv(OUTPUT_CSV, index=False, float_format="%.4f", encoding="utf-8-sig")

    print(f"✅ Dataset final generado: {OUTPUT_CSV}")
    print(df_full.head(10))


if __name__ == "__main__":
    main()
