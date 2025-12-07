import time
import requests
import pandas as pd
from pathlib import Path

# --------- CONFIGURACIÓN BÁSICA ---------

# Aquí pones las provincias que quieres usar como piloto
# Rellena tú las URLs de histórico de venta y alquiler de cada provincia.
PROVINCES = [
    {
        "provincia": "Madrid",
        "ccaa": "Comunidad de Madrid",
        "venta_url": "https://www.idealista.com/sala-de-prensa/informes-precio-vivienda/venta/madrid-comunidad/madrid-provincia/madrid/historico/",
        "alquiler_url": "https://www.idealista.com/sala-de-prensa/informes-precio-vivienda/alquiler/madrid-comunidad/madrid-provincia/madrid/historico/",
    },
    # Añade aquí más provincias copiando este bloque
]

# Carpeta donde se guardará el CSV final
OUTPUT_DIR = Path("data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_CSV = OUTPUT_DIR / "housing_es_from_idealista.csv"

# Pausa entre peticiones (en segundos) para ir con calma
REQUEST_SLEEP = 5


# --------- FUNCIONES AUXILIARES ---------

SPANISH_MONTHS = {
    "enero": 1,
    "febrero": 2,
    "marzo": 3,
    "abril": 4,
    "mayo": 5,
    "junio": 6,
    "julio": 7,
    "agosto": 8,
    "septiembre": 9,
    "setiembre": 9,  # por si acaso
    "octubre": 10,
    "noviembre": 11,
    "diciembre": 12,
}

def parse_mes_column(mes_str: str) -> int:
    """
    Recibe algo tipo 'Octubre 2025' y devuelve el año (2025).
    También podríamos devolver (año, mes), pero para este flujo solo
    necesitamos el año para luego hacer la media anual.
    """
    mes_str = str(mes_str).strip().lower()
    # suele venir "octubre 2025" -> ["octubre", "2025"]
    partes = mes_str.split()
    if len(partes) != 2:
        raise ValueError(f"No puedo parsear el campo Mes: {mes_str}")
    mes_nombre, anio_str = partes
    if mes_nombre not in SPANISH_MONTHS:
        raise ValueError(f"Mes no reconocido: {mes_nombre}")
    anio = int(anio_str)
    return anio


def clean_precio(precio_str: str) -> float:
    """
    Convierte cadenas tipo '2.597 €/m2' a 2597.0
    """
    s = str(precio_str)
    # nos quedamos solo con números, coma y punto
    s = s.replace("€/m2", "").replace("€", "").replace(" ", "")
    # Idealista usa punto como separador de miles y coma como decimal
    s = s.replace(".", "").replace(",", ".")
    return float(s)


def fetch_price_history(url: str) -> pd.DataFrame:
    """
    Descarga la página de Idealista para una URL de histórico y devuelve
    un DataFrame con columnas al menos ['Mes', 'Precio m2'].
    """
    print(f"Descargando {url}")
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; ProyectoViviendaDAVD/1.0; +https://github.com/Enriquesanzzz)"
    }
    resp = requests.get(url, headers=headers, timeout=20)
    resp.raise_for_status()

    # Leemos todas las tablas de la página
    tablas = pd.read_html(resp.text)
    if not tablas:
        raise ValueError(f"No se han encontrado tablas en {url}")

    # Buscamos la tabla que tenga la columna 'Precio m2'
    for df in tablas:
        if "Precio m2" in df.columns:
            historial = df.copy()
            break
    else:
        raise ValueError(f"No se ha encontrado una tabla con 'Precio m2' en {url}")

    # Nos quedamos con las columnas que nos interesan
    if "Mes" not in historial.columns:
        raise ValueError(f"La tabla no tiene columna 'Mes' en {url}")

    historial = historial[["Mes", "Precio m2"]].dropna()
    historial["anio"] = historial["Mes"].apply(parse_mes_column)
    historial["precio_m2"] = historial["Precio m2"].apply(clean_precio)

    return historial[["anio", "precio_m2"]]


def aggregate_by_year(hist_df: pd.DataFrame) -> pd.DataFrame:
    """
    Recibe un dataframe con columnas ['anio', 'precio_m2']
    y devuelve media anual por año.
    """
    return (
        hist_df
        .groupby("anio", as_index=False)["precio_m2"]
        .mean()
        .rename(columns={"precio_m2": "precio_m2_anual"})
    )


# --------- PIPELINE PRINCIPAL ---------

def main():
    rows = []

    for entry in PROVINCES:
        provincia = entry["provincia"]
        ccaa = entry["ccaa"]

        # --- Venta ---
        venta_url = entry.get("venta_url")
        if venta_url:
            hist_venta = fetch_price_history(venta_url)
            venta_anual = aggregate_by_year(hist_venta)
        else:
            venta_anual = pd.DataFrame(columns=["anio", "precio_m2_anual"])

        time.sleep(REQUEST_SLEEP)

        # --- Alquiler ---
        alquiler_url = entry.get("alquiler_url")
        if alquiler_url:
            hist_alq = fetch_price_history(alquiler_url)
            alq_anual = aggregate_by_year(hist_alq)
        else:
            alq_anual = pd.DataFrame(columns=["anio", "precio_m2_anual"])

        time.sleep(REQUEST_SLEEP)

        # Unimos por año
        df_merged = pd.merge(
            venta_anual.rename(columns={"precio_m2_anual": "precio_compra_m2"}),
            alq_anual.rename(columns={"precio_m2_anual": "precio_alquiler_m2"}),
            on="anio",
            how="outer",
        ).sort_values("anio")

        # Añadimos provincia y CCAA
        df_merged["provincia"] = provincia
        df_merged["ccaa"] = ccaa

        rows.append(df_merged)

    if not rows:
        print("No se han generado filas. ¿Has rellenado PROVINCES?")
        return

        full_df = pd.concat(rows, ignore_index=True)
    full_df = full_df[["provincia", "ccaa", "anio", "precio_compra_m2", "precio_alquiler_m2"]]

    # --- aquí empieza la parte nueva ---
    if OUTPUT_CSV.exists():
        old_df = pd.read_csv(OUTPUT_CSV)
        combined = pd.concat([old_df, full_df], ignore_index=True)
        # Nos quedamos con la última versión para cada (provincia, anio)
        combined = combined.drop_duplicates(subset=["provincia", "anio"], keep="last")
    else:
        combined = full_df
    # --- aquí termina la parte nueva ---

    combined.to_csv(OUTPUT_CSV, index=False)
    print(f"CSV generado/actualizado en: {OUTPUT_CSV}")
    print(combined.tail())



if __name__ == "__main__":
    main()
