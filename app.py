import json
import re
import unicodedata

import pandas as pd
import joblib
import numpy as np

from dash import Dash, dcc, html, Input, Output
import plotly.express as px

# --------------------------------------------------
# 1. CARGA DE DATOS Y MODELOS
# --------------------------------------------------

df = pd.read_csv("data/housing_final.csv", sep=";")
df = df.drop(columns=["renta_es_proyeccion", "renta_neta_anual"])

df["ccaa_mapa"] = df["ccaa"]

with open("data/spain_provinces.geojson", encoding="utf-8-sig") as f:
    geojson_provincias = json.load(f)

print("GeoJSON cargado. Nº de features:", len(geojson_provincias["features"]))

# --------------------------------------------------
# Función de normalización
# --------------------------------------------------
def normalize(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    # quitar tildes
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    # minúsculas + quitar espacios, guiones, etc.
    s = re.sub(r"[^a-z0-9]", "", s.lower())
    return s

# --------------------------------------------------
# 1) Tabla de provincias del GEOJSON
# --------------------------------------------------
geo_prov_norm_to_texto = {}
geo_prov_texto_to_ccaa = {}

for feat in geojson_provincias["features"]:
    nombre_prov = feat["properties"]["Texto"]          # p.ej. "La Coruña"
    nombre_ccaa = feat["properties"]["CCAA"]           # p.ej. "Galicia"
    key_norm = normalize(nombre_prov)

    geo_prov_norm_to_texto[key_norm] = nombre_prov
    geo_prov_texto_to_ccaa[nombre_prov] = nombre_ccaa

# --------------------------------------------------
# 2) Correcciones manuales para casos "raros"
#    (se aplican sobre tu CSV)
# --------------------------------------------------
def normalize_prov_df(nombre: str) -> str:
    base = normalize(nombre)
    fixes = {
        "acoruna": "lacoruna",                # "A Coruña" -> "La Coruña"
        "baleares": "islasbaleares",          # "Baleares" -> "Islas Baleares"
        "gipuzcoa": "guipuzcoa",              # "Gipúzcoa" -> "Guipúzcoa"
        "girona": "gerona",                   # "Girona" -> "Gerona"
        "lleida": "lleida",                   
        "ourense": "orense",                  # "Ourense" -> "Orense"
        "tenerife": "santacruzdetenerife",    # "Tenerife" -> "Santa Cruz de Tenerife"
    }
    return fixes.get(base, base)

# --------------------------------------------------
# 3) Crear columna de unión para el mapa
# --------------------------------------------------
def provincia_to_geo(nombre_prov: str):
    key = normalize_prov_df(nombre_prov)
    return geo_prov_norm_to_texto.get(key)

df["provincia_mapa"] = df["provincia"].apply(provincia_to_geo)

# (opcional) comprobar si alguna provincia no ha casado bien
provincias_fallidas = df[df["provincia_mapa"].isna()]["provincia"].unique()
print("Provincias sin match en el geojson:", provincias_fallidas)

# --------------------------------------------------
# 4) (Opcional) columna CCAA del geojson
# --------------------------------------------------
df["ccaa_geo"] = df["provincia_mapa"].map(geo_prov_texto_to_ccaa)

# Modelos entrenados
model_compra = joblib.load("models/model_compra.pkl")
model_alquiler = joblib.load("models/model_alquiler.pkl")

# Rango de sliders
default_ccaa = sorted(df["ccaa"].unique())[0]
default_provincias = sorted(df[df["ccaa"] == default_ccaa]["provincia"].unique())
default_provincia = default_provincias[0]

anio_min = int(df["anio"].min())
anio_max = int(df["anio"].max())

renta_min = int(df["renta_mensual_neta"].min())
renta_max = int(df["renta_mensual_neta"].max())
renta_med = int(df["renta_mensual_neta"].median())

interes_min = float(df["tipo_interes_hipoteca"].min())
interes_max = float(df["tipo_interes_hipoteca"].max())
interes_med = float(df["tipo_interes_hipoteca"].median())

PCT_ENTRADA = 0.20  # 20% de entrada

# --------------------------------------------------
# Estilos (solo cosmética)
# --------------------------------------------------
APP_STYLE = {"fontFamily": "Inter, Arial, sans-serif", "margin": "20px", "backgroundColor": "#f4f6f8"}
HEADER_STYLE = {"textAlign": "center", "color": "#0b2545", "marginBottom": "8px"}
LEFT_PANEL_STYLE = {
    "width": "32%",
    "padding": "18px",
    "backgroundColor": "#ffffff",
    "borderRadius": "12px",
    "boxShadow": "0 4px 14px rgba(11,37,69,0.08)",
    "alignSelf": "flex-start",  # no estirar al alto del contenedor
    "display": "flex",
    "flexDirection": "column",
    "justifyContent": "center",
    "maxHeight": "85vh",
    "overflowY": "auto",
}
RIGHT_PANEL_STYLE = {"width": "66%", "paddingLeft": "18px"}
CARD_STYLE = {"backgroundColor": "#ffffff", "padding": "16px", "borderRadius": "8px", "border": "1px solid #e6e9ee", "marginBottom": "20px"}
SMALL_HELP = {"fontSize": "0.85rem", "color": "#5f6b7a"}
CARD_TITLE_STYLE = {"fontSize": "1.05rem", "fontWeight": "600", "margin": "0 0 12px 0", "color": "#0b2545"}
CARD_SUBTITLE_STYLE = {"fontSize": "0.95rem", "fontWeight": "600", "color": "#0b2545", "margin": "8px 0"}
CARD_TEXT_STYLE = {"fontSize": "0.95rem", "color": "#263238", "lineHeight": "1.45"}
LABEL_STYLE = {"fontSize": "0.95rem", "fontWeight": "500", "color": "#243748", "marginBottom": "6px"}
SLIDER_LABEL_STYLE = {"marginTop": "8px", "fontSize": "0.9rem", "color": "#556770"}

def calcular_indicadores_provincias(
    anio,
    renta_mensual_individual,
    interes_hipoteca,
    tamano_vivienda_m2,
    n_salarios,
    pct_ahorro,
    plazo_anios,
):
    """
    Devuelve un DataFrame con una fila por provincia en el año dado,
    con columnas de esfuerzo de alquiler, esfuerzo de hipoteca y años
    para ahorrar la entrada, usando los valores de los sliders.
    """
    # 1) Filtrar año
    dff = df[df["anio"] == anio].copy()
    if dff.empty:
        # Por si el slider se va más allá: usamos el último año disponible
        ultimo_anio = df["anio"].max()
        dff = df[df["anio"] == ultimo_anio].copy()

    # 2) Ingresos y ahorro del hogar (constantes para todas las provincias)
    ingresos_mensuales_hogar = renta_mensual_individual * n_salarios
    ingresos_anuales_hogar = ingresos_mensuales_hogar * 12

    tasa_ahorro = pct_ahorro / 100.0
    ahorro_anual_posible = ingresos_anuales_hogar * tasa_ahorro

    # 3) Vivienda tipo (compra y alquiler) para cada provincia
    dff["precio_vivienda_tipo"] = dff["precio_compra_m2"] * tamano_vivienda_m2
    dff["alquiler_vivienda_tipo_mensual"] = (
        dff["precio_alquiler_m2"] * tamano_vivienda_m2
    )

    # 4) Esfuerzo alquiler (% ingreso)
    dff["esfuerzo_alquiler_pct"] = (
        dff["alquiler_vivienda_tipo_mensual"] / ingresos_mensuales_hogar * 100
    )

    # 5) Entrada necesaria y años para ahorrar
    dff["entrada_necesaria"] = dff["precio_vivienda_tipo"] * PCT_ENTRADA
    dff["ahorro_anual_posible"] = ahorro_anual_posible
    dff["anios_ahorrar_entrada"] = (
        dff["entrada_necesaria"] / dff["ahorro_anual_posible"]
    )

    # 6) Cuota de hipoteca
    principal = dff["precio_vivienda_tipo"] * (1 - PCT_ENTRADA)
    r = interes_hipoteca / 100.0 / 12.0
    n = plazo_anios * 12

    if r == 0:
        cuota_mensual = principal / n
    else:
        cuota_mensual = principal * (r * (1 + r) ** n) / ((1 + r) ** n - 1)

    dff["cuota_hipoteca_mensual"] = cuota_mensual
    dff["esfuerzo_cuota_pct"] = (
        dff["cuota_hipoteca_mensual"] / ingresos_mensuales_hogar * 100
    )

    return dff


# --------------------------------------------------
# 2. LAYOUT
# --------------------------------------------------

# --- parámetros del hogar / vivienda (por defecto) ---
DEFAULT_HOUSE_SIZE = 70
DEFAULT_N_SALARIES = 1.5
DEFAULT_SAVINGS_RATE = 20   # %
DEFAULT_MORTGAGE_YEARS = 25

app = Dash(__name__)
server = app.server

app.layout = html.Div(
    style=APP_STYLE,
    children=[
        html.H1("Mercado de vivienda en España", style=HEADER_STYLE),

        html.Div(
            style={"display": "flex", "gap": "40px", "alignItems": "flex-start"},
            children=[
                # --------- PANEL IZQUIERDO: INPUTS ---------
                html.Div(
                    className="card left-panel",
                    style=LEFT_PANEL_STYLE,
                    children=[
                        html.H3("Parámetros de entrada", className="card-title"),
                        html.Label("Comunidad Autónoma"),
                        dcc.Dropdown(
                            id="ccaa-dropdown",
                            options=[
                                {"label": c, "value": c}
                                for c in sorted(df["ccaa"].unique())
                            ],
                            value=default_ccaa,
                            clearable=False,
                        ),
                        html.Br(),

                        html.Label("Provincia"),
                        dcc.Dropdown(
                            id="provincia-dropdown",
                            options=[{"label": p, "value": p} for p in default_provincias],
                            value=default_provincia,
                            clearable=False,
                        ),
                        html.Br(),

                        html.Label("Año"),
                        dcc.Slider(
                            id="anio-slider",
                            min=anio_min,
                            max=anio_max,
                            step=1,
                            value=anio_max,
                            marks={a: str(a) for a in range(anio_min, anio_max + 1)},
                            tooltip={"placement": "bottom", "always_visible": True},
                        ),
                        html.Br(),

                        html.Label("Renta mensual neta (€)"),
                        dcc.Slider(
                            id="renta-slider",
                            min=renta_min,
                            max=renta_max,
                            step=50,
                            value=renta_med,
                            tooltip={"placement": "bottom", "always_visible": True},
                        ),
                        html.Div(id="renta-slider-label", className="slider-label", style=SLIDER_LABEL_STYLE),
                        html.Br(),

                        html.Label("Tipo interés hipoteca (%)"),
                        dcc.Slider(
                            id="interes-slider",
                            min=round(interes_min, 2),
                            max=round(interes_max, 2),
                            step=0.1,
                            value=round(interes_med, 2),
                            tooltip={"placement": "bottom", "always_visible": True},
                        ),
                        html.Div(id="interes-slider-label", className="slider-label", style=SLIDER_LABEL_STYLE),
                        html.Br(),

                        html.Label("Tamaño de la vivienda (m²)"),
                        dcc.Slider(
                            id="house-size-slider",
                            min=40,
                            max=120,
                            step=5,
                            value=DEFAULT_HOUSE_SIZE,
                            marks={m: str(m) for m in range(40, 125, 10)},
                            tooltip={"placement": "bottom", "always_visible": True},
                        ),
                        html.Br(),

                        html.Label("Nº salarios en el hogar"),
                        dcc.Slider(
                            id="n-salarios-slider",
                            min=1.0,
                            max=3.0,
                            step=0.5,
                            value=DEFAULT_N_SALARIES,
                            marks={x: str(x) for x in [1.0, 1.5, 2.0, 2.5, 3.0]},
                            tooltip={"placement": "bottom", "always_visible": True},
                        ),
                        html.Br(),

                        html.Label("Porcentaje del ingreso que podéis ahorrar (%)"),
                        dcc.Slider(
                            id="savings-rate-slider",
                            min=5,
                            max=40,
                            step=1,
                            value=DEFAULT_SAVINGS_RATE,
                            marks={p: f"{p}%" for p in range(5, 45, 5)},
                            tooltip={"placement": "bottom", "always_visible": True},
                        ),
                        html.Br(),

                        html.Label("Plazo de la hipoteca (años)"),
                        dcc.Slider(
                            id="mortgage-years-slider",
                            min=10,
                            max=35,
                            step=1,
                            value=DEFAULT_MORTGAGE_YEARS,
                            marks={y: str(y) for y in range(10, 40, 5)},
                            tooltip={"placement": "bottom", "always_visible": True},
                        ),
                        html.Br(),


                    ],
                ),

                # --------- PANEL DERECHO: PREDICCIONES + TABS ---------
                html.Div(
                    style=RIGHT_PANEL_STYLE,
                    children=[
                        html.Div(
                            style=CARD_STYLE,
                            children=[
                                        html.H4("Predicciones del modelo", className="card-title"),
                                                        html.Div(id="predicciones-output", className="card-text"),
                                            ],
                                        ),

                                                        html.Div(
                                                            className="card",
                                                            style=CARD_STYLE,
                                                            children=[
                                                dcc.Tabs(
                                                    id="tabs-graficos",
                                                    value="tab-evolucion",
                                                    children=[
                                dcc.Tab(
                                    label="Evolución provincia",
                                    value="tab-evolucion",
                                    children=[
                                        html.Div(
                                            [
                                                html.Label(
                                                    "Horizonte de predicción (años)"
                                                ),
                                                dcc.Slider(
                                                    id="horizonte-slider",
                                                    min=0,
                                                    max=10,
                                                    step=1,
                                                    value=0,
                                                    marks={
                                                        i: str(i) for i in range(0, 11)
                                                    },
                                                ),
                                                html.Small(
                                                    "0 = solo datos históricos · 1–10 = años adicionales desde el último año",
                                                    style=SMALL_HELP,
                                                ),
                                                html.Br(),
                                                html.Br(),
                                                dcc.Graph(
                                                    id="evolucion-compra-graph",
                                                    style={"height": "300px"},
                                                    config={"displayModeBar": False},
                                                ),
                                                dcc.Graph(
                                                    id="evolucion-alquiler-graph",
                                                    style={"height": "300px"},
                                                    config={"displayModeBar": False},
                                                ),
                                            ]
                                        )
                                    ],
                                ),
                                dcc.Tab(
                                    label="Mapa por provincias",
                                    value="tab-mapa",
                                    children=[
                                        html.Br(),
                                        dcc.Dropdown(
                                            id="mapa-variable",
                                            options=[
                                                {
                                                    "label": "Esfuerzo mensual en alquiler (% ingreso)",
                                                    "value": "esfuerzo_alquiler_pct",
                                                },
                                                {
                                                    "label": "Esfuerzo mensual en hipoteca (% ingreso)",
                                                    "value": "esfuerzo_cuota_pct",
                                                },
                                                {
                                                    "label": "Años para ahorrar la entrada",
                                                    "value": "anios_ahorrar_entrada",
                                                },
                                            ],
                                            value="esfuerzo_cuota_pct",
                                            clearable=False,
                                            style={"width": "60%", "marginBottom": "10px"},
                                        ),
                                        dcc.Graph(
                                            id="mapa-ccaa",
                                            style={"height": "500px"},
                                        ),
                                    ],
                                ),
                                dcc.Tab(
                                    label="Ranking provincias",
                                    value="tab-ranking",
                                    children=[
                                        dcc.Graph(
                                            id="ranking-prov-graph",
                                            style={"height": "500px"},
                                        )
                                    ],
                                ),
                                    ],
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        ),
    ],
)

# --------------------------------------------------
# 3. CALLBACKS
# --------------------------------------------------

def proyectar_serie_ultimos_anios(anios, valores, horizonte, ventana=5):
    """
    Construye una serie con:
    - tramo histórico ("Histórico")
    - tramo proyectado ("Predicción") hasta `horizonte` años vista

    La proyección usa el crecimiento medio anual (CAGR) calculado
    sobre los últimos `ventana` años disponibles de la serie.
    """
    anios = np.array(anios)
    valores = np.array(valores, dtype=float)

    # Solo histórico si no hay horizonte
    if horizonte <= 0 or len(anios) == 0:
        return pd.DataFrame(
            {"anio": anios, "valor": valores, "tipo": "Histórico"}
        )

    # Ventana de últimos años (máx. 'ventana' puntos)
    n = min(ventana, len(anios))
    anios_win = anios[-n:]
    vals_win = valores[-n:]

    first = vals_win[0]
    last = vals_win[-1]

    # CAGR sobre la ventana [primer año, último año]
    if n > 1 and first > 0:
        g = (last / first) ** (1 / (n - 1)) - 1
    else:
        g = 0.0  # sin variación si no hay info suficiente

    # Años futuros
    ultimo_anio = anios.max()
    futuros_anios = np.arange(ultimo_anio + 1, ultimo_anio + 1 + horizonte)

    # Valores futuros: crecimiento porcentual acumulado
    futuros_vals = [last * (1 + g) ** k for k in range(1, horizonte + 1)]
    futuros_vals = np.maximum(futuros_vals, 0)  # nunca negativos

    df_hist = pd.DataFrame(
        {"anio": anios, "valor": valores, "tipo": "Histórico"}
    )
    df_pred = pd.DataFrame(
        {"anio": futuros_anios, "valor": futuros_vals, "tipo": "Predicción"}
    )

    return pd.concat([df_hist, df_pred], ignore_index=True)


# Provincias por CCAA
@app.callback(
    Output("provincia-dropdown", "options"),
    Output("provincia-dropdown", "value"),
    Input("ccaa-dropdown", "value"),
)
def actualizar_provincias(ccaa):
    dff = df[df["ccaa"] == ccaa]
    provincias = sorted(dff["provincia"].unique())
    options = [{"label": p, "value": p} for p in provincias]
    value = provincias[0] if provincias else None
    return options, value


# Etiquetas sliders
@app.callback(
    Output("renta-slider-label", "children"),
    Output("interes-slider-label", "children"),
    Input("renta-slider", "value"),
    Input("interes-slider", "value"),
)
def actualizar_labels(renta, interes):
    txt_renta = f"Renta seleccionada: {renta:.0f} € / mes"
    txt_interes = f"Tipo de interés seleccionado: {interes:.2f} %"
    return txt_renta, txt_interes


# Predicciones
def cuota_mensual(principal, interes_anual, plazo_anios):
    r = interes_anual / 100 / 12
    n = plazo_anios * 12
    if r == 0:
        return principal / n
    return principal * (r * (1 + r) ** n) / ((1 + r) ** n - 1)


@app.callback(
    Output("predicciones-output", "children"),
    Input("ccaa-dropdown", "value"),
    Input("provincia-dropdown", "value"),
    Input("anio-slider", "value"),
    Input("renta-slider", "value"),
    Input("interes-slider", "value"),
    Input("house-size-slider", "value"),
    Input("n-salarios-slider", "value"),
    Input("savings-rate-slider", "value"),
    Input("mortgage-years-slider", "value"),
)
def actualizar_predicciones(
    ccaa,
    provincia,
    anio,
    renta_mensual_individual,
    interes_hipoteca,
    house_size_m2,
    n_salarios,
    savings_rate_pct,
    mortgage_years,
):
    if ccaa is None or provincia is None:
        return html.P("Selecciona una comunidad autónoma y una provincia.")

    # 1) Predicción de precios por m²
    row = pd.DataFrame(
        [
            {
                "anio": anio,
                "renta_mensual_neta": renta_mensual_individual,
                "tipo_interes_hipoteca": interes_hipoteca,
                "ccaa": ccaa,
                "provincia": provincia,
            }
        ]
    )

    pred_compra_m2 = model_compra.predict(row)[0]
    pred_alquiler_m2 = model_alquiler.predict(row)[0]

    # 2) Escenario del hogar según inputs del usuario
    ingresos_hogar_mensuales = renta_mensual_individual * n_salarios
    ingresos_hogar_anuales = ingresos_hogar_mensuales * 12

    precio_vivienda = pred_compra_m2 * house_size_m2
    alquiler_vivienda_mensual = pred_alquiler_m2 * house_size_m2

    # --- parámetros "financieros" fijos ---
    pct_entrada = 0.20     # 20% del precio
    pct_hipoteca = 0.80    # 80% financiado
    ahorro_rate = savings_rate_pct / 100.0

    # 3) Indicador 1: esfuerzo de alquiler
    esfuerzo_alquiler = (
        alquiler_vivienda_mensual / ingresos_hogar_mensuales * 100
        if ingresos_hogar_mensuales > 0
        else None
    )

    # 4) Indicador 2: años para ahorrar la entrada
    entrada = precio_vivienda * pct_entrada
    ahorro_anual_posible = ingresos_hogar_anuales * ahorro_rate
    anios_entrada = (
        entrada / ahorro_anual_posible if ahorro_anual_posible > 0 else None
    )

    # 5) Indicador 3: esfuerzo de cuota hipotecaria
    principal = precio_vivienda * pct_hipoteca
    cuota_mensual_hipoteca = cuota_mensual(
        principal, interes_hipoteca, mortgage_years
    )
    esfuerzo_cuota = (
        cuota_mensual_hipoteca / ingresos_hogar_mensuales * 100
        if ingresos_hogar_mensuales > 0
        else None
    )

    # 6) Construimos el bloque de texto
    return html.Div(
        [
            html.P(f"Precio de compra estimado: {pred_compra_m2:,.0f} €/m²", style=CARD_TEXT_STYLE),
            html.P(f"Precio de alquiler estimado: {pred_alquiler_m2:,.2f} €/m²", style=CARD_TEXT_STYLE),
            html.Hr(),
            html.P(
                f"Para una vivienda de {house_size_m2:.0f} m² y un hogar con {n_salarios:.1f} salarios:",
                style=CARD_SUBTITLE_STYLE,
            ),
            html.Ul(
                [
                    html.Li(
                        f"Alquiler aproximado: {alquiler_vivienda_mensual:,.0f} € / mes "
                        if esfuerzo_alquiler is not None
                        else "Alquiler: no se puede calcular (ingresos 0)."
                    ),
                    html.Li(
                        f"Años necesarios para ahorrar la entrada (20%): {anios_entrada:,.1f} años"
                        if anios_entrada is not None
                        else "Años para la entrada: no se puede calcular (ahorro 0)."
                    ),
                    html.Li(
                        f"Cuota hipotecaria estimada ({mortgage_years:.0f} años, {interes_hipoteca:.2f}%): "
                        f"{cuota_mensual_hipoteca:,.0f} € / mes "
                        if esfuerzo_cuota is not None
                        else "Cuota hipotecaria: no se puede calcular (ingresos 0)."
                    ),
                ],
                style={"marginLeft": "18px"},
            ),
            html.Hr(),
            html.Small(
                "Nota: los cálculos se basan en los precios por m² predichos por el modelo "
                "y en las suposiciones introducidas por el usuario sobre tamaño de vivienda, "
                "salarios, ahorro y plazo de la hipoteca.",
                style=SMALL_HELP,
            ),
        ]
    )



# Evolución histórica + predicción provincia (dos gráficas)
@app.callback(
    Output("evolucion-compra-graph", "figure"),
    Output("evolucion-alquiler-graph", "figure"),
    Input("provincia-dropdown", "value"),
    Input("horizonte-slider", "value"),
)
def update_evolucion_graphs(provincia, horizonte):
    # Por si acaso, si no hay provincia seleccionada usamos la primera del df
    if provincia is None:
        provincia = df["provincia"].iloc[0]

    # Datos de la provincia
    df_prov = df[df["provincia"] == provincia].sort_values("anio")

    # Si por lo que sea no hay datos, devolvemos figuras vacías
    if df_prov.empty:
        fig_vacio = px.line(title="Sin datos para esta provincia")
        return fig_vacio, fig_vacio

    # Vector de años
    anios = df_prov["anio"].values

    # ===== 1) Serie de COMPRA: histórico + predicción usando SOLO los últimos 5 años =====
    serie_compra = proyectar_serie_ultimos_anios(
        anios=anios,
        valores=df_prov["precio_compra_m2"].values,
        horizonte=horizonte,
        ventana=5,   # <-- usamos los últimos 5 años para calcular el crecimiento
    )

    fig_compra = px.line(
        serie_compra,
        x="anio",
        y="valor",
        color="tipo",
        line_dash="tipo",
        markers=True,
        labels={"anio": "Año", "valor": "€/m²", "tipo": ""},
        title=f"Evolución del precio de compra en {provincia}",
    )
    fig_compra.update_layout(
        margin=dict(l=40, r=10, t=60, b=40),
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.2),
    )

    # ===== 2) Serie de ALQUILER: histórico + predicción usando SOLO los últimos 5 años =====
    serie_alquiler = proyectar_serie_ultimos_anios(
        anios=anios,
        valores=df_prov["precio_alquiler_m2"].values,
        horizonte=horizonte,
        ventana=5,
    )

    fig_alquiler = px.line(
        serie_alquiler,
        x="anio",
        y="valor",
        color="tipo",
        line_dash="tipo",
        markers=True,
        labels={"anio": "Año", "valor": "€/m²", "tipo": ""},
        title=f"Evolución del precio de alquiler en {provincia}",
    )
    fig_alquiler.update_layout(
        margin=dict(l=40, r=10, t=60, b=40),
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.2),
    )

    return fig_compra, fig_alquiler


# --------------------------------------------------
# Mapa por provincias
# --------------------------------------------------
@app.callback(
    Output("mapa-ccaa", "figure"),
    Input("anio-slider", "value"),
    Input("mapa-variable", "value"),
    Input("renta-slider", "value"),
    Input("interes-slider", "value"),
    Input("house-size-slider", "value"),
    Input("n-salarios-slider", "value"),
    Input("savings-rate-slider", "value"),
    Input("mortgage-years-slider", "value"),
)
def actualizar_mapa_esfuerzo(
    anio,
    variable,
    renta,
    interes,
    tamano_vivienda,
    n_salarios,
    pct_ahorro,
    plazo_anios,
):
    # 1) Recalcular indicadores para TODAS las provincias
    dff = calcular_indicadores_provincias(
        anio=anio,
        renta_mensual_individual=renta,
        interes_hipoteca=interes,
        tamano_vivienda_m2=tamano_vivienda,
        n_salarios=n_salarios,
        pct_ahorro=pct_ahorro,
        plazo_anios=plazo_anios,
    )

    # 2) Nos quedamos solo con la columna que queremos pintar
    df_map = dff[["provincia_mapa", variable]].copy()

    # Título de la barra de color
    colorbar_titles = {
        "esfuerzo_alquiler_pct": "Esfuerzo alquiler (%)",
        "esfuerzo_cuota_pct": "Esfuerzo hipoteca (%)",
        "anios_ahorrar_entrada": "Años para entrada",
    }
    colorbar_title = colorbar_titles.get(variable, "")

    # --- Rango fijo para que los sliders cambien realmente el color del mapa ---
    if variable in ("esfuerzo_alquiler_pct", "esfuerzo_cuota_pct"):
        vmin, vmax = 0, 60     # 0–60 % del ingreso del hogar
    elif variable == "anios_ahorrar_entrada":
        vmin, vmax = 0, 15     # 0–15 años para ahorrar la entrada
    else:
        vmin, vmax = None, None

    fig = px.choropleth(
        df_map,
        geojson=geojson_provincias,
        locations="provincia_mapa",
        featureidkey="properties.Texto",
        color=variable,
        color_continuous_scale="RdYlGn_r",  # verde = menos esfuerzo, rojo = más
        range_color=(vmin, vmax) if vmin is not None else None,
        labels={
            "esfuerzo_alquiler_pct": "Esfuerzo alquiler (%)",
            "esfuerzo_cuota_pct": "Esfuerzo hipoteca (%)",
            "anios_ahorrar_entrada": "Años para entrada",
        },
    )

    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        height=500,
        coloraxis_colorbar=dict(title=colorbar_title),
    )

    return fig


# Ranking provincias
@app.callback(
    Output("ranking-prov-graph", "figure"),
    Input("ccaa-dropdown", "value"),
    Input("anio-slider", "value"),
    Input("mapa-variable", "value"),
    Input("renta-slider", "value"),
    Input("interes-slider", "value"),
    Input("house-size-slider", "value"),
    Input("n-salarios-slider", "value"),
    Input("savings-rate-slider", "value"),
    Input("mortgage-years-slider", "value"),
)
def actualizar_ranking(
    ccaa,
    anio,
    variable,
    renta,
    interes,
    tamano_vivienda,
    n_salarios,
    pct_ahorro,
    plazo_anios,
):
    if ccaa is None:
        return px.bar(title="Selecciona una CCAA para ver el ranking.")

    # Recalculamos los indicadores para TODAS las provincias
    dff = calcular_indicadores_provincias(
        anio=anio,
        renta_mensual_individual=renta,
        interes_hipoteca=interes,
        tamano_vivienda_m2=tamano_vivienda,
        n_salarios=n_salarios,
        pct_ahorro=pct_ahorro,
        plazo_anios=plazo_anios,
    )

    # Nos quedamos solo con la CCAA seleccionada
    dff = dff[dff["ccaa"] == ccaa].copy()
    if dff.empty:
        return px.bar(title="Sin datos para esa combinación.")

    # Ordenamos por el indicador elegido (de más esfuerzo a menos)
    dff = dff.sort_values(variable, ascending=False)

    # Etiquetas bonitas según el indicador
    y_labels = {
        "esfuerzo_alquiler_pct": "Esfuerzo alquiler (%)",
        "esfuerzo_cuota_pct": "Esfuerzo hipoteca (%)",
        "anios_ahorrar_entrada": "Años para entrada",
    }
    titulo_indicador = y_labels.get(variable, variable)

    fig = px.bar(
        dff,
        x="provincia",
        y=variable,
        labels={"provincia": "Provincia", variable: titulo_indicador},
        title=f"{titulo_indicador} por provincia en {ccaa} – {anio}",
    )

    fig.update_layout(
        xaxis_tickangle=-45,
        height=500,
        margin=dict(l=40, r=20, t=60, b=80),
    )

    return fig



# --------------------------------------------------
# MAIN
# --------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True, port=8053)
