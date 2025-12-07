import pandas as pd

tablas = pd.read_html("venta_html.html", encoding="utf-8")

print(f"Se han encontrado {len(tablas)} tablas")
for i, t in enumerate(tablas):
    print(f"\nTabla {i}: columnas = {list(t.columns)}")

df_venta_raw = tablas[0]
print("\nPrimeras filas de la tabla 0:")
print(df_venta_raw.head())
