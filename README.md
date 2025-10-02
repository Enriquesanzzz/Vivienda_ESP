# 🏠 Vivienda en España — Dashboard interactivo

**Autor:** Enrique Sanz Tur  
**Asignatura:** Desarrollo de Aplicaciones para la Visualización de Datos (DAVD, 2025–26)

## 🎯 Objetivo
Desarrollar una aplicación web interactiva para **explorar la evolución de los precios de vivienda en España**, tanto de **compra** como de **alquiler**, a nivel **provincial** y **anual**. La app incluirá un **simulador sencillo de hipoteca** y una métrica de **esfuerzo de acceso a la vivienda** (relación entre cuota/alquiler e ingresos medios).

## 👥 Usuarios objetivo
- Ciudadanía que compara accesibilidad de la vivienda entre provincias.
- Estudiantes y docentes que analizan tendencias económicas y sociales.
- Periodistas/analistas que necesitan visualizaciones claras y exportables.

## 📊 Contenidos previstos de la aplicación
1. **Vista general**: evolución media nacional (€/m² compra y alquiler).
2. **Mapa provincial**: coropletas por provincia y año.
3. **Comparativa compra vs alquiler**: selección de provincias y líneas temporales.
4. **Accesibilidad**: simulador de hipoteca y esfuerzo estimado (hipoteca/ingresos y alquiler/ingresos).

## 🗂️ Datos (plan)
Fuentes públicas y abiertas:
- **INE** — Índices/series de precios de vivienda (compra y alquiler).
- **MTMAU** (Vivienda) — Estadísticas de transacciones inmobiliarias (complementarias).
- **INE** — Ingresos medios por provincia.
- **Informes de portales** (Idealista/Fotocasa) — Referencia/validación (opcional).
  
**Esquema mínimo del dataset** (`housing_es.csv`):
provincia, ccaa, anio, precio_compra_m2, precio_alquiler_m2, ingreso_mensual_medio, tipo_interes_anual

## 🛠️ Stack técnico previsto
- **Python** + `pandas` (preparación de datos)
- **Altair / Plotly** (gráficos)
- **Streamlit** (aplicación web)
- **Streamlit Cloud / Render** (despliegue público)

## 🧭 Plan de trabajo (iterativo)
- **Semana 1**: CSV mínimo (3–5 provincias, 1–2 años) + estructura del repositorio.
- **Semana 2**: Vista general y comparativa de provincias.
- **Semana 3**: Mapa provincial y simulador de hipoteca.
- **Semana 4**: Pulido visual, despliegue y demo.




