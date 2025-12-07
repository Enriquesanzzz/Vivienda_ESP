# 🏡 Vivienda en España — Dashboard interactivo

**Autor:** Enrique Sanz Tur  
**Asignatura:** Desarrollo de Aplicaciones para la Visualización de Datos (DAVD, 2025–26)  
**Profesor:** David Martín Cañás  

---

## 🎯 Objetivo del proyecto

Este proyecto desarrolla una **aplicación web interactiva** para explorar la evolución de los **precios de vivienda en España** (tanto de **compra** como de **alquiler**) y su impacto en la **capacidad de acceso a la vivienda** de los hogares.

La app permite:

- Analizar la evolución histórica del precio por m² en cada provincia.
- Obtener **predicciones** a corto plazo de compra y alquiler.
- Calcular un **esfuerzo mensual en hipoteca** (porcentaje de la renta del hogar dedicado a la cuota).
- Visualizar, mediante mapas y rankings, qué provincias presentan mayor o menor dificultad de acceso a la vivienda para un perfil de hogar dado.

La idea es que el dashboard pueda funcionar como una **herramienta sencilla de simulación** para usuarios interesados en comprar o alquilar vivienda, así como un ejemplo completo de integración de:
**obtención de datos + modelado + visualización interactiva + despliegue en producción**.

---

## 👥 Usuarios objetivo

- **Usuarios finales / público general**  
  Personas que quieren hacerse una idea rápida de:
  - Cómo han evolucionado los precios en su provincia.
  - Si, con sus ingresos y condiciones de hipoteca, el esfuerzo mensual es razonable o excesivo.
  - En qué provincias el esfuerzo es mayor o menor para un perfil de hogar dado.

- **Perfil académico (asignatura DAVD)**  
  El proyecto sirve también como demostración de:
  - Integración de datos de distintas fuentes.
  - Entrenamiento y uso de modelos con `scikit-learn`.
  - Construcción de dashboards interactivos con **Dash + Plotly**.
  - Despliegue en un servicio cloud (**Render**).

---

## 🌐 Demo en producción

La aplicación está desplegada en Render y se puede probar en:

👉 **https://vivienda-esp.onrender.com**

> Nota: el servicio está en el *plan gratuito* de Render.  
> Si lleva un rato sin usarse, la primera carga puede tardar unos segundos mientras la instancia “despierta”.

---

## 🧩 Funcionalidades principales de la app

### 1. Panel de parámetros de entrada (columna izquierda)

El usuario puede configurar:

- **Comunidad Autónoma**  
- **Provincia** (filtrada por la comunidad seleccionada)
- **Año de referencia** (dentro del rango disponible)
- **Renta mensual neta del hogar (€)**  
- **Tipo de interés de la hipoteca (%)**
- **Tamaño de la vivienda (m²)**
- **Número de salarios en el hogar** (por ejemplo 1, 1.5, 2…)
- **Porcentaje del ingreso que se puede ahorrar (%)**
- **Plazo de la hipoteca (años)**

Estos parámetros alimentan tanto las **predicciones del modelo** como los cálculos de esfuerzo hipotecario.

---

### 2. Módulo de “Predicciones del modelo”

En la parte superior central se muestran, para la provincia seleccionada:

- **Precio de compra estimado (€/m²)**
- **Precio de alquiler estimado (€/m²)**

Y, para una vivienda tipo de 70 m² y 1,5 salarios:

- **Alquiler aproximado (€/mes)**
- **Años necesarios para ahorrar la entrada (20%)** en función del ahorro mensual introducido.
- **Cuota hipotecaria estimada** (según tipo de interés y plazo seleccionados).

Debajo se incluye una nota explicativa indicando que:
- Los precios se basan en predicciones del modelo.
- El esfuerzo y la cuota son cálculos aproximados en base a las hipótesis del usuario.

---

### 3. Pestañas de visualización

#### 🟦 Pestaña 1: *Evolución provincia*

Muestra, para la provincia seleccionada:

- **Serie histórica** del precio de compra (€/m²).
- **Serie histórica** del precio de alquiler (€/m²).
- En el futuro, la idea es añadir puntos/predicciones extrapoladas a partir del último año disponible.

Incluye un control de:

- **Horizonte de predicción (años)**  
  - 0 = solo datos históricos.  
  - 1–10 = añadir años adicionales a partir del último dato disponible.

#### 🟩 Pestaña 2: *Mapa por provincias*

- Mapa coroplético de España por provincias, coloreando cada provincia según el **esfuerzo mensual en hipoteca (% de la renta)** para el perfil fijado en la barra lateral.
- Permite ver de forma global qué zonas presentan mayor dificultad relativa de acceso a la compra, comparando provincias entre sí.
- El mapa se actualiza cuando el usuario modifica:
  - Renta del hogar
  - Tamaño de la vivienda
  - Tipo de interés
  - Plazo de la hipoteca
  - Número de salarios / porcentaje de ahorro

#### 🟨 Pestaña 3: *Ranking provincias*

- Tabla ordenada de provincias según el **esfuerzo hipotecario** calculado para el perfil seleccionado.
- Permite identificar rápidamente:
  - Las provincias con mayor esfuerzo (más “caras” para el usuario tipo).
  - Las provincias con menor esfuerzo (más accesibles).

---

## 🧮 Modelado y datos (resumen)

> Nota: aquí se describe el enfoque general, no el detalle de todas las transformaciones.

- **Tipo de modelo:**  
  Para cada caso (compra y alquiler) se entrena un modelo de **regresión lineal** a partir de:
  - Características socioeconómicas (rentas, salarios medios, etc.).
  - Información geográfica (provincia, comunidad).
  - Años (para capturar la evolución temporal).

- **Preprocesado:**  
  Se utiliza un `ColumnTransformer` con:
  - `OneHotEncoder` para variables categóricas (comunidad, provincia).
  - Transformaciones numéricas básicas para las variables continuas.

- **Entrenamiento y persistencia:**  
  - Los scripts `train_model_compra.py`, `train_model_alquiler.py` y `train_models.py` generan y guardan los modelos en la carpeta `models/` como ficheros `.joblib`.
  - La aplicación principal (`app.py`) carga estos modelos y los utiliza en cada callback de Dash para producir predicciones en tiempo real.

- **Estructura de datos:**
  - Carpeta `dataset/` o `data/` con los ficheros de datos originales / procesados.
  - Carpeta `notebooks/` con notebooks usados para exploración y preparación de los datos (EDA, pruebas de modelo, etc.).

---

## 🏗️ Estructura del repositorio

A alto nivel:

```text
Vivienda_ESP/
├── app.py               # Aplicación Dash principal
├── Procfile             # Comando de arranque para Gunicorn (Render/Heroku-style)
├── render.yaml          # Configuración del servicio en Render
├── requirements.txt     # Dependencias del proyecto
├── models/              # Modelos entrenados (.joblib)
├── data/                # Datos limpios usados por la app
├── dataset/             # Datos brutos / intermedios
├── notebooks/           # Notebooks de exploración y modelado
├── assets/              # Estilos CSS personalizados y recursos estáticos
└── README.md            # Este documento
