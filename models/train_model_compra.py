import os
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
import joblib
from math import sqrt

# Cargar datos
df = pd.read_csv("data/housing_final.csv", sep=";")
df = df.drop(columns=["renta_es_proyeccion", "renta_neta_anual"])

numeric_features = ["anio", "renta_mensual_neta", "tipo_interes_hipoteca"]
categorical_features = ["ccaa", "provincia"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numeric_features),
        ("cat", OneHotEncoder(drop="first"), categorical_features),
    ]
)

os.makedirs("models", exist_ok=True)

X_compra = df[numeric_features + categorical_features]
y_compra = df["precio_compra_m2"]

model_compra = Pipeline(
    steps=[
        ("preprocess", preprocessor),
        ("regressor", LinearRegression())
    ]
)

Xc_train, Xc_test, yc_train, yc_test = train_test_split(
    X_compra, y_compra, test_size=0.2, random_state=42
)

model_compra.fit(Xc_train, yc_train)

yc_pred = model_compra.predict(Xc_test)
print("Compra – R2:", r2_score(yc_test, yc_pred))
print("Compra – RMSE:", sqrt(mean_squared_error(yc_test, yc_pred)))

cv_compra = cross_val_score(model_compra, X_compra, y_compra, cv=5, scoring="r2")
print("Compra – R2 CV:", cv_compra.mean(), "±", cv_compra.std())

joblib.dump(model_compra, "models/model_compra.pkl")
