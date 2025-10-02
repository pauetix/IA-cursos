# %%
# ---------- Importación de las librerías en nuestro sistema ----------
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd                                                                             # Librerías para manipular datos


# %%
# ---------- Importación del Dataset Data.csv ubicado en la misma carpeta ----------
dataset = pd.read_csv('Position_Salaries.csv')                                                               # Guardamos el dataset en una carpeta
##print(dataset)                                                                                # --> Podemos descomentar para hacer traza   


# %%
# ---------- Separamos las variables dependientes de las independientes ----------
X = dataset.iloc[:, 1:2].values                                                                 # Indicamos que queremos desde la primera columna (:) hasta la última, salvo la última (:-1) !!!!!![filas, columnas]!!!!
##print(X)                                                                                      # --> Podemos descomentar para hacer traza
y = dataset.iloc[:, 2].values                                                                  # Indicamos que solo queremos la última columna <-- VARIABLE A PREDECIR
##print(y)                                                                                      # --> Podemos descomentar para hacer traza

# %%
# ---------- Ajustar la regresión lineal con el dataset ----------
from sklearn.linear_model import LinearRegression
linear_regression = LinearRegression()
linear_regression.fit(X,y)

# %%
# ---------- Ajustar la regresión polinómica con el dataset ----------
from sklearn.preprocessing import PolynomialFeatures                                            # Librería para establecer la regresión polinómica
polynomial_regression = PolynomialFeatures(degree=2)                                            # Genera características de X y sus cuadrados (o su cubo... epende del grado)
X_poly = polynomial_regression.fit_transform(X)                                                 # Confirmamos y aplicamos los cambios en X
X_poly = X[:, 1:]
# %%
