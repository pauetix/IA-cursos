# %%
# ---------- Importación de las librerías en nuestro sistema ----------
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# %%
# ---------- Importamos el dataset ----------
dataset = pd.read_csv('Salary_Data.csv')                                                                # Cargamos el dataset en una variable
## print(dataset)                                                                                       # Traza
X = dataset.iloc[:, :-1].values                                                                         # Variables independientes
y = dataset.iloc[:, -1].values                                                                          # Variable dependiente
## print(X)                                                                                             # Traza
## print(y)                                                                                             # Traza


# %%
# ---------- Se divide el dataset en Train y Test ----------
from sklearn.model_selection import train_test_split                                                    # Librería para dividir los datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)            # División de X e y en Train y Test
## print(X_train)                                                                                       # Traza
## print(X_test)                                                                                        # Traza
## print(y_train)                                                                                       # Traza
## print(y_test)                                                                                        # Traza


# %%
# ---------- Escalado de las variables ----------
# En regresión lineal no es necesario escalar las variables, 
# además de que en este caso solo tenemos una variable independiente


# %%
# ---------- Creación del modelo de regresión lineal con el conjunto de entrenamiento ----------
from sklearn.linear_model import LinearRegression                                                       # Clase para crear modelos de regresión lineal
regression = LinearRegression()                                                                         # Instanciamos el objeto para la creación del modelo
regression.fit(X_train, y_train)                                                                        # Ajustamos el modelo con la variable dependiente y la independiente (hay más parámetros)


# %%
# ---------- Predecimos el conjunto de test ----------
y_pred = regression.predict(X_test)                                                                     # Creamos el vector con los variables que se predicen
##print(y_test)                                                                                         # Traza
##print(y_pred)                                                                                         # Traza


# %%
# ---------- Visualización de los resultados de entrenamiento ----------
plt.scatter(X_train, y_train, color = "red")                                                            # Pintamos los puntos con la función scatter de pyplot
plt.plot(X_train, regression.predict(X_train), color = "blue")                                          # Pintamos la línea de regresión
plt.title("Sueldo vs años de experiencia (Conjunto de entrenamiento)")                                  # Damos un título al gráfico
plt.xlabel("Años de experiencia")                                                                       # Asignamos un valor al eje X
plt.ylabel("Sueldo (En $)")                                                                             # Asinamos un valor al eje Y
plt.show()                                                                                              # Renderiza lo que se ha hecho en la gráfica para mostrarlo


# %%
# ---------- Visualización de los resultados de test ----------
plt.scatter(X_test, y_test, color = "red")                                                              # Pintamos los puntos con la función scatter de pyplot
plt.plot(X_train, regression.predict(X_train), color = "blue")                                          # Pintamos la línea de regresión
plt.title("Sueldo vs años de experiencia (Conjunto de testing)")                                        # Damos un título al gráfico
plt.xlabel("Años de experiencia")                                                                       # Asignamos un valor al eje X
plt.ylabel("Sueldo (En $)")                                                                             # Asinamos un valor al eje Y
plt.show()                                                                                              # Renderiza lo que se ha hecho en la gráfica para mostrarlo
