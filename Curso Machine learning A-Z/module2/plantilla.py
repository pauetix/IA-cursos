# %%
# ---------- Importación de las librerías en nuestro sistema ----------
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd                                                                             # Librerías para manipular datos


# %%
# ---------- Importación del Dataset Data.csv ubicado en la misma carpeta ----------
dataset = pd.read_csv('Position_Salaries.csv')                                                  # Guardamos el dataset en una carpeta
##print(dataset)                                                                                # --> Podemos descomentar para hacer traza   


# %%
# ---------- Separamos las variables dependientes de las independientes ----------
X = dataset.iloc[:, 1:2].values                                                                 # Indicamos que queremos desde la primera columna (:) hasta la última, salvo la última (:-1) !!!!!![filas, columnas]!!!!
##print(X)                                                                                      # --> Podemos descomentar para hacer traza
y = dataset.iloc[:, 2].values                                                                  # Indicamos que solo queremos la última columna <-- VARIABLE A PREDECIR
##print(y)                                                                                      # --> Podemos descomentar para hacer traza


# %%
"""
# ---------- Dividir dataset en cjto de entrenamiento y test ----------
from sklearn.model_selection import train_test_split                                            # Librería para dividir dataset en conjuntos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)    # Creación de variables
##print(X_train)                                                                                # --> Descomentar para hacer la traza
##print(y_train)                                                                                # --> Descomentar para hacer la traza
##print(X_test)                                                                                 # --> Descomentar para hacer la traza
##print(y_test)                                                                                 # --> Descomentar para hacer la traza
"""

# %%
# ---------- Ajustar la regresión con el dataset ----------


# %%
# ---------- Visualización de los resultados del modelo lineal ----------
# Lo que hacemos aquí es usar pyplot para pintar las gráficas y la nube de puntos
plt.scatter(X, y, color = "red")
plt.plot(X, regression.predict(X), color = "blue")
plt.title("Modelo de regresión lineal")
plt.xlabel("Nivel del empleado")
plt.ylabel("Sueldo del empleado en $")
plt.ticklabel_format(style='plain', axis='y')                                                       # Para poder mostrar los valores reales que hay en el eje de las y
plt.show()


# %%
# ---------- Predicción de nuestros modelo de regresión lineal ----------
regression.predict([[6.5]])                                                                  # Predicción con modelo de regresión lineal


# %%
# ---------- Visualización de los resultados del modelo Polinomico ----------
# Lo que hacemos aquí es usar pyplot para pintar las gráficas y la nube de puntos
## X_grid = np.arange(min(X), max(X), 0.1)                                                             # Creamos un nuevo vector con más datos entre el mínimo y el máximo de X
##X_grid = X_grid.reshape((len(X_grid), 1))                                                           # Convertimos X_grid de vector fila a vector columna
plt.scatter(X, y, color = 'red')
plt.plot(X, regression.predict(X), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.ticklabel_format(style='plain', axis='y')                                                       # Para poder mostrar los valores reales que hay en el eje de las y
plt.show()