# %%
# ---------- Importación de las librerías en nuestro sistema ----------
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd                                                                             # Librerías para manipular datos


# %%
# ---------- Importación del Dataset ubicado en la misma carpeta ----------
dataset = pd.read_csv('Position_Salaries.csv')                                                  # Guardamos el dataset en una carpeta
##print(dataset)                                                                                # --> Podemos descomentar para hacer traza   


# %%
# ---------- Separamos las variables dependientes de las independientes ----------
X = dataset.iloc[:, 1:2].values                                                                 # Indicamos que queremos desde la primera columna (:) hasta la última, salvo la última (:-1) !!!!!![filas, columnas]!!!!
##print(X)                                                                                      # --> Podemos descomentar para hacer traza
y = dataset.iloc[:, 2].values                                                                   # Indicamos que solo queremos la última columna <-- VARIABLE A PREDECIR
##print(y)                                                                                      # --> Podemos descomentar para hacer traza


# %%
# ---------- Dividir dataset en cjto de entrenamiento y test ----------
"""
from sklearn.model_selection import train_test_split                                            # Librería para dividir dataset en conjuntos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)    # Creación de variables
"""


# %%
# ---------- Escalado de las variables ----------
from sklearn.preprocessing import StandardScaler                                                # Librería para poder normalizar los valores, que no sean tan lejanos entre sí para que no haya tanjta diferencia de peso
sc_X = StandardScaler()                                                                         # Creamos una instancia de StandardScaler para X
sc_y = StandardScaler()                                                                         # Creamos una instancia de StandardScaler para y
X = sc_X.fit_transform(X)                                                                       # Transformamos la matriz de características --> Valores alterados
y = sc_y.fit_transform(y.reshape(-1, 1))                                                        # Transoformamos el vector de lo que se quiere predecir --> Valores alterados


# %%
# ---------- Ajustar la regresión con el dataset ----------
from sklearn.svm import SVR
regression = SVR(kernel='rbf')                                                                  # Kernel gaussiano
regression.fit(X, y)


# %%
# ---------- Predicción de nuestros modelo de regresión lineal ----------
sc_y.inverse_transform(regression.predict(sc_X.transform(np.array([[6.5]]))).reshape(-1,1))     # Predicción (Con el valor a predecir adaptado a la transformación) --> Alterada por los valores alterados, requiere transformación inversa

# %%
# ---------- Visualización de los resultados del modelo Polinomico ----------
# Lo que hacemos aquí es usar pyplot para pintar las gráficas y la nube de puntos
##X_grid = np.arange(min(X), max(X), 0.1)                                                       # Creamos un nuevo vector con más datos entre el mínimo y el máximo de X
##X_grid = X_grid.reshape((len(X_grid), 1))                                                     # Convertimos X_grid de vector fila a vector columna
plt.scatter(X, y, color = 'red')
plt.plot(X, regression.predict(X), color = 'blue')                                              # Aquí usaríamos X_grid si quisiéramos suavizar la gráfica
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.ticklabel_format(style='plain', axis='y')                                                   # Para poder mostrar los valores reales que hay en el eje de las y
plt.show()

# %%
