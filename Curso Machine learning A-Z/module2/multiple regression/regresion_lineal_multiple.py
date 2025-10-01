# %%
# ---------- Importación de las librerías en nuestro sistema ----------
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd                                                                             # Librerías para manipular datos


# %%
# ---------- Importación del Dataset Data.csv ubicado en la misma carpeta ----------
dataset = pd.read_csv('50_Startups.csv')                                                        # Guardamos el dataset en una carpeta
##print(dataset)                                                                                # --> Podemos descomentar para hacer traza   


# %%
# ---------- Separamos las variables dependientes de las independientes ---------
X = dataset.iloc[:, :-1].values                                                                 # Indicamos que queremos desde la primera columna (:) hasta la última, salvo la última (:-1) !!!!!![filas, columnas]!!!!
##print(X)                                                                                      # --> Podemos descomentar para hacer traza
y = dataset.iloc[:, -1].values                                                                  # Indicamos que solo queremos la última columna <-- VARIABLE A PREDECIR
##print(y)                                                                                      # --> Podemos descomentar para hacer traza


# %%
# ---------- Codificar datos categóricos a numéricos ----------
from sklearn.compose import ColumnTransformer                                                   # Permite transformar diferentes a distintas columnas
from sklearn.preprocessing import OneHotEncoder                                                 # Transforma categorías en datos numéricos
ct = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(), [3])],                                           # Decimos qué transformación aplicar y a quién (encoder -> nombre que le damos, OneHotEncorder -> texto a binarios, [] -> Sobre qué columna se aplica)
    remainder='passthrough'                                                                     # Deja igual el resto de columnas
)
X = np.array(ct.fit_transform(X))
print(X)


# %%
# ---------- Evitar trampa de las variables ficticias ----------
X = X[:, 1:]
print(X)

# %%
# ---------- Dividir dataset en cjto de entrenamiento y test ---------- #
from sklearn.model_selection import train_test_split                                            # Librería para dividir dataset en conjuntos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)    # Creación de variables

# %%
## Ajustar modelo de regresión múltiple con el conjunto de entrenamiento
from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(X_train, y_train)


# %%
# ---------- Predicción de los resultados en el conjunto de testing ----------
y_pred = regression.predict(X_test)                                                             # Creamos el vector de predicciones


# %%
# ---------- Construir el modelo óptimo de RLM usando eliminación hacia atrás ----------
import statsmodels.api as sm                                                                    # Librería para añadir y quitar variables de los modelos
X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis = 1)                         # Añadimos una columna de 1 a la tabla original (axis = 1 --> Columna | axis = 0 --> Fila), ponemos ese orden porque si no se ponen los unos al final
SL = 0.05                                                                                       # Definimos el nivel de significación

# %%
# ---------- Primera iteración ----------
X_opt = X[:, [0, 1, 2, 3, 4, 5]].astype(float)                                                  # Matriz de características óptimas (mejores variables independientes)
regression_OLS = sm.OLS(endog = y, exog = X_opt).fit()                                          # Devuelve un p-valor, etc. para cada una de las variables independeintes
print(regression_OLS.summary())                                                                 # Matriz de características óptimas (mejores variables independientes)

# %%
# ---------- Segunda iteración ----------
X_opt = X[:, [0, 1, 3, 4, 5]].astype(float)                                                     # Matriz de características óptimas (mejores variables independientes)
regression_OLS = sm.OLS(endog = y, exog = X_opt).fit()                                          # Devuelve un p-valor, etc. para cada una de las variables independeintes
print(regression_OLS.summary())                                                                 # Matriz de características óptimas (mejores variables independientes)

# %%
# ---------- Tercera Iteración ----------
X_opt = X[:, [0, 3, 4, 5]].astype(float)                                                     # Matriz de características óptimas (mejores variables independientes)
regression_OLS = sm.OLS(endog = y, exog = X_opt).fit()                                          # Devuelve un p-valor, etc. para cada una de las variables independeintes
print(regression_OLS.summary())                                                                 # Matriz de características óptimas (mejores variables independientes)

# %%
# ---------- Cuarta iteración ----------
X_opt = X[:, [0, 3, 5]].astype(float)                                                     # Matriz de características óptimas (mejores variables independientes)
regression_OLS = sm.OLS(endog = y, exog = X_opt).fit()                                          # Devuelve un p-valor, etc. para cada una de las variables independeintes
print(regression_OLS.summary())                                                                 # Matriz de características óptimas (mejores variables independientes)

# %%
# ---------- Quinta iteración ----------
X_opt = X[:, [0, 3]].astype(float)                                                     # Matriz de características óptimas (mejores variables independientes)
regression_OLS = sm.OLS(endog = y, exog = X_opt).fit()                                          # Devuelve un p-valor, etc. para cada una de las variables independeintes
print(regression_OLS.summary())                                                                 # Matriz de características óptimas (mejores variables independientes)

# %%
