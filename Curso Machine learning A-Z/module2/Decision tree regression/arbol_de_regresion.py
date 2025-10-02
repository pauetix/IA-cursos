# %%
# ---------- Importación de las librerías en nuestro sistema ----------
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd                                                                             # Librerías para manipular datos


# %%
# ---------- Importación del Dataset Data.csv ubicado en la misma carpeta ----------
dataset = pd.read_csv('Data.csv')                                                               # Guardamos el dataset en una carpeta
##print(dataset)                                                                                # --> Podemos descomentar para hacer traza   


# %%
# ---------- Separamos las variables dependientes de las independientes ----------
X = dataset.iloc[:, :-1].values                                                                 # Indicamos que queremos desde la primera columna (:) hasta la última, salvo la última (:-1) !!!!!![filas, columnas]!!!!
##print(X)                                                                                      # --> Podemos descomentar para hacer traza
y = dataset.iloc[:, -1].values                                                                  # Indicamos que solo queremos la última columna <-- VARIABLE A PREDECIR
##print(y)                                                                                      # --> Podemos descomentar para hacer traza


# %%
# ---------- Tratamiento de los NAs --> Campos sin valor asociado ----------
from sklearn.impute import SimpleImputer                                                        # Importamos solo una función, con esta trataremos los NAs
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")                                 # Indicamos cómo debe actuar en los casos de atributos sin valor
imputer.fit(X[:, 1:3])                                                                          # De 1 a 3 porque python no toma el ultimo valor, con fit se calculan los valores de relleno en las columnas
X[:, 1:3] = imputer.transform(X[:, 1:3])                                                        # Selecciona las columnas y "confirma" esos valores NaN modificados por las medias
##print(X)                                                                                      # --> Descomentar para hacer la traza


# %%
# ---------- Dividir dataset en cjto de entrenamiento y test ----------
from sklearn.model_selection import train_test_split                                            # Librería para dividir dataset en conjuntos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)    # Creación de variables
##print(X_train)                                                                                # --> Descomentar para hacer la traza
##print(y_train)                                                                                # --> Descomentar para hacer la traza
##print(X_test)                                                                                 # --> Descomentar para hacer la traza
##print(y_test)                                                                                 # --> Descomentar para hacer la traza