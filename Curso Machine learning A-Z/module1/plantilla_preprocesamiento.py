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
# ---------- Codificar datos categóricos a numéricos ----------
from sklearn.compose import ColumnTransformer                                                   # Permite transformar diferentes a distintas columnas
from sklearn.preprocessing import OneHotEncoder, LabelEncoder                                   # Transforma categorías en datos numéricos
ct = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(), [0])],                                           # Decimos qué transformación aplicar y a quién (encoder -> nombre que le damos, OneHotEncorder -> texto a binarios, [] -> Sobre qué columna se aplica)
    remainder='passthrough'                                                                     # Deja igual el resto de columnas
    )
X = np.array(ct.fit_transform(X))                                                               # Aplica y ejecuta la transformación

##########También se puede así##########
#                                      #
#            ct.fit(X)                 #
#            X= ct.transform(X)        #
#                                      #
########################################

le = LabelEncoder()                                                                             # Instanciamos un label encorder para transformar la columna de si/no a 1/0
y = le.fit_transform(y)                                                                         # Aplica y ejecuta la transdormación
##print(X)                                                                                      # --> Descomentar para hacer la traza
##print(y)                                                                                      # --> Descomentar para hacer la traza


# %%
# ---------- Dividir dataset en cjto de entrenamiento y test ----------
from sklearn.model_selection import train_test_split                                            # Librería para dividir dataset en conjuntos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)    # Creación de variables
##print(X_train)                                                                                # --> Descomentar para hacer la traza
##print(y_train)                                                                                # --> Descomentar para hacer la traza
##print(X_test)                                                                                 # --> Descomentar para hacer la traza
##print(y_test)                                                                                 # --> Descomentar para hacer la traza


# %%
# ---------- Escalado de las variables ----------
"""from sklearn.preprocessing import StandardScaler                                                # Librería para poder normalizar los valores, que no sean tan lejanos entre sí para que no haya tanjta diferencia de peso
sc = StandardScaler()                                                                           # Creamos una instancia de StandardScaler
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])                                               # Normalizamos conjunto de entrenamiento y aplicamos
X_test[:, 3:] = sc.transform(X_test[:, 3:])                                                     # Aplicamos misma normalización para conjunto de test

print(X_train)
print(X_test)"""
