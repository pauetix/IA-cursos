# %%
# Importamos las librerías
print("1️⃣ Importamos librerías necesarias")
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC, OneClassSVM

# %%
# Generación de datos de prueba
print("2️⃣ Generamos datos de dos clases bien separadas (y una nube adicional de puntos lejanos)")
np.random.seed(42)

# Dos clases bien definidas
X_class0 = np.random.randn(50, 2) - [2, 2]
X_class1 = np.random.randn(50, 2) + [2, 2]
X = np.r_[X_class0, X_class1]
y = [0] * 50 + [1] * 50

# Puntos fuera del rango normal ("desconocidos")
X_outliers = np.random.uniform(low=-8, high=8, size=(20, 2))

# %%
# Configuraciones para comparar
print("3️⃣ Definimos tres configuraciones de modelos: lineal, RBF, y One-Class SVM")
configs = [
    ("SVC Lineal (C=1)", SVC(kernel="linear", C=1)),
    ("SVC RBF (C=1)", SVC(kernel="rbf", C=1, gamma=0.5)),
    ("One-Class SVM (solo clase 1)", OneClassSVM(kernel="rbf", gamma=0.5, nu=0.05)),
]

# %%
# Entrenamiento y visualización
print("4️⃣ Entrenamos y graficamos las tres versiones para compararlas visualmente")
plt.figure(figsize=(15, 4))

for i, (title, model) in enumerate(configs, 1):
    # Si es One-Class, solo entrenamos con una clase
    if isinstance(model, OneClassSVM):
        X_train = X_class1  # solo una clase
        model.fit(X_train)
        y_pred = model.predict(np.r_[X_class1, X_outliers])
    else:
        model.fit(X, y)
        y_pred = model.predict(X)

    # Malla para graficar
    xx, yy = np.meshgrid(
        np.linspace(-8, 8, 400),
        np.linspace(-8, 8, 400)
    )
    Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Gráfica
    plt.subplot(1, 3, i)
    plt.contourf(xx, yy, Z > 0, alpha=0.3, cmap=plt.cm.coolwarm)
    plt.contour(xx, yy, Z, colors="k", levels=[-1, 0, 1],
                alpha=0.8, linestyles=["--", "-", "--"])

    # Puntos de entrenamiento
    if isinstance(model, OneClassSVM):
        plt.scatter(X_class1[:, 0], X_class1[:, 1],
                    c="blue", s=50, edgecolors="k", label="Clase conocida (1)")
        plt.scatter(X_outliers[:, 0], X_outliers[:, 1],
                    c="red", marker="x", s=70, label="Fuera de distribución")
    else:
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm,
                    s=50, edgecolors="k", label="Datos conocidos")

    plt.title(title)
    plt.xlabel("Característica 1")
    plt.ylabel("Característica 2")
    plt.legend(loc="upper right")

plt.tight_layout()
plt.show()
# %%
