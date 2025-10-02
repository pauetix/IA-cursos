import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR

# ------------------------
# 1. Generamos datos sintéticos
# ------------------------
np.random.seed(42)
X = np.sort(5 * np.random.rand(40, 1), axis=0)
y = np.sin(X).ravel()

# Agregamos ruido en algunos puntos
y[::5] += 2 * (0.5 - np.random.rand(8))

# Rango de valores de epsilon a comparar
epsilons = [0.01, 0.1, 0.5]

# ------------------------
# 2. Graficar SVR para distintos epsilon
# ------------------------
plt.figure(figsize=(15, 4))

for i, eps in enumerate(epsilons, 1):
    # Entrenamos SVR con kernel RBF
    svr = SVR(kernel="rbf", C=100, epsilon=eps)
    svr.fit(X, y)

    # Generamos predicciones
    X_plot = np.linspace(0, 5, 200).reshape(-1, 1)
    y_pred = svr.predict(X_plot)

    # Subplot
    plt.subplot(1, len(epsilons), i)
    plt.scatter(X, y, color="red", label="Datos reales")
    plt.plot(X_plot, y_pred, color="blue", label="Predicción SVR")

    # Tubo epsilon
    plt.plot(X_plot, y_pred + eps, "k--", lw=1, label=r"Tubo $+\epsilon$")
    plt.plot(X_plot, y_pred - eps, "k--", lw=1, label=r"Tubo $-\epsilon$")

    # Vectores de soporte
    plt.scatter(
        svr.support_vectors_[:, 0], 
        y[svr.support_], 
        s=100, facecolors="none", edgecolors="k", label="Vectores de soporte"
    )

    plt.title(f"SVR con epsilon = {eps}\nVectores de soporte: {len(svr.support_)}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()

plt.tight_layout()
plt.show()
