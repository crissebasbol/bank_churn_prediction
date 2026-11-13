"""Optimización: regresión logística entrenada con descenso de gradiente estocástico (SGD).

Se usa `SGDClassifier` de sklearn con `loss="log_loss"` (regresión logística).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import log_loss
from sklearn.utils.class_weight import compute_sample_weight

from src import config as cfg


def make_sgd(eta0=0.01, alpha=1e-4, learning_rate="constant", class_weight=None, max_iter=1000):
    """SGDClassifier configurado como regresión logística.

    - eta0: tasa de aprendizaje
    - alpha: regularización L2.
    - learning_rate="constant": el paso no cambia entre épocas (para ver su efecto);
      "optimal" lo reduce automáticamente a medida que avanza.
    """
    return SGDClassifier(
        loss="log_loss", penalty="l2", alpha=alpha,
        learning_rate=learning_rate, eta0=eta0,
        class_weight=class_weight, max_iter=max_iter,
        random_state=cfg.RANDOM_STATE,
    )


def sgd_loss_curve(X, y, eta0=0.01, alpha=1e-4, epochs=50, balanced=True):
    """Entrena SGD época por época (partial_fit) y registra la pérdida en cada una."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    weights = compute_sample_weight("balanced", y) if balanced else None
    model = make_sgd(eta0=eta0, alpha=alpha)
    rng = np.random.default_rng(cfg.RANDOM_STATE)
    losses = []
    for _ in range(epochs):
        order = rng.permutation(len(y))  # se baraja el orden de los clientes en cada época
        model.partial_fit(X[order], y[order], classes=np.array([0, 1]),
                          sample_weight=None if weights is None else weights[order])
        proba = np.clip(model.predict_proba(X)[:, 1], 1e-15, 1 - 1e-15)
        losses.append(log_loss(y, proba, sample_weight=weights))
    return pd.Series(losses, index=pd.RangeIndex(1, epochs + 1, name="época"), name=f"eta0={eta0}")


def loss_curves_by_learning_rate(X, y, eta0_values, alpha=1e-4, epochs=50, balanced=True):
    """Curvas de pérdida para varias tasas de aprendizaje: DataFrame (épocas x eta0)."""
    return pd.concat(
        [sgd_loss_curve(X, y, eta0=e, alpha=alpha, epochs=epochs, balanced=balanced) for e in eta0_values],
        axis=1,
    )


def plot_loss_curves(curves, title="Curva de convergencia: pérdida por época", logy=False):
    """Grafica una o varias curvas de pérdida (Serie o DataFrame)."""
    curves = curves.to_frame() if isinstance(curves, pd.Series) else curves
    fig, ax = plt.subplots(figsize=(9, 5))
    for col in curves.columns:
        ax.plot(curves.index, curves[col], marker="o", markersize=3, label=col)
    ax.set_xlabel("Época (pasadas completas por el train)")
    ax.set_ylabel("Log-loss ponderada (train)")
    ax.set_title(title)
    if logy:
        ax.set_yscale("log")
    if curves.shape[1] > 1:
        # leyenda fuera del área del gráfico para no tapar las curvas
        ax.legend(title="Tasa de aprendizaje", loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.tight_layout()
    return fig


def loss_curve_summary(curves):
    """Pérdida inicial, final, mínima y época del mínimo para cada curva."""
    curves = curves.to_frame() if isinstance(curves, pd.Series) else curves
    return pd.DataFrame(
        {
            "pérdida época 1": curves.iloc[0],
            "pérdida final": curves.iloc[-1],
            "pérdida mínima": curves.min(),
            "época del mínimo": curves.idxmin(),
            "variación últimas 10 épocas": curves.iloc[-10:].max() - curves.iloc[-10:].min(),
        }
    ).round(4)
