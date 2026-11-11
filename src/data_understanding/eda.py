"""Funciones de gráficos y tablas para el análisis exploratorio."""

import matplotlib.pyplot as plt

from src import config as cfg
from src.viz import COLOR_CHURN, COLOR_STAY


def plot_target(df):
    """Barras con la distribución de la variable objetivo."""
    counts = df[cfg.TARGET].value_counts().sort_index()
    pct = 100 * counts / counts.sum()
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(["Se queda (0)", "Churn (1)"], counts.values, color=[COLOR_STAY, COLOR_CHURN])
    for bar, n, p in zip(bars, counts.values, pct.values):
        ax.text(bar.get_x() + bar.get_width() / 2, n, f"{n:,}\n({p:.1f}%)",
                ha="center", va="bottom")
    ax.set_ylabel("Clientes")
    ax.set_title("Distribución de la variable objetivo")
    ax.set_ylim(0, counts.max() * 1.18)
    fig.tight_layout()
    return fig
