"""Estilo común de gráficos y guardado de figuras."""

import matplotlib.pyplot as plt
import seaborn as sns

from src import config as cfg

# Colores consistentes: azul = se queda, naranja = churn
COLOR_STAY = "#2F6DB5"
COLOR_CHURN = "#E0702F"
PALETTE = {0: COLOR_STAY, 1: COLOR_CHURN}


def set_style():
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update(
        {
            "figure.figsize": (9, 5),
            "figure.dpi": 100,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save_fig(fig, name: str):
    """Guarda la figura en reports/figures para usarla en el documento."""
    cfg.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(cfg.FIGURES_DIR / f"{name}.png", dpi=150, bbox_inches="tight")
