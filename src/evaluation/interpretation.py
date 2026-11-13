"""Interpretación del modelo: qué variables usa para predecir el churn."""

import matplotlib.pyplot as plt
import pandas as pd


def model_feature_importance(pipeline):
    """Importancia propia del modelo final (ej. XGBoost `feature_importances_`).

    Se mide sobre las 23 columnas que ve el modelo después del preprocesamiento
    (incluye `avg_ticket` y los indicadores de Unknown). En XGBoost es la
    ganancia promedio que aporta cada variable en las divisiones de los árboles,
    normalizada para que sume 1.
    """
    model = pipeline.named_steps["model"]
    names = pipeline.named_steps["encode_scale"].get_feature_names_out()
    imp = pd.Series(model.feature_importances_, index=names, name="importancia")
    return imp.sort_values(ascending=False).to_frame()


def plot_importance(table, column, title, top=15, error_col=None, color="#E0702F"):
    """Barras horizontales con las `top` variables más importantes."""
    data = table.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 0.38 * len(data) + 1.4))
    ax.barh(data.index, data[column], color=color,
            xerr=data[error_col] if error_col else None,
            error_kw={"ecolor": "#333333", "capsize": 3, "elinewidth": 1})
    ax.set_xlabel(column)
    ax.set_title(title, loc="left")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig
