"""Gráficos de evaluación de modelos."""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.evaluation.metrics import METRIC_LABELS


def plot_grid_heatmap(search, param_x, param_y, metric="pr_auc"):
    """Heatmap de una métrica de validación cruzada para dos hiperparámetros.

    `param_x` y `param_y` son los nombres en el pipeline (ej. "model__gamma").
    Cada celda es la media de los 5 folds para esa combinación.
    """
    res = pd.DataFrame(search.cv_results_)
    table = res.pivot_table(
        index=f"param_{param_y}", columns=f"param_{param_x}", values=f"mean_test_{metric}"
    )
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(table, annot=True, fmt=".3f", cmap="viridis", ax=ax,
                cbar_kws={"label": f"{METRIC_LABELS[metric]} (media 5 folds)"})
    ax.set_xlabel(param_x.replace("model__", ""))
    ax.set_ylabel(param_y.replace("model__", ""))
    ax.set_title(f"{METRIC_LABELS[metric]} por combinación de hiperparámetros", loc="left")
    ax.grid(False)
    fig.tight_layout()
    return fig


def plot_param_curve(search, param, metrics=("pr_auc", "roc_auc"), logx=True):
    """Métricas de validación cruzada (media ± desviación) contra un hiperparámetro.

    Útil para búsquedas de un solo parámetro (ej. la regularización alpha).
    La banda sombreada es ± 1 desviación estándar entre los 5 folds.
    """
    res = pd.DataFrame(search.cv_results_)
    x = res[f"param_{param}"].astype(float)
    order = x.argsort()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for metric in metrics:
        mean = res[f"mean_test_{metric}"].iloc[order]
        std = res[f"std_test_{metric}"].iloc[order]
        xs = x.iloc[order]
        ax.plot(xs, mean, marker="o", label=METRIC_LABELS[metric])
        ax.fill_between(xs, mean - std, mean + std, alpha=0.2)
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(param.replace("model__", "") + (" (escala log)" if logx else ""))
    ax.set_ylabel("Métrica (media 5 folds)")
    ax.set_title(f"Métricas según {param.replace('model__', '')}", loc="left")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_mlp_curves(curves: dict):
    """Curvas de pérdida por época de varias redes (y la accuracy de validación
    interna, si usan early stopping, en un segundo panel)."""
    has_val = any("accuracy (validación interna)" in df for df in curves.values())
    fig, axes = plt.subplots(1, 2 if has_val else 1, figsize=(13 if has_val else 8, 4.5))
    axes = list(axes) if has_val else [axes]
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, (name, df) in enumerate(curves.items()):
        color = colors[i % len(colors)]  # mismo color para la misma red en ambos paneles
        axes[0].plot(df.index, df["pérdida (train)"], label=name, color=color)
        if has_val and "accuracy (validación interna)" in df:
            axes[1].plot(df.index, df["accuracy (validación interna)"], label=name, color=color)
    axes[0].set_title("Pérdida en el train por época (backpropagation)", loc="left")
    axes[0].set_xlabel("Época")
    axes[0].set_ylabel("Log-loss + penalización L2")
    axes[0].legend()
    if has_val:
        axes[1].set_title("Accuracy en la validación interna (early stopping)", loc="left")
        axes[1].set_xlabel("Época")
        axes[1].set_ylabel("Accuracy")
        axes[1].legend()
    fig.tight_layout()
    return fig


# Familia de cada modelo según su nombre (para colorear la comparación)
MODEL_FAMILIES = [
    ("Regresión logística", "Lineales (optimización)"),
    ("SVM", "SVM"),
    ("Red neuronal", "Redes neuronales"),
    ("MLP", "Redes neuronales"),
    ("Bagging", "Ensambles"),
    ("Random Forest", "Ensambles"),
    ("AdaBoost", "Ensambles"),
    ("Gradient Boosting", "Ensambles"),
    ("XGBoost", "Ensambles"),
]
FAMILY_COLORS = {
    "Lineales (optimización)": "#9AA5B1",
    "SVM": "#2F6DB5",
    "Redes neuronales": "#7B4FB5",
    "Ensambles": "#E0702F",
    "Otro": "#555555",
}


def model_family(name):
    for key, family in MODEL_FAMILIES:
        if key in name:
            return family
    return "Otro"


def plot_model_comparison(summary, metric="PR-AUC"):
    """Barras horizontales con la métrica (media ± desviación de 5 folds) de cada modelo.

    `summary` es la salida de training.summarize_cv. Los modelos se ordenan de
    mejor a peor y se colorean por familia del curso.
    """
    data = summary[[(metric, "media"), (metric, "desv")]].copy()
    data.columns = ["media", "desv"]
    data = data.sort_values("media")
    families = [model_family(n) for n in data.index]
    colors = [FAMILY_COLORS[f] for f in families]

    fig, ax = plt.subplots(figsize=(9, 0.45 * len(data) + 1.5))
    ax.barh(data.index, data["media"], xerr=data["desv"], color=colors,
            error_kw={"ecolor": "#333333", "capsize": 3, "elinewidth": 1})
    for y, (m, d) in enumerate(zip(data["media"], data["desv"])):
        ax.text(m + d + 0.004, y, f"{m:.3f}", va="center", fontsize=8.5)
    low = max(0, data["media"].min() - 0.1)
    ax.set_xlim(low, 1.0)
    ax.set_xlabel(f"{metric} (media de 5 folds ± desviación)")
    ax.set_title(f"Comparación de modelos por {metric}", loc="left")
    handles = [plt.Rectangle((0, 0), 1, 1, color=FAMILY_COLORS[f])
               for f in FAMILY_COLORS if f in families]
    ax.legend(handles, [f for f in FAMILY_COLORS if f in families], loc="lower right", fontsize=8.5)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig


def plot_threshold_curves(table, marks=None):
    """Precision, Recall y F1 según el umbral de decisión.

    `table` es la salida de metrics.threshold_table. `marks` es un diccionario
    {etiqueta: umbral} para dibujar líneas verticales de referencia.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    for col, color in [("Recall", "#E0702F"), ("Precision", "#2F6DB5"), ("F1", "#333333")]:
        ax.plot(table.index, table[col], marker="o", markersize=3, label=col, color=color)
    for label, t in (marks or {}).items():
        ax.axvline(t, color="gray", linestyle="--", linewidth=1)
        ax.text(t, 0.02, f" {label}\n umbral {t:.2f}", fontsize=8, va="bottom")
    ax.set_xlabel("Umbral de decisión (probabilidad mínima para marcar como churn)")
    ax.set_ylabel("Métrica (predicciones out-of-fold del train)")
    ax.set_ylim(0, 1.02)
    ax.set_title("Compromiso entre Recall y Precision según el umbral", loc="left")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 0.12))
    fig.tight_layout()
    return fig


def plot_confusion_matrix(table, title="Matriz de confusión"):
    """Heatmap de la matriz de confusión (salida de metrics.confusion_table)."""
    fig, ax = plt.subplots(figsize=(6, 4.5))
    total = table.to_numpy().sum()
    labels = [[f"{v:,}\n({100 * v / total:.1f}%)" for v in row] for row in table.to_numpy()]
    sns.heatmap(table, annot=labels, fmt="", cmap="Blues", cbar=False, ax=ax,
                annot_kws={"size": 11}, linewidths=1)
    ax.set_title(title, loc="left")
    ax.tick_params(axis="y", rotation=0)
    ax.grid(False)
    fig.tight_layout()
    return fig


def plot_roc_pr_curves(y, proba, base_rate=None):
    """Curvas ROC y Precision-Recall de un conjunto de predicciones."""
    from sklearn.metrics import (
        average_precision_score, precision_recall_curve, roc_auc_score, roc_curve,
    )

    fpr, tpr, _ = roc_curve(y, proba)
    prec, rec, _ = precision_recall_curve(y, proba)
    base_rate = base_rate if base_rate is not None else float(pd.Series(y).mean())
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].plot(fpr, tpr, color="#E0702F", label=f"Modelo (AUC = {roc_auc_score(y, proba):.3f})")
    axes[0].plot([0, 1], [0, 1], "--", color="gray", label="Al azar (AUC = 0.5)")
    axes[0].set_xlabel("Tasa de falsos positivos")
    axes[0].set_ylabel("Recall (tasa de verdaderos positivos)")
    axes[0].set_title("Curva ROC", loc="left")
    axes[0].legend(loc="lower right")
    axes[1].plot(rec, prec, color="#2F6DB5",
                 label=f"Modelo (AP = {average_precision_score(y, proba):.3f})")
    axes[1].axhline(base_rate, linestyle="--", color="gray",
                    label=f"Al azar (= tasa de churn {base_rate:.3f})")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Curva Precision-Recall", loc="left")
    axes[1].set_ylim(0, 1.02)
    axes[1].legend(loc="lower left")
    fig.tight_layout()
    return fig


def plot_gain_curve(deciles):
    """Curva de ganancia acumulada: % de churners capturados vs % de clientes contactados."""
    x = [0] + list(deciles["% clientes contactados (acum.)"])
    y = [0] + list(deciles["% churners capturados (acum.)"])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, y, marker="o", color="#E0702F", label="Modelo")
    ax.plot([0, 100], [0, 100], "--", color="gray", label="Al azar")
    for xi, yi in zip(x[1:4], y[1:4]):
        ax.annotate(f"{yi:.1f}%", (xi, yi), textcoords="offset points", xytext=(6, -12), fontsize=8.5)
    ax.set_xlabel("% de clientes contactados (ordenados por riesgo)")
    ax.set_ylabel("% de churners capturados")
    ax.set_title("Curva de ganancia acumulada (test)", loc="left")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig
