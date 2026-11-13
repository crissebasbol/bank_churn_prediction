"""Funciones de gráficos y tablas para el análisis exploratorio."""

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from src import config as cfg
from src.viz import COLOR_CHURN, COLOR_STAY, PALETTE

# Variables numéricas con pocos valores enteros: se grafican con una barra por valor
DISCRETE_MAX_UNIQUE = 10


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


def plot_numeric_distributions(df, cols=None, ncols=4):
    """Histogramas de cada variable numérica, separados por clase."""
    cols = cols or cfg.NUMERIC_COLS
    nrows = math.ceil(len(cols) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.2 * nrows))
    axes = axes.ravel()

    for ax, col in zip(axes, cols):
        discrete = df[col].nunique() <= DISCRETE_MAX_UNIQUE
        sns.histplot(
            data=df, x=col, hue=cfg.TARGET, ax=ax,
            stat="density", common_norm=False, discrete=discrete,
            bins="auto" if not discrete else None,
            palette=PALETTE, alpha=0.5, edgecolor=None, legend=False,
        )
        ax.set_title(col, fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(labelsize=8)

    for ax in axes[len(cols):]:
        ax.set_visible(False)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=COLOR_STAY, alpha=0.5),
        plt.Rectangle((0, 0), 1, 1, color=COLOR_CHURN, alpha=0.5),
    ]
    fig.legend(handles, ["Se queda (0)", "Churn (1)"], loc="upper right", ncol=2)
    fig.suptitle("Distribución de variables numéricas por clase (densidad)",
                 fontweight="bold", x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def numeric_medians_by_class(df, cols=None):
    """Mediana de cada variable numérica por clase y su diferencia relativa."""
    cols = cols or cfg.NUMERIC_COLS
    med = df.groupby(cfg.TARGET)[cols].median().T
    med.columns = ["mediana se queda (0)", "mediana churn (1)"]
    base = med["mediana se queda (0)"].replace(0, float("nan"))
    med["dif %"] = (100 * (med["mediana churn (1)"] - base) / base).round(1)
    return med.reindex(med["dif %"].abs().sort_values(ascending=False).index)


def churn_rate_by_value(df, col):
    """Tasa de churn (%) y número de clientes para cada valor de una variable.

    Las variables ordinales salen en su orden natural con 'Unknown' al final;
    las nominales, de mayor a menor número de clientes.
    """
    table = df.groupby(col, observed=True)[cfg.TARGET].agg(clientes="size", churn_pct="mean")
    table["churn_pct"] = (100 * table["churn_pct"]).round(1)
    if col in cfg.ORDINAL_ORDER:
        order = [c for c in cfg.ORDINAL_ORDER[col] + [cfg.UNKNOWN] if c in table.index]
        table = table.reindex(order)
    elif col in cfg.CATEGORICAL_COLS:
        table = table.sort_values("clientes", ascending=False)
    return table.rename(columns={"churn_pct": "churn %"})


def churn_rate_by_categorical(df, cols=None):
    """Tasa de churn por categoría para cada variable categórica: {variable: tabla}."""
    cols = cols or cfg.CATEGORICAL_COLS
    return {col: churn_rate_by_value(df, col) for col in cols}


def plot_categorical_churn_rate(df, cols=None, ncols=3):
    """Barras con la tasa de churn de cada categoría y la tasa general como referencia.

    Encima de cada barra se muestra la tasa y el número de clientes (n)
    """
    cols = cols or cfg.CATEGORICAL_COLS
    overall = 100 * df[cfg.TARGET].mean()
    nrows = math.ceil(len(cols) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.8 * nrows))
    axes = axes.ravel()

    for ax, col in zip(axes, cols):
        table = churn_rate_by_value(df, col)
        # '$' se escapa para que matplotlib no lo interprete como fórmula matemática
        labels = [str(c).replace("$", r"\$") for c in table.index]
        # naranja = por encima de la tasa general (comparando con 1 decimal, como se muestra)
        colors = [COLOR_CHURN if r > round(overall, 1) else COLOR_STAY for r in table["churn %"]]
        bars = ax.bar(labels, table["churn %"], color=colors, alpha=0.85)
        for bar, rate, n in zip(bars, table["churn %"], table["clientes"]):
            ax.text(bar.get_x() + bar.get_width() / 2, rate, f"{rate:.1f}%\nn={n:,}",
                    ha="center", va="bottom", fontsize=7.5)
        ax.axhline(overall, color="gray", linestyle="--", linewidth=1)
        ax.set_title(col, fontsize=10)
        ax.set_ylabel("Tasa de churn (%)")
        ax.set_ylim(0, max(table["churn %"].max(), overall) * 1.3)
        ax.tick_params(axis="x", labelsize=8, rotation=30)
        for label in ax.get_xticklabels():
            label.set_horizontalalignment("right")

    for ax in axes[len(cols):]:
        ax.set_visible(False)

    fig.suptitle(f"Tasa de churn por categoría (línea punteada = tasa general {overall:.1f}%)",
                 fontweight="bold", x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def churn_rate_by_discrete_values(df, cols=None):
    """Tasa de churn por valor para las variables numéricas discretas (pocos valores)."""
    cols = cols or [c for c in cfg.NUMERIC_COLS if df[c].nunique() <= DISCRETE_MAX_UNIQUE]
    return {col: churn_rate_by_value(df, col) for col in cols}


def numeric_findings(df):
    """Chequeos puntuales que soportan las observaciones del EDA numérico."""
    t = cfg.TARGET
    mob_mode = df["months_on_book"].mode()[0]
    cl_max = df["credit_limit"].max()
    high_amt = df["total_trans_amt"] > 11_000
    zero_bal = df["total_revolving_bal"] == 0
    return pd.Series(
        {
            "% churn con balance rotativo = 0": round(100 * zero_bal[df[t] == 1].mean(), 1),
            "% no churn con balance rotativo = 0": round(100 * zero_bal[df[t] == 0].mean(), 1),
            "clientes con total_trans_amt > 11.000": int(high_amt.sum()),
            "churn % con total_trans_amt > 11.000": round(100 * df.loc[high_amt, t].mean(), 1),
            f"% clientes con months_on_book = {mob_mode} (moda)": round(100 * (df["months_on_book"] == mob_mode).mean(), 1),
            f"% clientes con credit_limit = máximo ({cl_max:,.0f})": round(100 * (df["credit_limit"] == cl_max).mean(), 1),
            "correlación credit_limit vs avg_open_to_buy": round(df["credit_limit"].corr(df["avg_open_to_buy"]), 3),
        },
        name="valor",
        dtype=object,  # conserva enteros y decimales cada uno en su formato
    )


def chi2_tests(df, cols=None, alpha=0.05):
    """Prueba chi-cuadrado de independencia entre cada variable categórica y el churn."""
    cols = cols or cfg.CATEGORICAL_COLS
    rows = []
    for col in cols:
        table = pd.crosstab(df[col], df[cfg.TARGET])
        chi2, p, dof, expected = stats.chi2_contingency(table)
        n = table.to_numpy().sum()
        cramers_v = math.sqrt(chi2 / (n * (min(table.shape) - 1)))
        rows.append(
            {
                "variable": col,
                "chi2": round(chi2, 2),
                "gl": dof,
                # como texto con 4 decimales, para que no se redondee a 0 en el notebook
                "p-valor": f"{p:.4f}",
                f"significativa (p<{alpha})": "Sí" if p < alpha else "No",
                "V de Cramér": round(cramers_v, 3),
                "min. esperado": round(expected.min(), 1),
            }
        )
    return pd.DataFrame(rows).set_index("variable").sort_values("p-valor")


def correlation_matrix(df, cols=None, method="spearman"):
    """Matriz de correlación entre las variables numéricas y el churn.

    Por defecto usa Spearman (por rangos): es robusta a asimetrías y valores
    extremos, y capta relaciones monótonas aunque no sean lineales.
    """
    cols = (cols or cfg.NUMERIC_COLS) + [cfg.TARGET]
    return df[cols].corr(method=method)


def plot_correlation_matrix(df, cols=None, method="spearman"):
    """Heatmap del triángulo inferior de la matriz de correlación."""
    corr = correlation_matrix(df, cols, method)
    mask = np.triu(np.ones_like(corr, dtype=bool))  # oculta la diagonal y el triángulo superior
    # la primera fila y la última columna quedan vacías con la máscara: se quitan
    corr, mask = corr.iloc[1:, :-1], mask[1:, :-1]
    fig, ax = plt.subplots(figsize=(12, 9.5))
    sns.heatmap(
        corr, mask=mask, cmap="RdBu_r", vmin=-1, vmax=1, center=0,
        annot=True, fmt=".2f", annot_kws={"size": 7.5},
        linewidths=0.5, cbar_kws={"shrink": 0.7, "label": f"Correlación de {method.capitalize()}"},
        ax=ax,
    )
    ax.set_title(f"Matriz de correlación ({method.capitalize()}) de variables numéricas y churn",
                 loc="left")
    ax.tick_params(labelsize=8.5)
    ax.grid(False)  # la cuadrícula del estilo se dibuja encima del heatmap
    fig.tight_layout()
    return fig


def high_correlation_pairs(df, threshold=0.7, cols=None, method="spearman"):
    """Pares de variables con |correlación| >= threshold, de mayor a menor."""
    corr = correlation_matrix(df, cols, method).drop(index=cfg.TARGET, columns=cfg.TARGET)
    upper = corr.where(np.triu(np.ones_like(corr, dtype=bool), k=1))  # cada par una sola vez
    pairs = upper.stack().rename("correlación").reset_index()
    pairs.columns = ["variable 1", "variable 2", "correlación"]
    pairs = pairs[pairs["correlación"].abs() >= threshold]
    return (pairs.reindex(pairs["correlación"].abs().sort_values(ascending=False).index)
                 .round(3).reset_index(drop=True))


def correlation_with_target(df, cols=None, method="spearman"):
    """Correlación de cada variable numérica con el churn, ordenada por valor absoluto."""
    corr = correlation_matrix(df, cols, method)[cfg.TARGET].drop(cfg.TARGET)
    return (corr.reindex(corr.abs().sort_values(ascending=False).index)
                .round(3).to_frame(f"correlación con churn ({method})"))


def derived_variable_checks(df, tol_amount=1.0, tol_ratio=0.001):
    """Verifica si algunas variables son cálculos exactos de otras.

    - avg_open_to_buy == credit_limit - total_revolving_bal (cupo disponible)
    - avg_utilization_ratio == total_revolving_bal / credit_limit (utilización)

    Devuelve el % de clientes en que se cumple la igualdad (dentro de la
    tolerancia indicada) y la mayor diferencia encontrada.
    """
    open_diff = (df["credit_limit"] - df["total_revolving_bal"] - df["avg_open_to_buy"]).abs()
    util_diff = (df["total_revolving_bal"] / df["credit_limit"] - df["avg_utilization_ratio"]).abs()
    return pd.DataFrame(
        {
            "identidad": [
                "avg_open_to_buy = credit_limit - total_revolving_bal",
                "avg_utilization_ratio = total_revolving_bal / credit_limit",
            ],
            "tolerancia": [tol_amount, tol_ratio],
            "% clientes que la cumplen": [
                round(100 * (open_diff <= tol_amount).mean(), 2),
                round(100 * (util_diff <= tol_ratio).mean(), 2),
            ],
            "máxima diferencia": [round(open_diff.max(), 4), round(util_diff.max(), 4)],
        }
    ).set_index("identidad")


def _effect_label(r):
    """Etiqueta convencional para el tamaño del efecto (|r|)."""
    r = abs(r)
    if r < 0.1:
        return "despreciable"
    if r < 0.3:
        return "pequeño"
    if r < 0.5:
        return "mediano"
    return "grande"


def mann_whitney_tests(df, cols=None, alpha=0.05):
    """Prueba U de Mann-Whitney por variable numérica: churn (1) vs se queda (0)."""
    cols = cols or cfg.NUMERIC_COLS
    churn = df[df[cfg.TARGET] == 1]
    stay = df[df[cfg.TARGET] == 0]
    n1, n2 = len(churn), len(stay)
    rows = []
    for col in cols:
        u, p = stats.mannwhitneyu(churn[col], stay[col], alternative="two-sided")
        r = 2 * u / (n1 * n2) - 1
        rows.append(
            {
                "variable": col,
                "U": int(u),
                # los p-valores muy pequeños se muestran como texto para que no salgan como 0
                "p-valor": f"{p:.4f}" if p >= 1e-4 else "<0.0001",
                f"significativa (p<{alpha})": "Sí" if p < alpha else "No",
                "efecto r": round(r, 3),
                "magnitud": _effect_label(r),
            }
        )
    out = pd.DataFrame(rows).set_index("variable")
    return out.reindex(out["efecto r"].abs().sort_values(ascending=False).index)


def churn_rate_by_quantile(df, col, q=5):
    """Tasa de churn por cuantiles de una variable numérica (por defecto quintiles).

    Cada grupo tiene aproximadamente el mismo número de clientes; se muestra
    el rango de valores, el número de clientes y la tasa de churn.
    """
    groups = pd.qcut(df[col], q=q, duplicates="drop")
    table = df.groupby(groups, observed=True)[cfg.TARGET].agg(clientes="size", churn_pct="mean")
    table["churn_pct"] = (100 * table["churn_pct"]).round(1)
    table.index = [f"Q{i + 1}: {iv.left:,.1f} a {iv.right:,.1f}" for i, iv in enumerate(table.index)]
    table.index.name = f"{col} (cuantiles)"
    return table.rename(columns={"churn_pct": "churn %"})
