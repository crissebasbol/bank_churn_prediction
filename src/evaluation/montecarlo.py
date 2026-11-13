"""Simulación Montecarlo del impacto de negocio de la campaña de retención.

Pregunta: ¿a cuántos clientes (los de mayor riesgo según el modelo) conviene
ofrecerles el beneficio para maximizar la ganancia, considerando la
incertidumbre en el valor de los clientes, el costo del beneficio y la
efectividad de la oferta?

Todos los supuestos están en ASSUMPTIONS y se pueden modificar.
Cada supuesto incierto es una distribución triangular (mínimo, más probable, máximo).
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config as cfg

ASSUMPTIONS = {
    # Valor anual de cada cliente estimado con sus propios datos
    "interest_rate": (0.20, 0.25, 0.30),    # interés anual sobre total_revolving_bal
    "interchange_fee": (0.015, 0.02, 0.025),  # comisión sobre total_trans_amt
    # Campaña
    "benefit_cost": (50, 80, 120),          # costo anual del beneficio por cliente contactado (USD)
    "retention_prob": (0.20, 0.35, 0.50),   # prob. de que un churner contactado se quede
}

ASSUMPTION_LABELS = {
    "interest_rate": "Tasa de interés anual sobre el balance rotativo",
    "interchange_fee": "Comisión sobre el monto de transacciones",
    "benefit_cost": "Costo del beneficio por cliente contactado (USD/año)",
    "retention_prob": "Prob. de que un churner contactado se quede",
}


def assumptions_table(assumptions=None):
    """Tabla con los supuestos (mínimo, más probable, máximo) de la simulación."""
    assumptions = assumptions or ASSUMPTIONS
    return pd.DataFrame(
        assumptions, index=["mínimo", "más probable", "máximo"]
    ).T.rename(index=ASSUMPTION_LABELS)


def client_values(X, interest_rate, interchange_fee):
    """Valor anual estimado de cada cliente: intereses + comisiones."""
    return X["total_revolving_bal"].to_numpy() * interest_rate + X["total_trans_amt"].to_numpy() * interchange_fee


def client_value_summary(X, y, assumptions=None):
    """Valor anual estimado (con los valores 'más probables') por clase."""
    assumptions = assumptions or ASSUMPTIONS
    v = client_values(X, assumptions["interest_rate"][1], assumptions["interchange_fee"][1])
    df = pd.DataFrame({"valor anual (USD)": v, "clase": np.where(np.asarray(y) == 1, "churn (1)", "se queda (0)")})
    out = df.groupby("clase")["valor anual (USD)"].agg(["count", "mean", "median", "sum"])
    out.columns = ["clientes", "valor medio", "valor mediano", "valor total"]
    return out


def _draw(rng, params, size):
    low, mode, high = params
    return rng.triangular(low, mode, high, size=size)


def simulate_campaign(X, y, proba, fractions=None, n_sims=10_000, assumptions=None, seed=cfg.RANDOM_STATE):
    """Simula la ganancia neta de contactar a distintos % de clientes.

    Para cada % de clientes contactados (fracción k) se comparan dos formas de
    elegirlos: los k de **mayor probabilidad según el modelo** y k clientes
    **al azar**. En cada una de las `n_sims` simulaciones:
      1. Se sortean los supuestos inciertos (tasa, comisión, costo, efectividad).
      2. Se calcula el valor anual de cada cliente con sus datos.
      3. Cada churner real contactado se queda con probabilidad = efectividad
         (Bernoulli); los no churners contactados reciben el beneficio igual.
      4. Ganancia neta = valor de los churners retenidos - costo de los beneficios.
    La ganancia es **incremental** frente a no hacer campaña (ganancia 0).

    Devuelve un DataFrame largo con una fila por (estrategia, fracción, simulación).
    """
    assumptions = assumptions or ASSUMPTIONS
    fractions = fractions if fractions is not None else np.round(np.arange(0.05, 0.51, 0.05), 2)
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    n = len(y)
    order_model = np.argsort(-np.asarray(proba))

    rate = _draw(rng, assumptions["interest_rate"], n_sims)
    fee = _draw(rng, assumptions["interchange_fee"], n_sims)
    cost = _draw(rng, assumptions["benefit_cost"], n_sims)
    p_ret = _draw(rng, assumptions["retention_prob"], n_sims)
    bal = X["total_revolving_bal"].to_numpy()
    amt = X["total_trans_amt"].to_numpy()

    rows = []
    for frac in fractions:
        k = int(round(frac * n))
        # --- modelo: los k clientes de mayor probabilidad (misma lista en todas las simulaciones)
        idx = order_model[:k]
        churn_idx = idx[y[idx] == 1]
        retained = rng.random((n_sims, len(churn_idx))) < p_ret[:, None]
        value = bal[churn_idx][None, :] * rate[:, None] + amt[churn_idx][None, :] * fee[:, None]
        net_model = (retained * value).sum(axis=1) - cost * k
        # --- al azar: k clientes distintos en cada simulación
        net_random = np.empty(n_sims)
        for s in range(n_sims):
            pick = rng.choice(n, size=k, replace=False)
            c = pick[y[pick] == 1]
            ret = rng.random(len(c)) < p_ret[s]
            net_random[s] = ((bal[c] * rate[s] + amt[c] * fee[s]) * ret).sum() - cost[s] * k
        for name, net, churners in [("Modelo", net_model, len(churn_idx)), ("Al azar", net_random, np.nan)]:
            rows.append(pd.DataFrame({
                "estrategia": name, "% contactados": 100 * frac, "clientes contactados": k,
                "churners contactados (modelo)": churners, "simulación": np.arange(n_sims),
                "ganancia neta": net,
            }))
    return pd.concat(rows, ignore_index=True)


def simulation_summary(sims):
    """Media, percentiles 5 y 95 y probabilidad de pérdida por estrategia y %."""
    g = sims.groupby(["estrategia", "% contactados"])
    out = g.agg(
        clientes_contactados=("clientes contactados", "first"),
        churners_contactados=("churners contactados (modelo)", "first"),
        ganancia_media=("ganancia neta", "mean"),
        p5=("ganancia neta", lambda s: s.quantile(0.05)),
        p95=("ganancia neta", lambda s: s.quantile(0.95)),
        prob_perdida=("ganancia neta", lambda s: 100 * (s < 0).mean()),
    )
    out.columns = [
        "clientes contactados", "churners contactados", "ganancia media (USD)",
        "percentil 5 (USD)", "percentil 95 (USD)", "prob. de pérdida %",
    ]
    return out


def best_strategy(summary, strategy="Modelo"):
    """Fila con el % de clientes contactados que maximiza la ganancia media."""
    s = summary.loc[strategy]
    best = s["ganancia media (USD)"].idxmax()
    return best, s.loc[[best]]


def plot_profit_curves(summary):
    """Ganancia media (y banda percentil 5-95) vs % de clientes contactados."""
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, color in [("Modelo", "#E0702F"), ("Al azar", "#9AA5B1")]:
        s = summary.loc[name]
        ax.plot(s.index, s["ganancia media (USD)"], marker="o", color=color, label=f"{name} (media)")
        ax.fill_between(s.index, s["percentil 5 (USD)"], s["percentil 95 (USD)"], color=color,
                        alpha=0.2, label=f"{name} (percentil 5 a 95)")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("% de clientes contactados")
    ax.set_ylabel("Ganancia neta anual (USD)")
    ax.set_title("Ganancia neta de la campaña según el % de clientes contactados", loc="left")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.legend(loc="lower left", fontsize=8.5)
    fig.tight_layout()
    return fig


def plot_profit_distribution(sims, pct, strategy="Modelo"):
    """Histograma de la ganancia neta simulada para un % de clientes contactados."""
    data = sims[(sims["estrategia"] == strategy) & (sims["% contactados"] == pct)]["ganancia neta"]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.hist(data, bins=60, color="#E0702F", alpha=0.8)
    for q, label in [(0.05, "percentil 5"), (0.5, "mediana"), (0.95, "percentil 95")]:
        v = data.quantile(q)
        ax.axvline(v, color="#333333", linestyle="--", linewidth=1)
        ax.text(v, ax.get_ylim()[1] * 0.95, f" {label}\n {v:,.0f}", fontsize=8, va="top")
    ax.set_xlabel("Ganancia neta anual (USD)")
    ax.set_ylabel("Número de simulaciones")
    ax.set_title(f"Distribución de la ganancia neta: {strategy}, {pct:.0f}% de clientes contactados", loc="left")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    fig.tight_layout()
    return fig
