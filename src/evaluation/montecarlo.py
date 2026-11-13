"""Simulación Montecarlo del impacto de negocio de la campaña de retención.

Pregunta: ¿a qué clientes y a cuántos conviene ofrecerles el beneficio para
maximizar la ganancia, considerando la incertidumbre en el valor de los
clientes, el costo del beneficio y la efectividad de la oferta?

Todos los supuestos están en ASSUMPTIONS y se pueden modificar. Los supuestos
inciertos son distribuciones triangulares (mínimo, más probable, máximo).
Los cálculos se hacen en USD (la moneda de los datos); las tablas agregan
columnas en COP con la TRM supuesta.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config as cfg

ASSUMPTIONS = {
    # Valor anual de cada cliente estimado con sus propios datos
    "interest_rate": (0.20, 0.25, 0.30),      # interés anual sobre total_revolving_bal
    "interchange_fee": (0.015, 0.02, 0.025),  # comisión sobre total_trans_amt
    # Campaña
    "benefit_cost": (50, 80, 120),            # costo del beneficio por cliente contactado (USD)
    "retention_prob": (0.20, 0.35, 0.50),     # prob. de que un churner contactado se quede
    # Supuestos fijos
    "horizon_years": 3,                       # años de valor que aporta un cliente retenido
    "trm": 3000,                              # COP por USD (solo para reportar)
}

ASSUMPTION_LABELS = {
    "interest_rate": "Tasa de interés anual sobre el balance rotativo",
    "interchange_fee": "Comisión sobre el monto de transacciones",
    "benefit_cost": "Costo del beneficio por cliente contactado (USD)",
    "retention_prob": "Prob. de que un churner contactado se quede",
    "horizon_years": "Años de valor de un cliente retenido",
    "trm": "TRM (COP por USD), solo para reportar",
}

STRATEGY_COLORS = {
    "Por pérdida esperada": "#E0702F",
    "Por probabilidad": "#2F6DB5",
    "Al azar": "#9AA5B1",
}


def assumptions_table(assumptions=None):
    """Tabla con los supuestos de la simulación."""
    a = assumptions or ASSUMPTIONS
    rows = {}
    for key, label in ASSUMPTION_LABELS.items():
        v = a[key]
        if isinstance(v, tuple):
            rows[label] = {"mínimo": str(v[0]), "más probable": str(v[1]), "máximo": str(v[2])}
        else:
            rows[label] = {"mínimo": "-", "más probable": f"{v} (fijo)", "máximo": "-"}
    return pd.DataFrame(rows).T


def client_values(X, interest_rate, interchange_fee, horizon_years=1):
    """Valor estimado de cada cliente: (intereses + comisiones) x años de horizonte."""
    annual = (X["total_revolving_bal"].to_numpy() * interest_rate
              + X["total_trans_amt"].to_numpy() * interchange_fee)
    return annual * horizon_years


def client_value_summary(X, y, assumptions=None):
    """Valor anual estimado (con los valores 'más probables') por clase, en USD."""
    a = assumptions or ASSUMPTIONS
    v = client_values(X, a["interest_rate"][1], a["interchange_fee"][1])
    df = pd.DataFrame({"v": v, "clase": np.where(np.asarray(y) == 1, "churn (1)", "se queda (0)")})
    out = df.groupby("clase")["v"].agg(["count", "mean", "median", "sum"])
    out.columns = ["clientes", "valor anual medio (USD)", "valor anual mediano (USD)", "valor anual total (USD)"]
    out["clientes"] = out["clientes"].astype(str)  # se muestra como entero en el notebook
    return out


def priority_scores(X, proba, assumptions=None):
    """Puntajes para ordenar a los clientes según cada estrategia.

    - "Por probabilidad": probabilidad de churn del modelo.
    - "Por pérdida esperada": probabilidad x valor del cliente (con los valores
      más probables). Prioriza a quienes es más costoso perder.
    """
    a = assumptions or ASSUMPTIONS
    value = client_values(X, a["interest_rate"][1], a["interchange_fee"][1], a["horizon_years"])
    proba = np.asarray(proba)
    return {"Por probabilidad": proba, "Por pérdida esperada": proba * value}


def simulate_campaign(X, y, scores, fractions=None, n_sims=10_000, assumptions=None, seed=cfg.RANDOM_STATE):
    """Simula la ganancia neta de contactar a distintos % de clientes.

    `scores` es {estrategia: puntaje}; para cada estrategia se contacta a los
    k clientes de mayor puntaje. Se agrega la estrategia "Al azar" (k clientes
    distintos en cada simulación) como referencia.

    En cada una de las `n_sims` simulaciones (los mismos escenarios para todas
    las estrategias, para compararlas en igualdad de condiciones):
      1. Se sortean la tasa de interés, la comisión, el costo del beneficio y
         la efectividad de la oferta.
      2. Se calcula el valor de cada cliente con sus datos (x años de horizonte).
      3. Cada churner real contactado se queda con probabilidad = efectividad
         (Bernoulli). Los no churners contactados reciben el beneficio igual.
      4. Ganancia neta = valor de los churners retenidos - costo de los beneficios.
    La ganancia es incremental frente a no hacer campaña (ganancia 0).

    Devuelve un DataFrame con una fila por (estrategia, % contactados, simulación).
    """
    a = assumptions or ASSUMPTIONS
    fractions = fractions if fractions is not None else np.round(np.arange(0.01, 0.301, 0.01), 2)
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    n = len(y)

    rate = rng.triangular(*a["interest_rate"], size=n_sims)
    fee = rng.triangular(*a["interchange_fee"], size=n_sims)
    cost = rng.triangular(*a["benefit_cost"], size=n_sims)
    p_ret = rng.triangular(*a["retention_prob"], size=n_sims)

    bal = X["total_revolving_bal"].to_numpy(dtype=np.float32)
    amt = X["total_trans_amt"].to_numpy(dtype=np.float32)
    # valor que se recupera por cliente en cada simulación (0 si no es churner o no se retiene)
    value = (bal[None, :] * rate[:, None].astype(np.float32)
             + amt[None, :] * fee[:, None].astype(np.float32)) * a["horizon_years"]
    retained = rng.random((n_sims, n), dtype=np.float32) < p_ret[:, None]
    gain = value * retained * (y[None, :] == 1)
    del value, retained

    ks = np.array([int(round(f * n)) for f in fractions])
    orders = {name: np.argsort(-np.asarray(s), kind="stable") for name, s in scores.items()}
    rows = []
    for name, order in orders.items():
        cum = gain[:, order].cumsum(axis=1)
        churners = np.cumsum(y[order] == 1)
        for f, k in zip(fractions, ks):
            rows.append(_rows(name, f, k, churners[k - 1], cum[:, k - 1] - cost * k))
    # al azar: un orden distinto en cada simulación
    perm = rng.random((n_sims, n)).argsort(axis=1)
    cum = np.take_along_axis(gain, perm, axis=1).cumsum(axis=1)
    for f, k in zip(fractions, ks):
        rows.append(_rows("Al azar", f, k, np.nan, cum[:, k - 1] - cost * k))
    return pd.concat(rows, ignore_index=True)


def _rows(name, frac, k, churners, net):
    return pd.DataFrame({
        "estrategia": name, "% contactados": round(100 * frac, 1), "clientes contactados": k,
        "churners contactados": churners, "simulación": np.arange(len(net)), "ganancia neta": net,
    })


def simulation_summary(sims, trm=ASSUMPTIONS["trm"]):
    """Media, percentiles 5 y 95 y probabilidad de pérdida por estrategia y %.

    Incluye la ganancia media y los percentiles en millones de COP.
    """
    g = sims.groupby(["estrategia", "% contactados"])["ganancia neta"]
    first = sims.groupby(["estrategia", "% contactados"])[["clientes contactados", "churners contactados"]].first()
    out = pd.DataFrame({
        "clientes contactados": first["clientes contactados"].astype(int).astype(str),
        "churners contactados": first["churners contactados"].map(lambda v: "-" if pd.isna(v) else str(int(v))),
        "ganancia media (USD)": g.mean(),
        "percentil 5 (USD)": g.quantile(0.05),
        "percentil 95 (USD)": g.quantile(0.95),
        "prob. de pérdida %": g.apply(lambda s: 100 * (s < 0).mean()),
    })
    out["ganancia media (millones COP)"] = out["ganancia media (USD)"] * trm / 1e6
    out["percentil 5 (millones COP)"] = out["percentil 5 (USD)"] * trm / 1e6
    out["percentil 95 (millones COP)"] = out["percentil 95 (USD)"] * trm / 1e6
    return out


def best_by_strategy(summary):
    """Para cada estrategia, la fila del % contactado con mayor ganancia media."""
    rows = []
    for name in summary.index.get_level_values(0).unique():
        s = summary.loc[name]
        best = s["ganancia media (USD)"].idxmax()
        rows.append(s.loc[[best]].assign(estrategia=name).reset_index().set_index("estrategia"))
    return pd.concat(rows).sort_values("ganancia media (USD)", ascending=False)


def plot_profit_curves(summary, title_suffix=""):
    """Ganancia media (y banda percentil 5-95) vs % de clientes contactados."""
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for name in summary.index.get_level_values(0).unique():
        s = summary.loc[name]
        color = STRATEGY_COLORS.get(name, "#333333")
        ax.plot(s.index, s["ganancia media (USD)"], marker="o", markersize=3, color=color, label=f"{name} (media)")
        ax.fill_between(s.index, s["percentil 5 (USD)"], s["percentil 95 (USD)"], color=color, alpha=0.15)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("% de clientes contactados")
    ax.set_ylabel("Ganancia neta (USD)")
    ax.set_title(f"Ganancia neta de la campaña (banda = percentil 5 a 95){title_suffix}", loc="left")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.legend(loc="lower left", fontsize=8.5)
    fig.tight_layout()
    return fig


def plot_profit_distribution(sims, strategy, pct):
    """Histograma de la ganancia neta simulada para una estrategia y un % contactado."""
    data = sims[(sims["estrategia"] == strategy) & (sims["% contactados"] == pct)]["ganancia neta"]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.hist(data, bins=60, color=STRATEGY_COLORS.get(strategy, "#E0702F"), alpha=0.85)
    top = ax.get_ylim()[1]
    for q, label in [(0.05, "percentil 5"), (0.5, "mediana"), (0.95, "percentil 95")]:
        v = data.quantile(q)
        ax.axvline(v, color="#333333", linestyle="--", linewidth=1)
        ax.text(v, top * 0.97, f" {label}\n {v:,.0f}", fontsize=8, va="top")
    ax.set_xlabel("Ganancia neta (USD)")
    ax.set_ylabel("Número de simulaciones")
    ax.set_title(f"Distribución de la ganancia neta: {strategy}, {pct:g}% contactados", loc="left")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    fig.tight_layout()
    return fig


def with_horizon(years, assumptions=None):
    """Copia de los supuestos con otro horizonte de años (para sensibilidad)."""
    a = dict(assumptions or ASSUMPTIONS)
    a["horizon_years"] = years
    return a


def format_table(df):
    """Versión para mostrar: USD sin decimales, millones de COP con 1 decimal, % con 2.

    Solo para visualizar en el notebook; los cálculos y gráficos usan la tabla original.
    """
    out = df.copy()
    for col in out.columns:
        if not pd.api.types.is_numeric_dtype(out[col]):
            continue
        if "USD" in col:
            out[col] = out[col].map("{:,.0f}".format)
        elif "COP" in col:
            out[col] = out[col].map("{:,.1f}".format)
        elif "%" in col:
            out[col] = out[col].map("{:.2f}".format)
    return out
