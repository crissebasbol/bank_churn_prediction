"""Decisión individual: ¿vale la pena ofrecerle el beneficio a este cliente?

Regla económica (mismos supuestos de la simulación Montecarlo, en su valor medio):

    beneficio esperado = prob. churn x efectividad x valor anual x años  -  costo del beneficio
                         (lo que se espera recuperar)                       (lo que cuesta ofrecerlo)

Si es positivo, vale la pena retenerlo.
"""

import pandas as pd

from src.evaluation.montecarlo import ASSUMPTIONS, client_values

# Umbral de máximo F1 (sección 5.1): a partir de aquí el modelo clasifica al cliente como churn
CHURN_THRESHOLD = 0.65


def triangular_mean(params):
    """Media de una distribución triangular (mínimo, más probable, máximo)."""
    return sum(params) / 3


def complete_derived_fields(client: pd.DataFrame) -> pd.DataFrame:
    """Calcula las variables que son cálculos exactos de otras (sección 2.7)."""
    out = client.copy()
    out["avg_open_to_buy"] = out["credit_limit"] - out["total_revolving_bal"]
    out["avg_utilization_ratio"] = (out["total_revolving_bal"] / out["credit_limit"]).round(3)
    return out


def retention_decision(model, client: pd.DataFrame, assumptions=None) -> list[dict]:
    """Probabilidad, valor, beneficio esperado, decisión y explicación para cada cliente.

    `client` es un DataFrame con las columnas crudas del dataset (una fila por
    cliente). Si faltan avg_open_to_buy o avg_utilization_ratio, se calculan.
    """
    a = assumptions or ASSUMPTIONS
    client = complete_derived_fields(client)
    # mismas columnas y en el mismo orden que en el entrenamiento
    client = client[list(model.named_steps["features"].feature_names_in_)]
    proba = model.predict_proba(client)[:, 1]
    annual = client_values(client, a["interest_rate"][1], a["interchange_fee"][1])
    effectiveness = triangular_mean(a["retention_prob"])
    cost = triangular_mean(a["benefit_cost"])
    years = a["horizon_years"]

    results = []
    for p, v in zip(proba, annual):
        recovered = p * effectiveness * v * years
        net = recovered - cost
        worth_it = net > 0
        if worth_it:
            explanation = (
                f"Vale la pena retenerlo: tiene {100 * p:.1f}% de probabilidad de irse y un valor estimado "
                f"de {v:,.0f} USD al año. Ofrecerle el beneficio recupera en promedio {recovered:,.0f} USD "
                f"en {years} años, más que el costo del beneficio ({cost:,.0f} USD)."
            )
        elif p >= CHURN_THRESHOLD:
            explanation = (
                f"No vale la pena: aunque tiene alta probabilidad de irse ({100 * p:.1f}%), su valor estimado "
                f"es bajo ({v:,.0f} USD al año). Lo que se espera recuperar ({recovered:,.0f} USD en {years} años) "
                f"no cubre el costo del beneficio ({cost:,.0f} USD)."
            )
        else:
            explanation = (
                f"No hace falta: es poco probable que se vaya ({100 * p:.1f}%). Ofrecerle el beneficio costaría "
                f"{cost:,.0f} USD y solo se espera recuperar {recovered:,.0f} USD."
            )
        results.append(
            {
                "prob_churn": round(float(p), 4),
                "modelo_predice_churn": bool(p >= CHURN_THRESHOLD),
                "valor_anual_usd": round(float(v), 2),
                "perdida_esperada_usd": round(float(p * v * years), 2),
                "recuperacion_esperada_usd": round(float(recovered), 2),
                "costo_beneficio_usd": round(float(cost), 2),
                "beneficio_esperado_usd": round(float(net), 2),
                "vale_la_pena_retener": bool(worth_it),
                "explicacion": explanation,
            }
        )
    return results


def decisions_table(model, clients: pd.DataFrame, assumptions=None) -> pd.DataFrame:
    """Versión en tabla de retention_decision (una fila por cliente)."""
    return pd.DataFrame(retention_decision(model, clients, assumptions), index=clients.index)
