"""Uso del modelo en producción: puntuar clientes y priorizar la campaña.

Flujo pensado para el banco:
1. Guardar el modelo entrenado (pipeline completo) con `save_model`.
2. Cada período, cargarlo con `load_model` y pasarle los datos crudos de los
   clientes (mismas columnas que bank_churn.xlsx, sin attrition_flag).
3. `score_clients` devuelve la lista priorizada con la marca de a quién contactar.
"""

import joblib
import numpy as np
import pandas as pd

from src import config as cfg
from src.evaluation.montecarlo import ASSUMPTIONS, client_values

DEFAULT_MODEL_PATH = cfg.MODELS_DIR / "churn_xgboost.joblib"


def save_model(model, path=DEFAULT_MODEL_PATH):
    """Guarda el pipeline completo (preprocesamiento + modelo) en disco."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path


def load_model(path=DEFAULT_MODEL_PATH):
    """Carga el pipeline guardado con `save_model`."""
    return joblib.load(path)


def score_clients(model, X, contact_fraction=0.11, assumptions=None):
    """Lista de clientes priorizada para la campaña de retención.

    Para cada cliente calcula:
    - prob. churn: probabilidad de irse según el modelo.
    - valor anual (USD): intereses + comisiones estimados con sus datos.
    - pérdida esperada (USD): prob. churn x valor anual x años de horizonte.
    - prioridad: posición según la pérdida esperada (1 = contactar primero).
    - contactar: True para el `contact_fraction` de clientes con mayor pérdida
      esperada (por defecto 11%, el óptimo de la simulación Montecarlo).
    """
    a = assumptions or ASSUMPTIONS
    proba = model.predict_proba(X)[:, 1]
    annual = client_values(X, a["interest_rate"][1], a["interchange_fee"][1])
    expected_loss = proba * annual * a["horizon_years"]
    out = pd.DataFrame(
        {
            "prob. churn": proba,
            "valor anual (USD)": annual,
            "pérdida esperada (USD)": expected_loss,
        },
        index=X.index,
    ).sort_values("pérdida esperada (USD)", ascending=False)
    out["prioridad"] = np.arange(1, len(out) + 1)
    n_contact = int(round(contact_fraction * len(out)))
    out["contactar"] = out["prioridad"] <= n_contact
    return out


def campaign_summary(scored, y=None):
    """Resumen de la lista: clientes a contactar y, si se conoce la realidad, cuántos se iban."""
    contact = scored[scored["contactar"]]
    rows = {
        "clientes en la lista": str(len(scored)),
        "clientes a contactar": str(len(contact)),
        "% a contactar": f"{100 * len(contact) / len(scored):.1f}",
        "prob. churn media de los contactados": f"{contact['prob. churn'].mean():.3f}",
        "pérdida esperada total de los contactados (USD)": f"{contact['pérdida esperada (USD)'].sum():,.0f}",
    }
    if y is not None:
        real = pd.Series(y).reindex(contact.index)
        rows["churners reales entre los contactados"] = str(int(real.sum()))
        rows["% churners reales entre los contactados"] = f"{100 * real.mean():.1f}"
    return pd.Series(rows, name="valor").to_frame()
