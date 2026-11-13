"""Clientes de ejemplo precargados en la app: clientes reales del test.

Se eligieron para mostrar los distintos casos de la regla de decisión.
Se sabe si cada uno se fue o no en la realidad, lo que permite comparar.
"""

from src import config as cfg
from src.data_understanding.loading import load_raw

EXAMPLES = [
    {"clientnum": 759584958, "nombre": "Alto riesgo, alto valor",
     "descripcion": "Usa el crédito rotativo y transa bastante, pero el modelo lo ve muy propenso a irse."},
    {"clientnum": 714076683, "nombre": "Riesgo moderado, alto valor",
     "descripcion": "Probabilidad moderada de irse, pero perderlo cuesta mucho."},
    {"clientnum": 714602658, "nombre": "Alto riesgo, bajo valor",
     "descripcion": "Casi seguro se va, pero no usa el crédito y transa poco: le deja poco al banco."},
    {"clientnum": 713758758, "nombre": "Bajo riesgo, alto valor",
     "descripcion": "Cliente muy activo y rentable, sin señales de irse."},
]

# Variables que se piden en el formulario (avg_open_to_buy y avg_utilization_ratio se calculan)
INPUT_FIELDS = [c for c in cfg.CATEGORICAL_COLS + cfg.NUMERIC_COLS
                if c not in ("avg_open_to_buy", "avg_utilization_ratio")]


def load_examples():
    """Datos crudos de los clientes de ejemplo y si se fueron en la realidad."""
    raw = load_raw().set_index(cfg.ID_COL)
    out = []
    for ex in EXAMPLES:
        row = raw.loc[ex["clientnum"]]
        fields = {c: (row[c].item() if hasattr(row[c], "item") else row[c]) for c in INPUT_FIELDS}
        out.append({**ex, "datos": fields, "realidad": "Se fue" if row[cfg.TARGET_RAW] == cfg.POSITIVE_LABEL else "Se quedó"})
    return out
