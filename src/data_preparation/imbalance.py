"""Estrategias para el desbalance de clases (16% de churn)."""

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.linear_model import LogisticRegression

from src import config as cfg
from src.data_preparation.features import build_preprocessor


def make_pipeline(model, strategy="none", add_avg_ticket=True):
    """Pipeline preprocesamiento + (SMOTE) + modelo según la estrategia.

    - "none": el modelo tal cual.
    - "class_weight": el modelo con class_weight="balanced" (si lo admite):
      equivocarse con un churner pesa ~5 veces más que con un cliente que se queda.
    - "smote": SMOTE genera churners sintéticos hasta igualar las clases.
    """
    steps = list(build_preprocessor(add_avg_ticket=add_avg_ticket).steps)
    if strategy == "smote":
        steps.append(("smote", SMOTE(random_state=cfg.RANDOM_STATE)))
    elif strategy == "class_weight":
        model = model.set_params(class_weight="balanced")
    elif strategy != "none":
        raise ValueError(f"Estrategia desconocida: {strategy}")
    steps.append(("model", model))
    return ImbPipeline(steps)


def imbalance_experiment_models(base_model=None):
    """Los tres pipelines a comparar, con el mismo modelo base."""
    from sklearn.base import clone

    base_model = base_model or LogisticRegression(max_iter=2000, random_state=cfg.RANDOM_STATE)
    return {
        "Sin tratamiento": make_pipeline(clone(base_model), "none"),
        "Pesos de clase": make_pipeline(clone(base_model), "class_weight"),
        "SMOTE": make_pipeline(clone(base_model), "smote"),
    }
