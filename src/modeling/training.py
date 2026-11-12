"""Validación cruzada y comparación de modelos."""

import time

import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate

from src import config as cfg
from src.evaluation.metrics import CV_SCORING, METRIC_LABELS


def make_cv(n_splits=cfg.CV_FOLDS, random_state=cfg.RANDOM_STATE):
    """Validación cruzada estratificada: cada fold conserva el 16% de churn."""
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def cross_validate_model(model, X, y, cv=None, scoring=None):
    """Evalúa un pipeline con validación cruzada estratificada."""
    cv = cv or make_cv()
    scoring = scoring or CV_SCORING
    res = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    folds = pd.DataFrame({name: res[f"test_{name}"] for name in scoring})
    folds["fit_time_s"] = res["fit_time"]
    folds.index = [f"fold {i + 1}" for i in range(len(folds))]
    return folds


def summarize_cv(results: dict) -> pd.DataFrame:
    """Tabla resumen 'media ± desviación' a partir de {nombre: folds}."""
    rows = {}
    for name, folds in results.items():
        stats = {}
        for col in folds.columns:
            stats[(METRIC_LABELS.get(col, col), "media")] = folds[col].mean()
            stats[(METRIC_LABELS.get(col, col), "desv")] = folds[col].std()
        rows[name] = stats
    out = pd.DataFrame(rows).T
    out.columns = pd.MultiIndex.from_tuples(out.columns)
    return out


def format_cv_summary(summary: pd.DataFrame, decimals=3) -> pd.DataFrame:
    """Convierte el resumen en texto 'media ± desv' por métrica."""
    metrics = summary.columns.get_level_values(0).unique()
    return pd.DataFrame(
        {
            m: summary[(m, "media")].round(decimals).astype(str)
            + " ± "
            + summary[(m, "desv")].round(decimals).astype(str)
            for m in metrics
        }
    )


def run_experiments(models: dict, X, y, cv=None) -> tuple[pd.DataFrame, dict]:
    """Corre validación cruzada para varios modelos {nombre: pipeline}.

    Devuelve (resumen, resultados por fold).
    """
    cv = cv or make_cv()
    results = {}
    for name, model in models.items():
        start = time.perf_counter()
        results[name] = cross_validate_model(model, X, y, cv=cv)
        print(f"{name}: {time.perf_counter() - start:.1f} s")
    return summarize_cv(results), results
