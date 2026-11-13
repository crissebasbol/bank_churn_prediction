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
    fmt = f"{{:.{decimals}f}}"  # decimales fijos (0.020 y no 0.02)
    return pd.DataFrame(
        {
            m: summary[(m, "media")].map(fmt.format) + " ± " + summary[(m, "desv")].map(fmt.format)
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


def tune_model(pipeline, param_grid, X, y, refit_metric="pr_auc", cv=None):
    """Búsqueda de hiperparámetros con GridSearchCV y los mismos 5 folds.

    Prueba todas las combinaciones de `param_grid` (nombres con el prefijo
    del paso del pipeline, ej. "model__C"), calcula todas las métricas de
    CV_SCORING y elige la mejor según `refit_metric` (por defecto PR-AUC).
    Devuelve el objeto GridSearchCV ya ajustado.
    """
    from sklearn.model_selection import GridSearchCV

    cv = cv or make_cv()
    search = GridSearchCV(
        pipeline, param_grid, scoring=CV_SCORING, refit=refit_metric, cv=cv, n_jobs=-1
    )
    start = time.perf_counter()
    search.fit(X, y)
    n = len(search.cv_results_["params"])
    print(f"{n} combinaciones x {cv.get_n_splits()} folds en {time.perf_counter() - start:.1f} s")
    return search


def tuning_results(search, top=10):
    """Tabla con las mejores combinaciones de hiperparámetros (media y desviación).

    Ordenada por la métrica de refit; los nombres de parámetros se muestran
    sin el prefijo "model__".
    """
    res = pd.DataFrame(search.cv_results_)
    params = [c for c in res.columns if c.startswith("param_")]
    # los hiperparámetros se muestran como texto para que valores como 0.0001
    # no aparezcan como 0.000 con el formato de 3 decimales del notebook
    # None (ej. max_depth sin límite) se muestra como "None" y no como NaN
    out = res[params].apply(lambda col: col.map(str))
    out.columns = [c.replace("param_model__", "").replace("param_", "") for c in params]
    for name in CV_SCORING:
        out[METRIC_LABELS[name]] = res[f"mean_test_{name}"]
    refit_label = METRIC_LABELS[search.refit]
    # desviación muestral (n-1) de los folds, igual que en summarize_cv, para que
    # el mismo modelo muestre la misma desviación en ambas tablas
    # (sklearn guarda en std_test_* la desviación poblacional, que divide por n)
    split_cols = [c for c in res.columns if c.startswith("split") and c.endswith(f"_test_{search.refit}")]
    out[f"desv {refit_label}"] = res[split_cols].std(axis=1, ddof=1)
    out = out.sort_values(refit_label, ascending=False).head(top).reset_index(drop=True)
    out.index = out.index + 1
    out.index.name = "ranking"
    return out


def mlp_training_curves(pipelines: dict, X, y) -> dict:
    """Entrena pipelines con MLP en el train completo y extrae sus curvas.

    Para cada {nombre: pipeline} devuelve un DataFrame por época con:
    - "pérdida (train)": `loss_curve_` de sklearn (log-loss + penalización L2).
    - "accuracy (validación interna)": `validation_scores_`, solo si la red
      usa early stopping (10% del train reservado internamente).
    También guarda el número de épocas entrenadas.
    """
    curves = {}
    for name, pipe in pipelines.items():
        pipe.fit(X, y)
        mlp = pipe.named_steps["model"]
        df = pd.DataFrame({"pérdida (train)": mlp.loss_curve_})
        if getattr(mlp, "validation_scores_", None) is not None:
            df["accuracy (validación interna)"] = mlp.validation_scores_
        df.index = pd.RangeIndex(1, len(df) + 1, name="época")
        curves[name] = df
    return curves


def curves_summary(curves: dict) -> pd.DataFrame:
    """Épocas entrenadas y pérdida inicial/final de cada red."""
    rows = []
    for name, df in curves.items():
        val = "accuracy (validación interna)"
        rows.append(
            {
                "modelo": name,
                "épocas entrenadas": str(len(df)),  # texto: se ve como entero en el notebook
                "pérdida época 1": round(df["pérdida (train)"].iloc[0], 3),
                "pérdida final": round(df["pérdida (train)"].iloc[-1], 3),
                "mejor accuracy validación interna": (
                    f"{df[val].max():.3f}" if val in df else "-"
                ),
            }
        )
    return pd.DataFrame(rows).set_index("modelo")
