"""Métricas de evaluación para la clase churn (1)."""

# Métricas que se calculan en validación cruzada (nombres de sklearn).
# Todas se calculan para la clase positiva (churn = 1).
CV_SCORING = {
    "recall": "recall",                  # de los que se van, ¿cuántos detecta el modelo?
    "precision": "precision",            # de los que el modelo marca, ¿cuántos se van de verdad?
    "f1": "f1",                          # media armónica de precision y recall
    "roc_auc": "roc_auc",                # capacidad de ordenar churn por encima de no churn
    "pr_auc": "average_precision",       # área bajo la curva precision-recall (sensible al desbalance)
    "balanced_accuracy": "balanced_accuracy",  # promedio del acierto en cada clase
}

METRIC_LABELS = {
    "recall": "Recall",
    "precision": "Precision",
    "f1": "F1",
    "roc_auc": "ROC-AUC",
    "pr_auc": "PR-AUC",
    "balanced_accuracy": "Accuracy balanceada",
}


def oof_probabilities(pipeline, X, y, cv=None):
    """Probabilidades de churn "out-of-fold" sobre el train.

    Con cross_val_predict cada cliente recibe la probabilidad que le dio el
    modelo entrenado SIN él (cuando estaba en el fold de validación). Así se
    pueden elegir umbrales con el train sin usar el test y sin sobreestimar.
    """
    from sklearn.model_selection import cross_val_predict

    from src.modeling.training import make_cv

    cv = cv or make_cv()
    proba = cross_val_predict(pipeline, X, y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    import pandas as pd

    return pd.Series(proba, index=X.index, name="prob_churn")


def threshold_table(y, proba, thresholds=None):
    """Precision, Recall, F1 y % de clientes marcados para cada umbral."""
    import numpy as np
    import pandas as pd
    from sklearn.metrics import f1_score, precision_score, recall_score

    thresholds = np.round(np.arange(0.05, 0.96, 0.05), 2) if thresholds is None else thresholds
    rows = []
    for t in thresholds:
        pred = (proba >= t).astype(int)
        rows.append(
            {
                "umbral": t,
                "Recall": recall_score(y, pred),
                "Precision": precision_score(y, pred, zero_division=0),
                "F1": f1_score(y, pred),
                "% clientes marcados": 100 * pred.mean(),
                "churners detectados": int(((pred == 1) & (y == 1)).sum()),
                "falsos positivos": int(((pred == 1) & (y == 0)).sum()),
            }
        )
    return pd.DataFrame(rows).set_index("umbral")


def best_f1_threshold(y, proba):
    """Umbral (de 0,01 en 0,01) que maximiza el F1 en las predicciones dadas."""
    import numpy as np

    table = threshold_table(y, proba, thresholds=np.round(np.arange(0.01, 1.0, 0.01), 2))
    best = table["F1"].idxmax()
    # se devuelve como tabla de una fila para que los conteos se vean como enteros
    return float(best), table.loc[[best]]


def evaluate_predictions(y, proba, threshold=0.5):
    """Métricas de un conjunto (ej. test) a partir de probabilidades y un umbral."""
    import pandas as pd
    from sklearn.metrics import (
        accuracy_score, average_precision_score, balanced_accuracy_score,
        f1_score, precision_score, recall_score, roc_auc_score,
    )

    pred = (proba >= threshold).astype(int)
    return pd.Series(
        {
            "Recall": recall_score(y, pred),
            "Precision": precision_score(y, pred),
            "F1": f1_score(y, pred),
            "ROC-AUC": roc_auc_score(y, proba),
            "PR-AUC": average_precision_score(y, proba),
            "Accuracy balanceada": balanced_accuracy_score(y, pred),
            "Accuracy": accuracy_score(y, pred),
        },
        name=f"test (umbral {threshold:.2f})",
    )


def compare_with_cv(test_metrics, oof_y, oof_proba, threshold):
    """Métricas del test al lado de las out-of-fold del train con el mismo umbral.

    Si son parecidas, el modelo generaliza a clientes nuevos (no sobreajustó).
    """
    import pandas as pd

    cv = evaluate_predictions(oof_y, oof_proba, threshold).rename(
        f"validación cruzada (umbral {threshold:.2f})"
    )
    out = pd.concat([cv, test_metrics], axis=1)
    out["diferencia"] = out.iloc[:, 1] - out.iloc[:, 0]
    return out


def confusion_table(y, proba, threshold=0.5):
    """Matriz de confusión con nombres de negocio y conteos."""
    import pandas as pd
    from sklearn.metrics import confusion_matrix

    pred = (proba >= threshold).astype(int)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    return pd.DataFrame(
        cm,
        index=pd.Index(["Real: se queda (0)", "Real: churn (1)"], name=""),
        columns=["Predicho: se queda (0)", "Predicho: churn (1)"],
    )


def decile_table(y, proba, n_groups=10):
    """Tabla de ganancia y lift por deciles de riesgo.

    Se ordenan los clientes de mayor a menor probabilidad y se dividen en
    `n_groups` grupos del mismo tamaño (decil 1 = el 10% de mayor riesgo).
    - tasa de churn: % de churners reales en el decil.
    - % churners capturados (acum.): qué % del total de churners se alcanza
      contactando hasta ese decil.
    - lift: cuántas veces más churn hay en el decil que en la base completa
      (un lift de 5 = 5 veces mejor que elegir clientes al azar).
    """
    import numpy as np
    import pandas as pd

    df = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(proba)})
    df = df.sort_values("p", ascending=False).reset_index(drop=True)
    df["decil"] = np.arange(len(df)) * n_groups // len(df) + 1
    base_rate = df["y"].mean()
    g = df.groupby("decil").agg(
        clientes=("y", "size"), prob_min=("p", "min"), prob_media=("p", "mean"), churners=("y", "sum")
    )
    g["tasa de churn %"] = 100 * g["churners"] / g["clientes"]
    g["% clientes contactados (acum.)"] = 100 * g["clientes"].cumsum() / g["clientes"].sum()
    g["% churners capturados (acum.)"] = 100 * g["churners"].cumsum() / g["churners"].sum()
    g["lift"] = (g["churners"] / g["clientes"]) / base_rate
    return g.rename(columns={"prob_min": "prob. mínima", "prob_media": "prob. media"})
