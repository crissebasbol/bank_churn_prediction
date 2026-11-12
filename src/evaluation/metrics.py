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
