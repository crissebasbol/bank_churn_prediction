"""Catálogo de modelos del curso, cada uno dentro de su pipeline completo"""

from sklearn.svm import SVC

from src import config as cfg
from src.data_preparation.imbalance import make_pipeline


def svm_pipeline(kernel="rbf", C=1.0, gamma="scale", degree=3, strategy="class_weight"):
    """SVM (SVC de sklearn) con el kernel indicado y pesos de clase.

    - kernel="linear": frontera de decisión lineal (un hiperplano).
    - kernel="rbf": frontera flexible basada en la similitud entre clientes
      (kernel gaussiano). gamma controla qué tan local es la influencia de cada cliente.
    - kernel="poly": frontera polinomial de grado `degree`.
    - C: penalización de los errores de clasificación. C grande = margen
      estrecho que intenta clasificar bien todo el train (riesgo de sobreajuste);
      C pequeño = margen amplio que tolera más errores.
    """
    model = SVC(kernel=kernel, C=C, gamma=gamma, degree=degree, random_state=cfg.RANDOM_STATE)
    return make_pipeline(model, strategy)


def svm_kernel_models():
    """Los tres kernels con sus parámetros por defecto, para compararlos."""
    return {
        "SVM lineal": svm_pipeline(kernel="linear"),
        "SVM RBF": svm_pipeline(kernel="rbf"),
        "SVM polinomial (grado 3)": svm_pipeline(kernel="poly", degree=3),
    }


# Grilla de búsqueda para SVM RBF
SVM_RBF_GRID = {
    "model__C": [0.1, 1, 10, 100, 1000],
    "model__gamma": [0.001, 0.003, 0.01, 0.03, 0.1, 1],
}


def mlp_pipeline(hidden_layer_sizes=(32,), alpha=1e-4, early_stopping=False,
                 max_iter=1000, strategy="smote"):
    """Red neuronal feed-forward (MLPClassifier de sklearn) con SMOTE.

    - hidden_layer_sizes: neuronas por capa oculta, ej. (32,) = una capa de 32;
      (64, 32) = dos capas de 64 y 32.
    - Activación ReLU en las capas ocultas y sigmoide en la salida (por defecto).
    - Entrenamiento por backpropagation con el optimizador Adam.
    - alpha: regularización L2 (penaliza pesos grandes).
    - early_stopping: separa 10% del train como validación y detiene el
      entrenamiento cuando la métrica en validación deja de mejorar.
    - strategy="smote": MLPClassifier no acepta class_weight (sección 3.4).
    """
    from sklearn.neural_network import MLPClassifier

    model = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes, alpha=alpha,
        early_stopping=early_stopping, max_iter=max_iter,
        random_state=cfg.RANDOM_STATE,
    )
    return make_pipeline(model, strategy)


# Arquitecturas a comparar: de una capa pequeña a dos capas más grandes
MLP_ARCHITECTURES = [(16,), (32,), (64,), (32, 16), (64, 32)]


def mlp_architecture_models(architectures=None):
    """Una red por arquitectura, con los demás parámetros por defecto."""
    architectures = architectures or MLP_ARCHITECTURES
    return {f"MLP {arch}": mlp_pipeline(hidden_layer_sizes=arch) for arch in architectures}


# Grilla de regularización L2 para la red (alpha), en escala logarítmica
MLP_ALPHA_GRID = {"model__alpha": [0.0001, 0.001, 0.01, 0.1, 1, 10]}


# ---------------------------------------------------------------------------
# Ensambles
# ---------------------------------------------------------------------------

def churn_ratio(y):
    """Número de clientes que se quedan por cada churner (para scale_pos_weight)."""
    return float((y == 0).sum() / (y == 1).sum())


def ensemble_models(y_train, n_estimators=100):
    """Cinco ensambles con parámetros por defecto (salvo n_estimators).

    - Bagging: árboles de decisión completos, cada uno entrenado con una muestra
      bootstrap del train; la predicción es el promedio de sus votos. Reduce la
      varianza de un árbol individual. Desbalance: pesos de clase en cada árbol.
    - Random Forest: Bagging + en cada corte del árbol solo se considera un
      subconjunto aleatorio de variables, lo que hace a los árboles más
      distintos entre sí. Desbalance: class_weight="balanced".
    - AdaBoost: árboles de un solo corte (stumps) entrenados en secuencia; cada
      uno da más peso a los clientes que los anteriores clasificaron mal.
      Desbalance: SMOTE (ver `adaboost_imbalance_models`).
    - Gradient Boosting: árboles poco profundos en secuencia; cada uno se ajusta
      al gradiente de la pérdida (el error) que dejaron los anteriores.
      Desbalance: SMOTE (no acepta class_weight).
    - XGBoost: Gradient Boosting optimizado, con regularización L1/L2 en los
      árboles. Desbalance: scale_pos_weight = no churn / churn.
    """
    from sklearn.ensemble import (
        AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier, RandomForestClassifier,
    )
    from sklearn.tree import DecisionTreeClassifier
    from xgboost import XGBClassifier

    rs = cfg.RANDOM_STATE
    return {
        "Bagging": make_pipeline(
            BaggingClassifier(
                estimator=DecisionTreeClassifier(class_weight="balanced", random_state=rs),
                n_estimators=n_estimators, random_state=rs,
            ),
            "none",
        ),
        "Random Forest": make_pipeline(
            RandomForestClassifier(n_estimators=n_estimators, random_state=rs), "class_weight"
        ),
        "AdaBoost": make_pipeline(
            AdaBoostClassifier(
                estimator=DecisionTreeClassifier(max_depth=1, random_state=rs),
                n_estimators=n_estimators, random_state=rs,
            ),
            "smote",  # pesos de clase en los stumps chocan con la reponderación de AdaBoost
        ),
        "Gradient Boosting": make_pipeline(
            GradientBoostingClassifier(n_estimators=n_estimators, random_state=rs), "smote"
        ),
        "XGBoost": make_pipeline(
            XGBClassifier(
                n_estimators=n_estimators, scale_pos_weight=churn_ratio(y_train),
                eval_metric="logloss", random_state=rs, n_jobs=1,
            ),
            "none",
        ),
    }


def adaboost_imbalance_models(n_estimators=100):
    """AdaBoost con tres formas de manejar el desbalance, para compararlas.

    AdaBoost ya repondera a los clientes en cada iteración (más peso a los mal
    clasificados). Si además cada stump usa class_weight="balanced", ese peso
    se multiplica sobre la reponderación de AdaBoost en cada iteración, los
    pesos se desequilibran y el algoritmo se detiene antes de tiempo.
    """
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.tree import DecisionTreeClassifier

    rs = cfg.RANDOM_STATE

    def ada(class_weight=None):
        stump = DecisionTreeClassifier(max_depth=1, class_weight=class_weight, random_state=rs)
        return AdaBoostClassifier(estimator=stump, n_estimators=n_estimators, random_state=rs)

    return {
        "AdaBoost + pesos de clase": make_pipeline(ada("balanced"), "none"),
        "AdaBoost sin tratamiento": make_pipeline(ada(), "none"),
        "AdaBoost + SMOTE": make_pipeline(ada(), "smote"),
    }


def xgboost_pipeline(y_train, add_avg_ticket=True, drop_cols=("avg_open_to_buy",), **params):
    """Pipeline con XGBoost y scale_pos_weight calculado del train.

    add_avg_ticket y drop_cols permiten probar el modelo con y sin variables.
    """
    from xgboost import XGBClassifier

    model = XGBClassifier(
        scale_pos_weight=churn_ratio(y_train), eval_metric="logloss",
        random_state=cfg.RANDOM_STATE, n_jobs=1, **params,
    )
    return make_pipeline(model, "none", add_avg_ticket=add_avg_ticket, drop_cols=drop_cols)


def feature_ablation_models(y_train, **params):
    """XGBoost con y sin las variables candidatas de la Fase 3.

    - avg_ticket: variable creada (monto promedio por transacción).
    - avg_utilization_ratio: cálculo exacto de total_revolving_bal / credit_limit.
    """
    base_drop = ("avg_open_to_buy",)
    return {
        "Con avg_ticket y avg_utilization_ratio": xgboost_pipeline(y_train, **params),
        "Sin avg_ticket": xgboost_pipeline(y_train, add_avg_ticket=False, **params),
        "Sin avg_utilization_ratio": xgboost_pipeline(
            y_train, drop_cols=base_drop + ("avg_utilization_ratio",), **params
        ),
        "Sin ninguna de las dos": xgboost_pipeline(
            y_train, add_avg_ticket=False, drop_cols=base_drop + ("avg_utilization_ratio",), **params
        ),
    }


def random_forest_pipeline(n_estimators=300, **params):
    """Pipeline con Random Forest y pesos de clase."""
    from sklearn.ensemble import RandomForestClassifier

    model = RandomForestClassifier(n_estimators=n_estimators, random_state=cfg.RANDOM_STATE, **params)
    return make_pipeline(model, "class_weight")


# Grillas de búsqueda de los dos mejores ensambles
# XGBoost: se amplió porque con [100, 300] x [3, 5, 7] x [0.05, 0.1, 0.3] el
# mejor resultado quedaba en el borde de la grilla en los tres parámetros.
XGB_GRID = {
    "model__n_estimators": [100, 300, 500],
    "model__max_depth": [2, 3, 4, 5],
    "model__learning_rate": [0.05, 0.1, 0.3, 0.5],
}
RF_GRID = {
    "model__max_depth": [None, 10, 20],
    "model__min_samples_leaf": [1, 3, 5],
    "model__max_features": ["sqrt", 0.5],
}


# Hiperparámetros del modelo final (resultado de GridSearchCV en la sección 4.4).
# Se usan para reentrenar el modelo fuera del notebook (src/deployment/train_model.py).
XGB_BEST_PARAMS = {"learning_rate": 0.3, "max_depth": 3, "n_estimators": 300}
