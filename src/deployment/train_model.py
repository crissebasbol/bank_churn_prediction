"""Entrena el modelo final y lo guarda, para usar la app sin ejecutar el notebook.

Uso:  python -m src.deployment.train_model
"""

from src.data_preparation.split import split_data
from src.data_understanding.loading import load_data
from src.deployment.scoring import save_model
from src.modeling.models import XGB_BEST_PARAMS, xgboost_pipeline


def main():
    X_train, X_test, y_train, y_test = split_data(load_data())
    model = xgboost_pipeline(y_train, **XGB_BEST_PARAMS).fit(X_train, y_train)
    path = save_model(model)
    print(f"Modelo entrenado con {len(X_train):,} clientes y guardado en {path}")


if __name__ == "__main__":
    main()
