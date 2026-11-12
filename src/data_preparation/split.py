"""Partición estratificada train/test."""

import pandas as pd
from sklearn.model_selection import train_test_split

from src import config as cfg


def split_data(df: pd.DataFrame, test_size=cfg.TEST_SIZE, random_state=cfg.RANDOM_STATE):
    """Partición estratificada train/test. Devuelve X_train, X_test, y_train, y_test."""
    X = df.drop(columns=[cfg.TARGET])
    y = df[cfg.TARGET]
    return train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )


def split_summary(y_train: pd.Series, y_test: pd.Series) -> pd.DataFrame:
    """Tamaño, % del total y tasa de churn de cada conjunto."""
    total = len(y_train) + len(y_test)
    rows = []
    for name, y in [("train", y_train), ("test", y_test)]:
        rows.append(
            {
                "conjunto": name,
                "clientes": len(y),
                "% del total": round(100 * len(y) / total, 1),
                "churn (1)": int(y.sum()),
                "se queda (0)": int((y == 0).sum()),
                "churn %": round(100 * y.mean(), 2),
            }
        )
    return pd.DataFrame(rows).set_index("conjunto")
