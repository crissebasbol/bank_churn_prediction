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
