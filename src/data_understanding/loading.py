"""Carga de los datos y creación del target."""

from __future__ import annotations

import pandas as pd

from src import config as cfg


def load_raw(path=cfg.DATA_PATH) -> pd.DataFrame:
    """Lee el archivo original sin transformaciones."""
    return pd.read_excel(path)


def prepare_base(df: pd.DataFrame) -> pd.DataFrame:
    """Tipado básico y creación del target binario (1 = churn).

    No elimina ni imputa nada: las decisiones de preprocesamiento
    se toman en `features.py` dentro de pipelines.
    """
    out = df.copy()
    out[cfg.TARGET] = (out[cfg.TARGET_RAW] == cfg.POSITIVE_LABEL).astype(int)
    out = out.drop(columns=[cfg.TARGET_RAW])
    for col in cfg.CATEGORICAL_COLS:
        out[col] = out[col].astype("category")
    return out.set_index(cfg.ID_COL)


def load_data(path=cfg.DATA_PATH) -> pd.DataFrame:
    """Carga los datos listos para EDA y modelado."""
    return prepare_base(load_raw(path))
