"""Carga, validación y partición de los datos."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

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


def data_dictionary(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Tabla resumen por variable: tipo, nulos, únicos, ejemplo y descripción."""
    rows = []
    for col in df_raw.columns:
        s = df_raw[col]
        rows.append(
            {
                "variable": col,
                "descripcion": cfg.VARIABLE_DESCRIPTIONS.get(col, ""),
                "tipo": str(s.dtype),
                "nulos": int(s.isna().sum()),
                "unicos": int(s.nunique()),
                "ejemplo": s.iloc[0],
            }
        )
    return pd.DataFrame(rows).set_index("variable")


def quality_report(df_raw: pd.DataFrame) -> pd.Series:
    """Chequeos de calidad generales de la base."""
    unknown_cells = (df_raw[cfg.CATEGORICAL_COLS] == cfg.UNKNOWN).sum().sum()
    rows_with_unknown = (df_raw[cfg.CATEGORICAL_COLS] == cfg.UNKNOWN).any(axis=1).sum()
    return pd.Series(
        {
            "filas": len(df_raw),
            "columnas": df_raw.shape[1],
            "celdas nulas": int(df_raw.isna().sum().sum()),
            "filas duplicadas": int(df_raw.duplicated().sum()),
            "clientnum único": bool(df_raw[cfg.ID_COL].is_unique),
            "celdas 'Unknown'": int(unknown_cells),
            "filas con algún 'Unknown'": int(rows_with_unknown),
            "% filas con 'Unknown'": round(100 * rows_with_unknown / len(df_raw), 2),
        },
        name="valor",
    )


def unknown_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Cantidad de 'Unknown' por variable categórica y tasa de churn asociada."""
    rows = []
    for col in cfg.CATEGORICAL_COLS:
        mask = df[col] == cfg.UNKNOWN
        n = int(mask.sum())
        if n == 0:
            continue
        rows.append(
            {
                "variable": col,
                "n_unknown": n,
                "% registros": round(100 * n / len(df), 2),
                "churn % (Unknown)": round(100 * df.loc[mask, cfg.TARGET].mean(), 2),
                "churn % (resto)": round(100 * df.loc[~mask, cfg.TARGET].mean(), 2),
            }
        )
    return pd.DataFrame(rows).set_index("variable")


def target_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Conteo y proporción de la variable objetivo."""
    counts = df[cfg.TARGET].value_counts().sort_index()
    return pd.DataFrame(
        {
            "clase": ["Existing (0)", "Attrited (1)"],
            "n": counts.values,
            "%": (100 * counts / counts.sum()).round(2).values,
        }
    ).set_index("clase")


def split_data(df: pd.DataFrame, test_size=cfg.TEST_SIZE, random_state=cfg.RANDOM_STATE):
    """Partición estratificada train/test. Devuelve X_train, X_test, y_train, y_test."""
    X = df.drop(columns=[cfg.TARGET])
    y = df[cfg.TARGET]
    return train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
