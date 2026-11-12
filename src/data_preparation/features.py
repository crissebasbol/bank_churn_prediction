"""Ingeniería de variables y preprocesamiento (pipelines de sklearn)."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config as cfg

# Categóricas nominales y la categoría de referencia que se elimina en one-hot
NOMINAL_REFERENCE = {"gender": "M", "marital_status": "Married"}


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Aplica las decisiones del EDA sobre las variables."""

    def __init__(self, drop_cols=("avg_open_to_buy",), add_avg_ticket=True):
        self.drop_cols = drop_cols
        self.add_avg_ticket = add_avg_ticket

    def fit(self, X, y=None):
        # se guardan las columnas de entrada para get_feature_names_out
        self.feature_names_in_ = np.array(X.columns, dtype=object)
        return self

    def transform(self, X):
        out = X.copy()
        if self.add_avg_ticket:
            out["avg_ticket"] = out["total_trans_amt"] / out["total_trans_ct"]
        cols_to_drop = [c for c in (self.drop_cols or []) if c in out.columns]
        return out.drop(columns=cols_to_drop)

    def get_feature_names_out(self, input_features=None):
        cols = list(input_features if input_features is not None else self.feature_names_in_)
        if self.add_avg_ticket:
            cols.append("avg_ticket")
        return np.array([c for c in cols if c not in (self.drop_cols or [])], dtype=object)


def column_changes(X_before: pd.DataFrame, X_after: pd.DataFrame) -> pd.Series:
    """Resumen de columnas eliminadas y creadas por una transformación."""
    removed = [c for c in X_before.columns if c not in X_after.columns]
    added = [c for c in X_after.columns if c not in X_before.columns]
    return pd.Series(
        {
            "columnas antes": X_before.shape[1],
            "columnas después": X_after.shape[1],
            "eliminadas": ", ".join(removed) or "-",
            "creadas": ", ".join(added) or "-",
        },
        name="valor",
        dtype=object,
    )


class OrdinalWithUnknown(BaseEstimator, TransformerMixin):
    """Codificación ordinal que maneja la categoría 'Unknown'."""

    def __init__(self, order=None, unknown=cfg.UNKNOWN):
        self.order = order
        self.unknown = unknown

    def fit(self, X, y=None):
        order = self.order or cfg.ORDINAL_ORDER
        self.columns_ = list(X.columns)
        self.medians_ = {}
        self.has_unknown_ = {}
        for col in self.columns_:
            codes = X[col].astype(str).map({c: i for i, c in enumerate(order[col])})
            self.medians_[col] = float(np.median(codes.dropna()))
            self.has_unknown_[col] = bool((X[col].astype(str) == self.unknown).any())
        return self

    def transform(self, X):
        order = self.order or cfg.ORDINAL_ORDER
        out = pd.DataFrame(index=X.index)
        for col in self.columns_:
            values = X[col].astype(str)
            codes = values.map({c: i for i, c in enumerate(order[col])})
            if codes[values != self.unknown].isna().any():
                raise ValueError(f"Categoría no reconocida en {col}")
            out[col] = codes.fillna(self.medians_[col]).astype(float)
            if self.has_unknown_[col]:
                out[f"{col}_unknown"] = (values == self.unknown).astype(int)
        return out

    def get_feature_names_out(self, input_features=None):
        names = []
        for col in self.columns_:
            names.append(col)
            if self.has_unknown_[col]:
                names.append(f"{col}_unknown")
        return np.array(names, dtype=object)


def numeric_features(add_avg_ticket=True, drop_cols=("avg_open_to_buy",)):
    """Lista de variables numéricas que salen de FeatureEngineer."""
    cols = [c for c in cfg.NUMERIC_COLS if c not in (drop_cols or [])]
    return cols + (["avg_ticket"] if add_avg_ticket else [])


def build_preprocessor(add_avg_ticket=True, drop_cols=("avg_open_to_buy",)):
    """Pipeline completo de preprocesamiento: variables + codificación + escalado.

    1. FeatureEngineer: elimina redundantes y crea avg_ticket.
    2. ColumnTransformer:
       - numéricas -> StandardScaler (media 0, desviación 1).
       - ordinales (education, income, card) -> OrdinalWithUnknown + StandardScaler
         sobre el código ordinal (los indicadores _unknown quedan en 0/1).
       - nominales (gender, marital) -> one-hot eliminando la categoría de
         referencia (M y Married) para evitar redundancia.

    Todo se ajusta con fit sobre el train y se aplica con transform al test.
    """
    ordinal_cols = list(cfg.ORDINAL_ORDER)
    nominal_cols = list(NOMINAL_REFERENCE)

    ordinal_pipe = Pipeline(
        [
            ("ordinal", OrdinalWithUnknown()),
            ("scale", ColumnTransformer(
                [("scale_codes", StandardScaler(), ordinal_cols)],
                remainder="passthrough", verbose_feature_names_out=False,
            )),
        ]
    )
    encoder = ColumnTransformer(
        [
            ("num", StandardScaler(), numeric_features(add_avg_ticket, drop_cols)),
            ("ord", ordinal_pipe, ordinal_cols),
            ("nom", OneHotEncoder(drop=[NOMINAL_REFERENCE[c] for c in nominal_cols],
                                  sparse_output=False), nominal_cols),
        ],
        verbose_feature_names_out=False,
    )
    pipe = Pipeline(
        [
            ("features", FeatureEngineer(drop_cols=drop_cols, add_avg_ticket=add_avg_ticket)),
            ("encode_scale", encoder),
        ]
    )
    return pipe.set_output(transform="pandas")


def preprocessing_summary(X_processed: pd.DataFrame) -> pd.DataFrame:
    """Media, desviación, mínimo, máximo y valores únicos de cada columna procesada."""
    return pd.DataFrame(
        {
            "media": X_processed.mean(),
            "desv. est.": X_processed.std(),
            "mín": X_processed.min(),
            "máx": X_processed.max(),
            "valores únicos": X_processed.nunique(),
        }
    ).round(3)
