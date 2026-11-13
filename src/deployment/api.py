"""API local con FastAPI para predecir el churn y decidir si vale la pena retener.

Uso:
    python -m src.deployment.train_model          # solo si no existe models/churn_xgboost.joblib
    uvicorn src.deployment.api:app --reload
Luego abrir http://localhost:8000 (interfaz) o http://localhost:8000/docs (documentación de la API).
"""

from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from src import config as cfg
from src.deployment.decision import retention_decision
from src.deployment.examples import load_examples
from src.deployment.scoring import DEFAULT_MODEL_PATH, load_model

STATIC_DIR = Path(__file__).parent / "static"
UNKNOWN = (cfg.UNKNOWN,)

Education = Literal[tuple(cfg.ORDINAL_ORDER["education_level"]) + UNKNOWN]
Income = Literal[tuple(cfg.ORDINAL_ORDER["income_category"]) + UNKNOWN]
Card = Literal[tuple(cfg.ORDINAL_ORDER["card_category"])]


class Client(BaseModel):
    """Datos de un cliente (mismas variables de bank_churn.xlsx).

    avg_open_to_buy y avg_utilization_ratio no se piden: se calculan a partir
    de credit_limit y total_revolving_bal.
    """
    gender: Literal["F", "M"]
    education_level: Education
    marital_status: Literal["Married", "Single", "Divorced", "Unknown"]
    income_category: Income
    card_category: Card
    customer_age: int = Field(ge=18, le=100)
    dependent_count: int = Field(ge=0, le=10)
    months_on_book: int = Field(ge=0, le=120)
    total_relationship_count: int = Field(ge=1, le=10)
    months_inactive_12_mon: int = Field(ge=0, le=12)
    contacts_count_12_mon: int = Field(ge=0, le=12)
    credit_limit: float = Field(gt=0)
    total_revolving_bal: float = Field(ge=0)
    total_amt_chng_q4_q1: float = Field(ge=0)
    total_trans_amt: float = Field(ge=0)
    total_trans_ct: int = Field(ge=1)
    total_ct_chng_q4_q1: float = Field(ge=0)


app = FastAPI(
    title="Predicción de churn - Banco",
    description="Probabilidad de churn y recomendación de retención por cliente.",
)
_model = None


def get_model():
    global _model
    if _model is None:
        if not DEFAULT_MODEL_PATH.exists():
            raise HTTPException(
                status_code=503,
                detail="No hay modelo entrenado. Ejecuta: python -m src.deployment.train_model",
            )
        _model = load_model()
    return _model


def to_frame(client: Client) -> pd.DataFrame:
    """Convierte los datos recibidos al formato que espera el pipeline."""
    df = pd.DataFrame([client.model_dump()])
    for col in cfg.CATEGORICAL_COLS:
        df[col] = df[col].astype("category")
    if client.total_revolving_bal > client.credit_limit:
        raise HTTPException(status_code=422, detail="total_revolving_bal no puede ser mayor que credit_limit")
    return df


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/examples")
def examples():
    """Clientes reales del test precargados como ejemplo."""
    return load_examples()


@app.post("/predict")
def predict(client: Client):
    """Probabilidad de churn, valor estimado y si vale la pena retener al cliente."""
    return retention_decision(get_model(), to_frame(client))[0]
