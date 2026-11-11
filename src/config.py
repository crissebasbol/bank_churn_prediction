"""Configuración global del proyecto: rutas, semilla y definición de columnas."""

from pathlib import Path

# Rutas
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "bank_churn.xlsx"
FIGURES_DIR = ROOT_DIR / "reports" / "figures"
MODELS_DIR = ROOT_DIR / "models"

# Reproducibilidad
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

# Variables
ID_COL = "clientnum"
TARGET_RAW = "attrition_flag"
TARGET = "churn"  # 1 = Attrited Customer, 0 = Existing Customer
POSITIVE_LABEL = "Attrited Customer"

CATEGORICAL_COLS = [
    "gender",
    "education_level",
    "marital_status",
    "income_category",
    "card_category",
]

NUMERIC_COLS = [
    "customer_age",
    "dependent_count",
    "months_on_book",
    "total_relationship_count",
    "months_inactive_12_mon",
    "contacts_count_12_mon",
    "credit_limit",
    "total_revolving_bal",
    "avg_open_to_buy",
    "total_amt_chng_q4_q1",
    "total_trans_amt",
    "total_trans_ct",
    "total_ct_chng_q4_q1",
    "avg_utilization_ratio",
]

# Orden natural de las categorías ordinales (Unknown se trata aparte)
ORDINAL_ORDER = {
    "education_level": [
        "Uneducated",
        "High School",
        "College",
        "Graduate",
        "Post-Graduate",
        "Doctorate",
    ],
    "income_category": [
        "Less than $40K",
        "$40K - $60K",
        "$60K - $80K",
        "$80K - $120K",
        "$120K +",
    ],
    "card_category": ["Blue", "Silver", "Gold", "Platinum"],
}

UNKNOWN = "Unknown"

# Descripción de variables (Tabla 1 del caso), usada en tablas del notebook
VARIABLE_DESCRIPTIONS = {
    "clientnum": "Identificador único del cliente",
    "attrition_flag": "Variable objetivo: el cliente se fue a otro banco o no",
    "customer_age": "Edad del cliente",
    "gender": "Sexo del cliente",
    "dependent_count": "Número de personas económicamente dependientes",
    "education_level": "Nivel educativo",
    "marital_status": "Estado civil",
    "income_category": "Categoría de ingresos",
    "card_category": "Categoría de tarjeta",
    "months_on_book": "Antigüedad del cliente (meses)",
    "total_relationship_count": "Total de productos con el banco",
    "months_inactive_12_mon": "Meses inactivo en el último año",
    "contacts_count_12_mon": "Contactos con el cliente en el último año",
    "credit_limit": "Cupo de crédito",
    "total_revolving_bal": "Balance de crédito rotativo",
    "avg_open_to_buy": "Promedio de cupo disponible en tarjetas",
    "total_amt_chng_q4_q1": "Cambio en valor de transacciones Q4 vs Q1",
    "total_trans_amt": "Valor total de transacciones",
    "total_trans_ct": "Cantidad de transacciones",
    "total_ct_chng_q4_q1": "Cambio en cantidad de transacciones Q4 vs Q1",
    "avg_utilization_ratio": "Razón de utilización de la tarjeta",
}
