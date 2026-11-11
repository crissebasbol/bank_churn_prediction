# Predicción de Churn en un Banco

Caso de negocio de Métodos de Analítica II (Pontificia Universidad Javeriana, 2026-3).
Modelo predictivo para identificar los clientes con mayor probabilidad de retirarse del banco y priorizar la campaña de beneficios de retención.

## Estructura

- `bank_churn.xlsx`: datos originales (10.127 clientes, 2021; churn a cierre 2022).
- `notebooks/churn_analysis.ipynb`: notebook principal (interacción, gráficos y conclusiones).
- `src/`: funciones y clases de utilidad que usa el notebook, organizadas por fase de CRISP-DM:
  - `data_understanding/`: carga, calidad de datos y EDA.
  - `data_preparation/`: partición, variables y pipelines.
  - `modeling/`, `evaluation/`, `deployment/`: modelado, evaluación (incluye Montecarlo) y scoring.
- `reports/figures/`: figuras exportadas para el documento y la presentación.
- `models/`: modelos entrenados (no versionados).

## Cómo ejecutar

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook notebooks/churn_analysis.ipynb
```
