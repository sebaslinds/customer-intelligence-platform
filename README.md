# Customer Intelligence Platform

Production-ready customer analytics platform built with Python, Snowflake, dbt, FastAPI, Streamlit, and machine learning.

The project ingests Instacart-style customer order data, models it in Snowflake with dbt, builds reusable customer features, trains a reorder prediction model, exposes predictions through an API, and visualizes business KPIs in a live dashboard.

## Live Demo

- Streamlit dashboard: https://customer-intelligence-platform-d2pmcjetsrlgm2zwep7vgf.streamlit.app/
- Render API health check: https://customer-intelligence-platform-3v6q.onrender.com/health
- Render API docs: https://customer-intelligence-platform-3v6q.onrender.com/docs

## Highlights

- End-to-end data engineering workflow from raw CSV ingestion to analytics marts
- Snowflake-backed warehouse architecture
- dbt staging and mart models with data tests
- Feature engineering layer for customer-level ML features
- Random Forest reorder prediction model
- FastAPI prediction and AI copilot endpoints
- Streamlit dashboard connected directly to Snowflake
- Dashboard chatbot for AI-generated business insights
- Structured logging and pipeline metrics
- GitHub Actions CI for linting, tests, and dbt validation
- Deployment-ready configuration for Render and Streamlit Cloud

## Architecture

```text
Raw Instacart CSVs
        |
        v
Python ingestion pipeline
        |
        v
Snowflake raw tables
        |
        v
dbt staging models
        |
        v
dbt mart models
        |
        +--> Streamlit dashboard
        |
        +--> Feature store
                 |
                 v
            ML training
                 |
                 v
            FastAPI prediction service
```

## Repository Structure

```text
customer-intelligence-platform/
  api/                    FastAPI service and AI copilot
  config/                 Environment settings and logging configuration
  dashboard/              Streamlit dashboard
  data/raw/               Local raw data landing area
  data_quality/           Snowflake data validation checks
  ingestion/              CSV ingestion and Snowflake loading
  ml/                     Feature engineering, training, and model artifacts
  orchestration/airflow/  Airflow DAG for pipeline orchestration
  streaming/              Kafka producer and consumer example
  transformations/dbt/    dbt project for Snowflake models
  utils/                  Shared logging utilities
```

## Tech Stack

- Python
- Snowflake
- dbt
- FastAPI
- Streamlit
- scikit-learn
- pandas
- SQLAlchemy
- Apache Airflow
- Kafka
- GitHub Actions
- Render
- Streamlit Cloud

## Data Models

### Staging

- `stg_orders`
- `stg_order_products`
- `stg_products`

### Marts

- `fct_orders`
- `dim_products`
- `dim_users`
- `feature_store`
- `customer_cohort_retention`
- `customer_churn_probability`
- `customer_lifetime_value`

## API Endpoints

### Health

```http
GET /health
```

Example response:

```json
{
  "status": "ok",
  "environment": "production",
  "model_loaded": true
}
```

### Prediction

```http
POST /predict
```

Example request:

```json
{
  "total_orders": 10,
  "avg_basket_size": 8,
  "reorder_ratio": 0.35,
  "unique_products": 25,
  "days_between_orders": 7
}
```

Example response:

```json
{
  "reorder_probability": 1,
  "model_version": "random_forest_reorder_model"
}
```

### AI Copilot

```http
POST /copilot/insights
```

Generates structured business insights from Snowflake metrics, with a local fallback mode if the OpenAI API is unavailable.

The Streamlit dashboard includes an `AI Copilot` tab that calls this endpoint and renders explanations, impacted segments, recommendations, detailed insights, and follow-up questions.

## Machine Learning

The reorder model is trained on a time-aware dataset to reduce target leakage. For each training row, customer features are calculated from prior orders only, while the label indicates whether the next target order contains at least one reordered item.

Current model artifacts:

- Model: `ml/artifacts/random_forest_reorder_model.joblib`
- Metrics: `ml/artifacts/training_metrics.json`
- Feature importance: `ml/artifacts/feature_importance.csv`

Current validation metrics:

```text
Accuracy: 0.615
Precision: 0.963
Recall: 0.612
F1: 0.748
ROC AUC: 0.682
```

## Local Setup

### 1. Create a virtual environment

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Optional workflow-specific dependencies:

```bash
pip install -r requirements-dbt.txt
pip install -r requirements-ingestion.txt
pip install -r requirements-streaming.txt
pip install -r requirements-data-quality.txt
```

### 3. Configure environment variables

Copy the example file:

```bash
cp .env.example .env
```

Required Snowflake variables:

```text
SNOWFLAKE_ACCOUNT=
SNOWFLAKE_USER=
SNOWFLAKE_PASSWORD=
SNOWFLAKE_ROLE=
SNOWFLAKE_WAREHOUSE=
SNOWFLAKE_DATABASE=
SNOWFLAKE_SCHEMA=
```

Optional:

```text
OPENAI_API_KEY=
MODEL_PATH=ml/artifacts/random_forest_reorder_model.joblib
API_BASE_URL=http://localhost:8000
```

## Running Locally

### Load raw data

```bash
python -m ingestion.load_data
```

### Run dbt

```bash
cd transformations/dbt
dbt debug --profiles-dir .
dbt run --profiles-dir .
dbt test --profiles-dir .
```

### Run data validation

```bash
python -m data_quality.validation
```

### Train the model

```bash
python -m ml.train_model
```

### Run the API

```bash
uvicorn api.main:app --reload
```

### Run the dashboard

```bash
streamlit run dashboard/app.py
```

## Deployment

### Render API

The FastAPI service is deployed on Render using `Dockerfile` and `render.yaml`.

Required environment variables:

- `APP_ENV=production`
- `LOG_LEVEL=INFO`
- `SNOWFLAKE_ACCOUNT`
- `SNOWFLAKE_USER`
- `SNOWFLAKE_PASSWORD`
- `SNOWFLAKE_ROLE`
- `SNOWFLAKE_WAREHOUSE`
- `SNOWFLAKE_DATABASE`
- `SNOWFLAKE_SCHEMA`
- `OPENAI_API_KEY`
- `MODEL_PATH=ml/artifacts/random_forest_reorder_model.joblib`

### Streamlit Cloud

The dashboard is deployed on Streamlit Cloud.

Configuration:

- Main file path: `dashboard/app.py`
- Python version: `3.11`
- Dependencies: `dashboard/requirements.txt`
- Secrets: values from `.streamlit/secrets.toml.example`
- `API_BASE_URL` must point to the deployed Render API for the AI Copilot tab

## CI/CD

GitHub Actions runs on push and pull requests:

- Install dependencies
- Lint Python code with Ruff
- Run Python tests
- Validate the dbt project

Workflow file:

```text
.github/workflows/pipeline.yml
```

## Security Notes

- Never commit `.env` files or credentials.
- Rotate Snowflake credentials if they are exposed in screenshots, logs, or chat.
- Store production secrets in Render, Streamlit Cloud, and GitHub repository secrets.
- Keep raw data out of Git unless it is intentionally sampled and anonymized.

## Portfolio Value

This project demonstrates practical production data engineering skills:

- Warehouse-first modeling with Snowflake and dbt
- Python ingestion with batching, logging, and error handling
- ML feature engineering and model serving
- API and dashboard deployment
- CI/CD and deployment configuration
- Data validation and operational monitoring patterns

## Future Improvements

- Store trained model artifacts in Snowflake stage or object storage instead of Git
- Add API authentication
- Add historical model evaluation tracking
- Add richer dashboard filters and drill-downs
- Add Airflow deployment with a managed scheduler
