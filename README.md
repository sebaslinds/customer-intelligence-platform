# Customer Intelligence Platform

An end-to-end data engineering, machine learning, and AI analytics platform built on Instacart-style customer order data.

The platform ingests raw CSV files, models them in Snowflake with dbt, builds customer-level features, trains a reorder prediction model, serves predictions with FastAPI, visualizes insights in Streamlit, and generates business recommendations with an OpenAI-powered copilot. The dashboard is deployed publicly, supports English/French language switching, and includes interactive business charts and AI insight cards.

## Live Demo

| Service | Link |
| --- | --- |
| Streamlit Dashboard | https://customer-intelligence-platform-d2pmcjetsrlgm2zwep7vgf.streamlit.app/ |
| FastAPI Docs | https://customer-intelligence-platform-3v6q.onrender.com/docs |
| API Health Check | https://customer-intelligence-platform-3v6q.onrender.com/health |

## Business Problem

Retail teams need a reliable way to understand reorder behavior, identify churn risk, monitor retention, and surface product trends from customer purchase history.

This project demonstrates how a modern data stack can convert raw transaction data into:

- executive KPIs
- clean analytics marts
- customer-level machine learning features
- reorder probability predictions
- product and customer insights
- AI-generated business recommendations

## Key Features

- Batch ingestion from raw Instacart CSV files into Snowflake
- Modular dbt project with staging, fact, dimension, and mart models
- Incremental fact model for order processing
- Customer feature store for analytics and machine learning
- Advanced analytics marts for retention, churn probability, and customer lifetime value
- RandomForest classification model for reorder prediction
- FastAPI service with `/health`, `/predict`, and `/copilot/insights`
- Decision Engine service with `/decision` for alerts and recommendations
- Streamlit dashboard deployed on Streamlit Cloud
- AI copilot powered by OpenAI and grounded in Snowflake metrics
- Local fallback mode when OpenAI is unavailable
- English/French dashboard toggle with localized quick prompts and chat UI
- Interactive Altair charts with readable axis labels and tooltips
- AI insight cards for impacted segments, recommendations, priority mix, and follow-up questions
- Pipeline health tab for Snowflake, Render API, model artifact, and training metric checks
- Data quality tab for raw table validation and row count monitoring
- Structured logging, data validation, and CI/CD with GitHub Actions
- Deployment-ready configuration for Render and Streamlit Cloud

## Architecture

```text
Instacart CSV Files
    |
    v
Python Ingestion Pipeline
    |
    v
Snowflake Raw Tables
    |
    v
dbt Staging Models
    |
    v
dbt Mart Models
    |
    +--> Customer and Product Analytics
    |
    +--> Feature Store
             |
             v
        RandomForest Model Training
             |
             v
        FastAPI Prediction Service
             |
             v
        Streamlit Dashboard
             |
             v
        OpenAI Business Copilot
```

Detailed documentation:

- [Architecture](docs/architecture.md)
- [Project Walkthrough](docs/project_walkthrough.md)

## Tech Stack

| Area | Tools |
| --- | --- |
| Language | Python |
| Data Warehouse | Snowflake |
| Transformations | dbt |
| Data Processing | pandas, SQLAlchemy |
| Machine Learning | scikit-learn, joblib |
| API | FastAPI, Pydantic, Uvicorn |
| Dashboard | Streamlit, Altair |
| AI | OpenAI API |
| Decision Intelligence | Rule engine, anomaly detection, Gemini explanations |
| Data Quality | Great Expectations-inspired validation checks |
| Orchestration | Apache Airflow DAG scaffold |
| Streaming | Kafka producer and consumer scaffold |
| CI/CD | GitHub Actions |
| Deployment | Render, Streamlit Cloud |

## Repository Structure

```text
customer-intelligence-platform/
  api/                    FastAPI service and AI copilot
  config/                 Environment settings and logging configuration
  dashboard/              Streamlit dashboard
  data/raw/               Local raw data landing area
  data_quality/           Snowflake data validation checks
  docs/                   Architecture and project walkthrough
  ingestion/              CSV ingestion and Snowflake loading
  ml/                     Feature engineering, training, and model artifacts
  orchestration/airflow/  Airflow DAG for pipeline orchestration
  streaming/              Kafka producer and consumer example
  transformations/dbt/    dbt project for Snowflake models
  utils/                  Shared logging utilities
```

## Data Models

### Staging Models

- `stg_orders`
- `stg_order_products`
- `stg_products`

### Mart Models

- `fct_orders`
- `dim_products`
- `dim_users`
- `feature_store`
- `customer_cohort_retention`
- `customer_churn_probability`
- `customer_lifetime_value`

## Machine Learning

The reorder model is trained on customer behavioral features from Snowflake.

Feature examples:

- `total_orders`
- `avg_basket_size`
- `reorder_ratio`
- `unique_products`
- `days_between_orders`

Current model artifacts:

- `ml/artifacts/random_forest_reorder_model.joblib`
- `ml/artifacts/training_metrics.json`
- `ml/artifacts/feature_importance.csv`

Current validation metrics:

```text
Accuracy: 0.615
Precision: 0.963
Recall: 0.612
F1: 0.748
ROC AUC: 0.682
```

The dashboard includes model performance metrics, feature importance, and a confusion matrix.

## API

### Health Check

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

### Reorder Prediction

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
  "reorder_probability": 0.72,
  "model_version": "random_forest_reorder_model"
}
```

### AI Copilot

```http
POST /copilot/insights
```

Example request:

```json
{
  "question": "Why is churn increasing?"
}
```

The copilot returns:

- AI source indicator
- summary
- explanation
- impacted segments
- recommendations
- detailed insights
- follow-up questions
- charts in the Streamlit dashboard

### Decision Engine

```http
POST /decision
```

Example request:

```json
{
  "data": {
    "reorder_rate": 0.22,
    "churn_rate": 0.61,
    "days_between_orders": 24,
    "data_quality_failures": 0
  },
  "anomalies": [],
  "use_gemini": true
}
```

The decision engine:

- detects anomalies from incoming metrics
- applies business decision rules
- assigns priority levels: `low`, `medium`, `high`, `critical`
- returns alerts and recommendations
- uses Gemini for an executive explanation when `GEMINI_API_KEY` is configured
- falls back to a deterministic local explanation when Gemini is unavailable

## Dashboard

The Streamlit dashboard includes:

- Project Overview
- KPIs
- Customer Insights
- Product Trends
- Data Quality
- Pipeline Health
- Model Performance
- AI Copilot

Recent dashboard improvements:

- bilingual English/French interface
- localized copilot quick prompts and chat input
- revenue proxy explanation, since Instacart does not include product prices
- larger AI response cards that avoid truncated metric values
- clearer priority mix explanation for recommendations
- interactive Altair charts with horizontal or angled x-axis labels
- health checks for Snowflake, Render API, model artifacts, and model metrics

The AI Copilot tab supports questions such as:

```text
Why is churn increasing?
Which products drive repeat purchases?
Which customer segment has the longest days between orders?
How does retention change by cohort period?
How can we improve retention?
```

French examples are also supported:

```text
Pourquoi le churn augmente-t-il?
Quels produits generent le plus de rachats?
Quels segments clients sont a risque de churn?
Comment pouvons-nous ameliorer la retention?
```

Copilot responses include:

- AI source indicator, showing whether the answer came from OpenAI or local fallback logic
- concise summary and business explanation
- impacted segment cards
- metric comparison chart
- recommendation cards
- priority mix explanation
- detailed evidence and follow-up questions

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
OPENAI_MODEL=gpt-4o-mini
GEMINI_API_KEY=
GEMINI_MODEL=gemini-1.5-flash
MODEL_PATH=ml/artifacts/random_forest_reorder_model.joblib
API_BASE_URL=http://localhost:8000
```

## Running Locally

Load raw data:

```bash
python -m ingestion.load_data
```

Run dbt:

```bash
cd transformations/dbt
dbt debug --profiles-dir .
dbt run --profiles-dir .
dbt test --profiles-dir .
```

Run data validation:

```bash
python -m data_quality.validation
```

Train the model:

```bash
python -m ml.train_model
```

Run the API:

```bash
uvicorn api.main:app --reload
```

Run the dashboard:

```bash
streamlit run dashboard/app.py
```

## Deployment

### Render API

The FastAPI service is deployed on Render using `Dockerfile` and `render.yaml`.

Required production environment variables:

```text
APP_ENV=production
LOG_LEVEL=INFO
SNOWFLAKE_ACCOUNT
SNOWFLAKE_USER
SNOWFLAKE_PASSWORD
SNOWFLAKE_ROLE
SNOWFLAKE_WAREHOUSE
SNOWFLAKE_DATABASE
SNOWFLAKE_SCHEMA
OPENAI_API_KEY
OPENAI_MODEL
MODEL_PATH=ml/artifacts/random_forest_reorder_model.joblib
```

### Streamlit Cloud

Configuration:

```text
Main file: dashboard/app.py
Python version: 3.11
Dependencies: requirements.txt
API_BASE_URL: deployed Render API URL
```

## CI/CD

GitHub Actions runs on push and pull requests:

- install dependencies
- lint Python code with Ruff
- run Python tests
- validate the dbt project

Workflow file:

```text
.github/workflows/pipeline.yml
```

## Security Notes

- Never commit `.env` files or credentials.
- Store production secrets in Render, Streamlit Cloud, and GitHub repository secrets.
- Rotate Snowflake, OpenAI, and Kaggle credentials if they are exposed in screenshots, logs, or chat.
- Keep raw data out of Git unless it is intentionally sampled and safe to share.

## Limitations

- Revenue is represented as a proxy because the Instacart dataset does not include product prices.
- The model uses historical behavioral features and should not be interpreted as real-time customer intent.
- Airflow and Kafka are included as production-style scaffolding, not hosted services in the current deployment.
- The model artifact is committed for portfolio deployment simplicity. A production system should use object storage or a model registry.
- OpenAI responses are grounded in aggregate Snowflake metrics and do not have unrestricted SQL execution.

## Future Improvements

- Store model artifacts in Snowflake stage, S3, or a model registry
- Add API authentication and rate limiting
- Add historical model evaluation tracking
- Add model drift monitoring
- Add richer dashboard filters by cohort, product department, customer segment, and language-aware insight type
- Add downloadable AI insight reports
- Add screenshots or a short demo GIF to the README
- Deploy Airflow as a managed scheduler
- Add real-time scoring from Kafka events
