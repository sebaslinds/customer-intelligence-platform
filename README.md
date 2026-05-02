# Customer Intelligence Platform

An end-to-end data engineering, machine learning, and AI analytics platform built on Instacart-style customer order data.

The platform ingests raw CSV files, models them in Snowflake with dbt, builds customer-level features, trains and compares reorder prediction models, serves predictions with FastAPI, visualizes insights in Streamlit, and generates business recommendations with AI-assisted explanations. The dashboard is deployed publicly, supports English/French language switching, and includes interactive business charts, ML diagnostics, a Decision Engine, and AI insight cards.

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
- ML training pipeline that compares Gradient Boosting, Random Forest, and Logistic Regression
- Reorder prediction model with model card, feature importance, confusion matrix, threshold guidance, and recommendations
- FastAPI service with public `/health` plus protected `/predict`, `/decision`, and `/copilot/insights`
- Decision Engine service with `/decision` for alerts and recommendations
- Streamlit dashboard deployed on Streamlit Cloud
- AI copilot powered by OpenAI/Gemini where configured and grounded in Snowflake metrics
- Deterministic local fallback mode when an AI provider is unavailable
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
        Model Training and Comparison
             |
             v
        FastAPI Prediction Service
             |
             +--> Streamlit Dashboard
             |
             +--> Decision Engine
             |
             +--> AI Business Copilot
```

Detailed documentation:

- [Architecture](docs/architecture.md)
- [Project Walkthrough](docs/project_walkthrough.md)
- [Model Card](docs/model_card.md)
- [Decision Engine](docs/decision_engine.md)

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
| AI | OpenAI API, Gemini API, deterministic fallback |
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
  docs/                   Architecture, walkthrough, model card, and decision engine docs
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

The reorder model is trained on time-aware customer behavioral features from Snowflake. Features are calculated from prior order history before the evaluated order to reduce direct target leakage.

Feature groups:

- Order history: `total_orders`, `customer_tenure_days`, `order_frequency_30d`
- Recency and gaps: `days_since_last_order`, `days_between_orders`, `stddev_days_between_orders`
- Time behavior: `avg_order_dow`, `avg_order_hour_of_day`, `weekend_order_ratio`, `evening_order_ratio`
- Basket coverage: `observed_basket_orders`, `avg_basket_size`
- Product mix: `unique_products`, `unique_departments`, `unique_aisles`, `produce_item_ratio`, `fresh_fruits_item_ratio`, `fresh_vegetables_item_ratio`

The training pipeline benchmarks:

| Model | Accuracy | Balanced Accuracy | Precision | Recall | F1 | ROC AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Gradient Boosting | 0.544 | 0.674 | 0.977 | 0.525 | 0.683 | 0.741 |
| Random Forest | 0.630 | 0.677 | 0.971 | 0.622 | 0.758 | 0.729 |
| Logistic Regression | 0.626 | 0.663 | 0.968 | 0.621 | 0.756 | 0.715 |

Current model artifacts:

- `ml/artifacts/random_forest_reorder_model.joblib`
- `ml/artifacts/training_metrics.json`
- `ml/artifacts/feature_importance.csv`
- `ml/artifacts/model_evaluation_history.csv`
- `ml/artifacts/model_drift_report.json`

Current selected model:

```text
Gradient Boosting
```

Current validation summary:

```text
Accuracy: 0.544
Balanced Accuracy: 0.674
Precision: 0.977
Recall: 0.525
F1: 0.683
ROC AUC: 0.741
Average Precision: 0.974
Brier Score: 0.058
Positive Rate: 93.4%
Recommended Threshold: 0.95
```

The dashboard includes model performance metrics, model comparison, feature importance, a confusion matrix, a metric glossary, model evaluation history, drift status, and business recommendations. Because the dataset is highly imbalanced toward reorder-positive examples, balanced accuracy, ROC AUC, precision, recall, threshold analysis, and monitoring history are more useful than accuracy alone.

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
    "data_quality_failures": 0,
    "api_health": "ok",
    "revenue_proxy_delta": -0.12
  },
  "anomalies": [],
  "use_gemini": true,
  "language": "en"
}
```

The decision engine:

- detects anomalies from incoming metrics
- separates manually simulated business inputs from live operational checks
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
- Decision Engine
- AI Copilot

Recent dashboard improvements:

- bilingual English/French interface
- localized copilot quick prompts and chat input
- revenue proxy explanation, since Instacart does not include product prices
- larger AI response cards that avoid truncated metric values
- clearer priority mix explanation for recommendations
- interactive Altair charts with horizontal or angled x-axis labels
- health checks for Snowflake, Render API, model artifacts, and model metrics
- decision scenario simulator for anomaly-driven alerts and recommendations
- carbon footprint estimates for AI conversations and Decision Engine explanations

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
MODEL_URI=
MODEL_CACHE_DIR=
MODEL_RUN_HISTORY_TABLE=CUSTOMER_INTELLIGENCE.ML_ARTIFACTS.MODEL_TRAINING_RUNS
API_BASE_URL=http://localhost:8000
API_AUTH_ENABLED=false
API_KEY=
API_RATE_LIMIT_PER_MINUTE=60
REDIS_URL=
REDIS_RATE_LIMIT_PREFIX=customer-intelligence:rate-limit
```

Set `API_AUTH_ENABLED=true` and provide `API_KEY` when you want to protect `/predict`,
`/decision`, and `/copilot/insights`. The `/health` endpoint remains public for uptime checks.
When API auth is enabled on Render, add the same `API_KEY` value to Streamlit Cloud secrets.
The dashboard sends it as the `X-API-Key` header for protected API calls.
Set `REDIS_URL` in production to share rate limit counters across API instances. When it is
not configured, the API uses the local in-memory limiter for development and single-instance runs.
Set `MODEL_URI` to a Snowflake stage file path when the API should download the model artifact
at startup instead of reading the local `MODEL_PATH` file.
The training pipeline writes every model run to `MODEL_RUN_HISTORY_TABLE` for production auditability.

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

Each run writes local artifacts and appends a Snowflake record to
`CUSTOMER_INTELLIGENCE.ML_ARTIFACTS.MODEL_TRAINING_RUNS`. To run locally without writing the
Snowflake history table, use:

```bash
python -m ml.train_model --skip-model-run-history
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
MODEL_URI=@CUSTOMER_INTELLIGENCE.ML_ARTIFACTS.MODEL_STAGE/models/random_forest_reorder_model.joblib
API_AUTH_ENABLED=true
API_KEY
API_RATE_LIMIT_PER_MINUTE=60
REDIS_URL
```

When using `MODEL_URI`, grant the API role read access to the model stage:

```sql
GRANT USAGE ON DATABASE CUSTOMER_INTELLIGENCE TO ROLE TRANSFORMER;
GRANT USAGE ON SCHEMA CUSTOMER_INTELLIGENCE.ML_ARTIFACTS TO ROLE TRANSFORMER;
GRANT READ ON STAGE CUSTOMER_INTELLIGENCE.ML_ARTIFACTS.MODEL_STAGE TO ROLE TRANSFORMER;
```

When writing model run history from training, grant table privileges to the training role:

```sql
GRANT CREATE TABLE ON SCHEMA CUSTOMER_INTELLIGENCE.ML_ARTIFACTS TO ROLE TRANSFORMER;
GRANT INSERT, SELECT ON FUTURE TABLES IN SCHEMA CUSTOMER_INTELLIGENCE.ML_ARTIFACTS TO ROLE TRANSFORMER;
```

### Streamlit Cloud

Configuration:

```text
Main file: dashboard/app.py
Python version: 3.11
Dependencies: requirements.txt
API_BASE_URL: deployed Render API URL
API_KEY: same value as Render API_KEY when API_AUTH_ENABLED=true
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
- Protected API routes support `X-API-Key` authentication and per-client rate limiting.
- `REDIS_URL` enables shared rate limiting across Render instances; without it, limits are in-memory per instance.
- Keep `/health` unauthenticated so Render, Streamlit, and monitoring tools can check service status.
- Opening `/decision` directly in a browser sends a `GET` request and returns `Method Not Allowed` by design.
  Use Swagger, curl, or the dashboard button to send a `POST` request with `X-API-Key`.

## Limitations

- Revenue is represented as a proxy because the Instacart dataset does not include product prices.
- The Instacart order history has a cap of 100 observed orders per user, so top-user rankings are less informative than behavioral segmentation.
- Product-level basket coverage is incomplete for many users, so aisle and department features should be interpreted cautiously.
- The model uses historical behavioral features and should not be interpreted as real-time customer intent.
- Airflow and Kafka are included as production-style scaffolding, not hosted services in the current deployment.
- The API can load the model artifact from a Snowflake stage with `MODEL_URI`; the committed artifact remains a local development fallback.
- Training runs are persisted to `MODEL_RUN_HISTORY_TABLE` with metrics, drift findings, feature importance, and model comparison payloads.
- AI responses are grounded in aggregate Snowflake metrics and do not have unrestricted SQL execution.

## Future Improvements

- Add model artifact versioning and promotion metadata
- Surface Snowflake model run history directly in the dashboard
- Tune Redis-backed rate limit thresholds by endpoint and environment
- Add model drift monitoring
- Add SHAP or permutation importance for clearer model explanations
- Persist Decision Engine outputs to Snowflake for auditability
- Add Slack or email alert routing from the Decision Engine
- Add richer dashboard filters by cohort, product department, customer segment, and language-aware insight type
- Add downloadable AI insight reports
- Add screenshots or a short demo GIF to the README
- Deploy Airflow as a managed scheduler
- Add real-time scoring from Kafka events
