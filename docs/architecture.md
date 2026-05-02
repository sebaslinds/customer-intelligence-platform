# Customer Intelligence Platform Architecture

This document describes the end-to-end architecture for the Customer Intelligence Platform, from raw Instacart CSV files to deployed analytics, machine learning, and AI insight experiences.

## Architecture Flow

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
        Model Artifact
             |
             v
        FastAPI Prediction Service
             |
             +--> Streamlit Dashboard
             |
             +--> AI Copilot and Decision Engine
```

## Components

| Layer | Component | Responsibility |
| --- | --- | --- |
| Data source | Instacart CSV files | Provides orders, order-product lines, products, aisles, and departments. |
| Ingestion | Python + pandas | Loads raw CSV files in batches and writes them into Snowflake. |
| Warehouse | Snowflake | Stores raw, staging, mart, feature, and analytics tables. |
| Transformations | dbt | Builds staging views, dimensional models, fact tables, feature store, and advanced analytics marts. |
| Data quality | Great Expectations-inspired validation | Checks key constraints before downstream transformations. |
| Machine learning | scikit-learn model comparison | Compares Gradient Boosting, Random Forest, and Logistic Regression for reorder prediction. |
| API | FastAPI on Render | Serves health checks, prediction scoring, decision engine outputs, and AI copilot insights. |
| Rate limiting | Redis with local fallback | Shares API rate limit counters across production instances and keeps in-memory limits for local development. |
| Dashboard | Streamlit Cloud | Displays KPIs, product trends, customer insights, data quality, pipeline health, ML performance, decisions, and copilot responses. |
| AI | OpenAI API, Gemini API, local fallback | Generates structured business insights and decision explanations grounded in Snowflake metrics. |
| CI/CD | GitHub Actions | Runs linting, tests, and dbt validation on code changes. |

## Data Flow

1. Raw CSV files are placed in `data/raw/`.
2. `ingestion/load_data.py` reads each file in batches with pandas.
3. Raw records are loaded into Snowflake analytics tables.
4. dbt creates clean staging models for orders, order products, and products.
5. dbt marts create facts, dimensions, customer cohorts, churn probabilities, lifetime value, and a feature store.
6. The ML training pipeline reads Snowflake features, compares candidate models, and writes model artifacts and evaluation metrics.
7. FastAPI loads the selected model and exposes `/predict`, `/decision`, and `/copilot/insights`.
8. Streamlit reads Snowflake marts and calls the deployed API.
9. The AI copilot and Decision Engine collect metric context, use configured LLM providers when available, and fall back to deterministic explanations when providers are unavailable.

## Snowflake Layers

| Layer | Examples |
| --- | --- |
| Raw | `RAW_INSTACART_ORDERS`, `RAW_INSTACART_PRODUCTS`, `RAW_INSTACART_ORDER_PRODUCTS_TRAIN` |
| Staging | `stg_orders`, `stg_order_products`, `stg_products` |
| Marts | `fct_orders`, `dim_products`, `dim_users` |
| Analytics | `feature_store`, `customer_cohort_retention`, `customer_churn_probability`, `customer_lifetime_value` |

## Deployment Topology

```text
GitHub
  |
  +--> GitHub Actions CI
  |
  +--> Render
  |      |
  |      +--> FastAPI service
  |      +--> Model artifact
  |      +--> Snowflake + OpenAI/Gemini + Redis environment variables
  |      +--> Redis-backed rate limiting
  |
  +--> Streamlit Cloud
         |
         +--> Dashboard UI
         +--> Snowflake environment variables
         +--> API_BASE_URL pointing to Render
```

## Production Notes

- Secrets are not stored in the repository.
- Render, Streamlit Cloud, and GitHub Actions use environment-specific secret stores.
- `REDIS_URL` enables shared API rate limiting in production; local development falls back to in-memory counters.
- The dashboard includes local fallback logic for AI copilot resilience.
- The model artifact is currently committed for portfolio deployment simplicity. A production version should store model artifacts in object storage or a Snowflake stage.
