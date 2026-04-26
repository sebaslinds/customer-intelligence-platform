# Project Walkthrough

## 1. Problem Statement

Retail teams need a practical way to understand customer reorder behavior, identify churn risk, monitor retention, and turn operational data into business recommendations. This project demonstrates an end-to-end customer intelligence workflow using Instacart-style market basket data.

## 2. Data Source

The platform uses Instacart dataset CSV files:

- `orders.csv`
- `order_products__train.csv`
- `products.csv`
- `aisles.csv`
- `departments.csv`

The dataset provides order history and product metadata. It does not include actual prices, so revenue is represented with a proxy based on item counts.

## 3. Data Ingestion

The ingestion layer is implemented in Python with pandas and Snowflake support.

Key characteristics:

- Batch loading with chunked CSV reads
- Environment-driven Snowflake configuration
- Structured logging
- Reusable loading functions
- Error handling around file and database operations

Main file:

```text
ingestion/load_data.py
```

## 4. Snowflake Warehouse

Snowflake acts as the central warehouse for raw data, transformed models, analytics marts, and feature tables.

The project uses the configured Snowflake database and schema from environment variables:

```text
SNOWFLAKE_DATABASE
SNOWFLAKE_SCHEMA
```

The raw layer stores data loaded directly from CSV files. Downstream dbt models clean, join, and reshape the data for analytics and machine learning.

## 5. dbt Transformations

The dbt project lives in:

```text
transformations/dbt/
```

The model structure includes:

```text
models/
  staging/
  marts/
```

Staging models:

- `stg_orders`
- `stg_order_products`
- `stg_products`

Mart models:

- `fct_orders`
- `dim_products`
- `dim_users`
- `feature_store`
- `customer_cohort_retention`
- `customer_churn_probability`
- `customer_lifetime_value`

dbt tests validate important fields with uniqueness and not-null checks.

## 6. Feature Engineering

The feature store creates customer-level behavior features used for machine learning and dashboard analysis.

Example features:

- `total_orders`
- `avg_basket_size`
- `reorder_ratio`
- `unique_products`
- `days_between_orders`

The feature store is saved in Snowflake so the same feature definitions can support ML training, API scoring, dashboard reporting, and AI context.

## 7. Machine Learning

The ML pipeline trains a RandomForest classifier to predict customer reorder behavior.

Main file:

```text
ml/train_model.py
```

The training pipeline:

1. Loads the feature store from Snowflake.
2. Builds a classification dataset.
3. Splits training and test data.
4. Trains a scikit-learn RandomForest model.
5. Evaluates accuracy, precision, recall, F1, ROC AUC, and confusion matrix.
6. Saves the model artifact and metrics locally.

Artifacts:

```text
ml/artifacts/random_forest_reorder_model.joblib
ml/artifacts/training_metrics.json
ml/artifacts/feature_importance.csv
```

## 8. API Layer

FastAPI exposes the serving layer.

Main file:

```text
api/main.py
```

Endpoints:

- `GET /health`
- `POST /predict`
- `POST /copilot/insights`

The prediction endpoint loads the trained model artifact and returns a reorder probability for user feature inputs.

The copilot endpoint collects Snowflake metric context, sends it to OpenAI, and returns structured insights. If OpenAI is unavailable, a local metric-based fallback response is returned.

## 9. Dashboard

The Streamlit dashboard is the user-facing analytics layer.

Main file:

```text
dashboard/app.py
```

Dashboard sections:

- Project Overview
- KPIs
- Customer Insights
- Product Trends
- Model Performance
- AI Copilot

The dashboard connects to Snowflake for analytics data and to the deployed Render API for prediction and AI copilot functionality.

## 10. AI Copilot

The AI copilot turns Snowflake metrics into structured business insights.

It supports questions such as:

- Why is churn increasing?
- Which products drive repeat purchases?
- Which customer segment has the longest days between orders?
- How does retention change by cohort period?
- How can we improve retention?

Each response includes:

- AI source indicator
- Summary
- Explanation
- Impacted segments
- Charts
- Recommendations
- Detailed insights
- Follow-up questions

## 11. Orchestration And CI/CD

The project includes orchestration and CI/CD scaffolding:

- Airflow DAG for ingestion, dbt, feature engineering, and ML training
- GitHub Actions workflow for linting, Python tests, and dbt validation
- Render deployment for FastAPI
- Streamlit Cloud deployment for dashboard

## 12. Limitations

- Revenue is a proxy because the Instacart dataset does not include product prices.
- The model uses historical behavioral features and should not be interpreted as real-time intent prediction.
- Some production components, such as Airflow and Kafka, are included as architecture-ready scaffolding rather than fully hosted services.
- The committed model artifact is useful for portfolio deployment, but production systems should store model artifacts in object storage or a model registry.
- OpenAI responses are grounded in aggregate Snowflake metrics and do not have unrestricted SQL execution.

## 13. Future Improvements

- Store model artifacts in Snowflake stage, S3, or a model registry.
- Add API authentication and rate limiting.
- Add model drift monitoring and scheduled retraining.
- Add richer dashboard filters by cohort, product department, and customer segment.
- Deploy Airflow as a managed scheduler.
- Add real-time scoring from Kafka events.
- Add screenshot-based portfolio documentation in the README.
