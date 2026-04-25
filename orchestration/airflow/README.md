# Airflow Orchestration

This folder contains the Airflow DAG for the Customer Intelligence Platform.

## DAG

`customer_intelligence_daily_pipeline`

Daily task order:

1. Run Instacart ingestion with `python -m ingestion.load_data`
2. Run dbt dependencies, models, and tests
3. Run the `feature_store` dbt model and tests
4. Train the reorder classification model with `python -m ml.train_model`

## Required Environment Variables

- `PROJECT_ROOT`: absolute path to this repository on the Airflow worker
- `PYTHON_BIN`: Python executable available to Airflow, defaults to `python`
- `DBT_PROFILES_DIR`: dbt profiles directory, defaults to `transformations/dbt`
- `SNOWFLAKE_ACCOUNT`
- `SNOWFLAKE_USER`
- `SNOWFLAKE_PASSWORD`
- `SNOWFLAKE_ROLE`
- `SNOWFLAKE_WAREHOUSE`
- `SNOWFLAKE_DATABASE`
- `SNOWFLAKE_SCHEMA`
- `INSTACART_DATA_DIR`
- `INGESTION_CHUNK_SIZE`

## Local Airflow Placement

Copy or symlink `orchestration/airflow/dags/customer_intelligence_pipeline.py` into your Airflow `dags/` directory, then set `PROJECT_ROOT` to the repository path.
