import os
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.bash import BashOperator

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", "/opt/airflow/customer-intelligence-platform"))
PYTHON_BIN = os.getenv("PYTHON_BIN", "python")
DBT_PROJECT_DIR = PROJECT_ROOT / "transformations" / "dbt"
DBT_PROFILES_DIR = Path(os.getenv("DBT_PROFILES_DIR", DBT_PROJECT_DIR))

default_args = {
    "owner": "data-engineering",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}


with DAG(
    dag_id="customer_intelligence_daily_pipeline",
    description="Daily Instacart ingestion, dbt transformation, feature engineering, and ML training pipeline.",
    default_args=default_args,
    start_date=datetime(2026, 1, 1),
    schedule="@daily",
    catchup=False,
    max_active_runs=1,
    tags=["customer-intelligence", "snowflake", "dbt", "ml"],
) as dag:
    run_ingestion = BashOperator(
        task_id="run_instacart_ingestion",
        bash_command=(
            f"cd {PROJECT_ROOT} && "
            f"{PYTHON_BIN} -m ingestion.load_data"
        ),
        env={
            **os.environ,
            "PYTHONPATH": str(PROJECT_ROOT),
        },
    )

    run_dbt_models = BashOperator(
        task_id="run_dbt_models",
        bash_command=(
            f"cd {DBT_PROJECT_DIR} && "
            f"dbt deps --profiles-dir {DBT_PROFILES_DIR} && "
            f"dbt run --profiles-dir {DBT_PROFILES_DIR} && "
            f"dbt test --profiles-dir {DBT_PROFILES_DIR}"
        ),
        env={
            **os.environ,
            "PYTHONPATH": str(PROJECT_ROOT),
        },
    )

    run_data_validation = BashOperator(
        task_id="run_data_validation",
        bash_command=(
            f"cd {PROJECT_ROOT} && "
            f"{PYTHON_BIN} -m data_quality.validation"
        ),
        env={
            **os.environ,
            "PYTHONPATH": str(PROJECT_ROOT),
        },
    )

    run_feature_engineering = BashOperator(
        task_id="run_feature_engineering",
        bash_command=(
            f"cd {DBT_PROJECT_DIR} && "
            f"dbt run --select feature_store --profiles-dir {DBT_PROFILES_DIR} && "
            f"dbt test --select feature_store --profiles-dir {DBT_PROFILES_DIR}"
        ),
        env={
            **os.environ,
            "PYTHONPATH": str(PROJECT_ROOT),
        },
    )

    train_model = BashOperator(
        task_id="train_reorder_model",
        bash_command=(
            f"cd {PROJECT_ROOT} && "
            f"{PYTHON_BIN} -m ml.train_model"
        ),
        env={
            **os.environ,
            "PYTHONPATH": str(PROJECT_ROOT),
        },
    )

    run_ingestion >> run_data_validation >> run_dbt_models >> run_feature_engineering >> train_model
