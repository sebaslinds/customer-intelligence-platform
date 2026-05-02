from pathlib import Path

import pandas as pd
import pytest

from config.settings import Settings
from ml.model_registry import (
    build_model_run_record,
    normalize_snowflake_table_identifier,
    save_model_run_to_snowflake,
)


def _metrics() -> dict[str, object]:
    return {
        "selected_model": "gradient_boosting",
        "model_selection_metric": "roc_auc",
        "accuracy": 0.54,
        "balanced_accuracy": 0.67,
        "precision": 0.97,
        "recall": 0.52,
        "f1": 0.68,
        "roc_auc": 0.74,
        "average_precision": 0.97,
        "brier_score": 0.06,
        "positive_rate": 0.93,
        "train_rows": 100,
        "test_rows": 25,
        "recommended_threshold": 0.95,
        "model_comparison": [{"model_name": "gradient_boosting", "roc_auc": 0.74}],
    }


def _feature_importance() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature": ["total_orders", "days_between_orders"],
            "importance": [0.42, 0.25],
        }
    )


def test_normalize_snowflake_table_identifier_accepts_qualified_names() -> None:
    table_name = normalize_snowflake_table_identifier("customer_intelligence.ml_artifacts.model_training_runs")

    assert table_name == "CUSTOMER_INTELLIGENCE.ML_ARTIFACTS.MODEL_TRAINING_RUNS"


def test_normalize_snowflake_table_identifier_rejects_unsafe_names() -> None:
    with pytest.raises(ValueError):
        normalize_snowflake_table_identifier("CUSTOMER_INTELLIGENCE.ML_ARTIFACTS.RUNS;drop table users")


def test_build_model_run_record_captures_auditable_payloads() -> None:
    record = build_model_run_record(
        metrics=_metrics(),
        feature_importance=_feature_importance(),
        drift_report={"status": "stable"},
        model_path=Path("ml/artifacts/random_forest_reorder_model.joblib"),
        source_relation="fct_orders",
        model_uri="@CUSTOMER_INTELLIGENCE.ML_ARTIFACTS.MODEL_STAGE/models/random_forest_reorder_model.joblib",
        run_id="run-1",
        run_timestamp="2026-05-02T00:00:00+00:00",
    )

    assert record["run_id"] == "run-1"
    assert record["selected_model"] == "gradient_boosting"
    assert record["threshold"] == 0.95
    assert record["top_feature"] == "total_orders"
    assert record["model_artifact_uri"].startswith("@CUSTOMER_INTELLIGENCE")
    assert record["model_comparison"] == [{"model_name": "gradient_boosting", "roc_auc": 0.74}]


def test_save_model_run_to_snowflake_creates_table_and_inserts(monkeypatch) -> None:
    executed_sql: list[str] = []
    executed_params: list[list[object]] = []

    class FakeCursor:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return None

        def execute(self, sql: str, params: list[object] | None = None) -> None:
            executed_sql.append(sql)
            if params is not None:
                executed_params.append(params)

    class FakeConnection:
        def cursor(self) -> FakeCursor:
            return FakeCursor()

        def close(self) -> None:
            return None

    monkeypatch.setattr("ml.model_registry._build_snowflake_connection", lambda settings: FakeConnection())

    record = save_model_run_to_snowflake(
        metrics=_metrics(),
        feature_importance=_feature_importance(),
        drift_report={"status": "stable"},
        model_path=Path("ml/artifacts/random_forest_reorder_model.joblib"),
        source_relation="fct_orders",
        settings=Settings(),
        table_name="CUSTOMER_INTELLIGENCE.ML_ARTIFACTS.MODEL_TRAINING_RUNS",
        model_uri=None,
    )

    assert record["source_relation"] == "fct_orders"
    assert any("create table if not exists CUSTOMER_INTELLIGENCE.ML_ARTIFACTS.MODEL_TRAINING_RUNS" in sql for sql in executed_sql)
    assert any("insert into CUSTOMER_INTELLIGENCE.ML_ARTIFACTS.MODEL_TRAINING_RUNS" in sql for sql in executed_sql)
    assert len(executed_params[0]) == 25
