from __future__ import annotations

import json
import math
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd
import snowflake.connector

from config.settings import Settings
from ml.model_monitoring import build_evaluation_record

IDENTIFIER_PART_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_$]*$")
MODEL_RUN_HISTORY_COLUMNS = (
    "run_id",
    "run_timestamp",
    "selected_model",
    "model_selection_metric",
    "source_relation",
    "model_artifact_path",
    "model_artifact_uri",
    "train_rows",
    "test_rows",
    "positive_rate",
    "threshold",
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "average_precision",
    "brier_score",
    "top_feature",
    "top_feature_importance",
)


def normalize_snowflake_table_identifier(table_name: str) -> str:
    parts = [part.strip() for part in table_name.split(".")]
    if not 1 <= len(parts) <= 3 or any(not part for part in parts):
        raise ValueError("Snowflake table name must contain one to three identifier parts.")

    invalid_parts = [part for part in parts if not IDENTIFIER_PART_PATTERN.fullmatch(part)]
    if invalid_parts:
        raise ValueError(f"Invalid Snowflake identifier part(s): {', '.join(invalid_parts)}")

    return ".".join(part.upper() for part in parts)


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return _to_jsonable(value.item())
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if pd.isna(value) and not isinstance(value, bool | str | bytes):
        return None
    return value


def _json_dumps(value: Any) -> str:
    return json.dumps(_to_jsonable(value), allow_nan=False, sort_keys=True)


def build_model_run_record(
    metrics: dict[str, Any],
    feature_importance: pd.DataFrame,
    drift_report: dict[str, Any],
    model_path: Path,
    source_relation: str,
    model_uri: str | None = None,
    run_id: str | None = None,
    run_timestamp: str | None = None,
) -> dict[str, Any]:
    timestamp = run_timestamp or datetime.now(UTC).isoformat()
    evaluation_record = build_evaluation_record(metrics, feature_importance, run_timestamp=timestamp)
    feature_importance_rows = feature_importance.to_dict(orient="records")

    record = {
        **evaluation_record,
        "run_id": run_id or str(uuid4()),
        "model_selection_metric": metrics.get("model_selection_metric"),
        "source_relation": source_relation,
        "model_artifact_path": str(model_path),
        "model_artifact_uri": model_uri,
        "metrics": metrics,
        "feature_importance": feature_importance_rows,
        "model_comparison": metrics.get("model_comparison", []),
        "drift_report": drift_report,
    }
    return record


def _build_snowflake_connection(settings: Settings) -> snowflake.connector.SnowflakeConnection:
    required = {
        "account": settings.snowflake_account,
        "user": settings.snowflake_user,
        "password": settings.snowflake_password,
        "warehouse": settings.snowflake_warehouse,
        "database": settings.snowflake_database,
        "schema": settings.snowflake_schema,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise ValueError(f"Missing Snowflake configuration: {', '.join(missing)}")

    return snowflake.connector.connect(
        account=settings.snowflake_account,
        user=settings.snowflake_user,
        password=settings.snowflake_password,
        role=settings.snowflake_role,
        warehouse=settings.snowflake_warehouse,
        database=settings.snowflake_database,
        schema=settings.snowflake_schema,
        insecure_mode=settings.snowflake_insecure_mode,
        login_timeout=15,
        network_timeout=30,
        socket_timeout=30,
    )


def _create_history_table(cursor: Any, table_name: str) -> None:
    cursor.execute(
        f"""
        create table if not exists {table_name} (
            run_id varchar not null,
            run_timestamp timestamp_tz not null,
            selected_model varchar,
            model_selection_metric varchar,
            source_relation varchar,
            model_artifact_path varchar,
            model_artifact_uri varchar,
            train_rows number,
            test_rows number,
            positive_rate float,
            threshold float,
            accuracy float,
            balanced_accuracy float,
            precision float,
            recall float,
            f1 float,
            roc_auc float,
            average_precision float,
            brier_score float,
            top_feature varchar,
            top_feature_importance float,
            metrics variant,
            feature_importance variant,
            model_comparison variant,
            drift_report variant,
            created_at timestamp_tz default current_timestamp()
        )
        """
    )


def _create_drift_summary_view(cursor: Any, table_name: str, view_name: str) -> None:
    cursor.execute(
        f"""
        create or replace view {view_name} as
        with ranked_runs as (
            select
                *,
                row_number() over (order by run_timestamp desc) as recency_rank,
                count(*) over () as run_count,
                lag(run_id) over (order by run_timestamp) as previous_run_id,
                lag(run_timestamp) over (order by run_timestamp) as previous_run_timestamp,
                lag(roc_auc) over (order by run_timestamp) as previous_roc_auc,
                lag(balanced_accuracy) over (order by run_timestamp) as previous_balanced_accuracy,
                lag(brier_score) over (order by run_timestamp) as previous_brier_score,
                lag(top_feature) over (order by run_timestamp) as previous_top_feature,
                lag(top_feature_importance) over (order by run_timestamp) as previous_top_feature_importance
            from {table_name}
        )
        select
            run_count,
            run_id as current_run_id,
            run_timestamp as current_run_timestamp,
            previous_run_id,
            previous_run_timestamp,
            selected_model,
            model_selection_metric,
            source_relation,
            model_artifact_uri,
            coalesce(
                drift_report:status::string,
                iff(run_count <= 1, 'baseline', 'stable')
            ) as drift_status,
            coalesce(
                drift_report:summary::string,
                'No drift report saved for the latest run.'
            ) as drift_summary,
            to_json(drift_report:findings) as drift_findings_json,
            roc_auc,
            previous_roc_auc,
            roc_auc - previous_roc_auc as roc_auc_delta,
            balanced_accuracy,
            previous_balanced_accuracy,
            balanced_accuracy - previous_balanced_accuracy as balanced_accuracy_delta,
            brier_score,
            previous_brier_score,
            brier_score - previous_brier_score as brier_score_delta,
            precision,
            recall,
            f1,
            top_feature,
            previous_top_feature,
            top_feature_importance,
            previous_top_feature_importance,
            top_feature_importance - previous_top_feature_importance as top_feature_importance_delta
        from ranked_runs
        where recency_rank = 1
        """
    )


def _insert_history_record(cursor: Any, table_name: str, record: dict[str, Any]) -> None:
    columns = ", ".join(MODEL_RUN_HISTORY_COLUMNS)
    placeholders = ", ".join(["%s"] * len(MODEL_RUN_HISTORY_COLUMNS))
    values = [record.get(column) for column in MODEL_RUN_HISTORY_COLUMNS]
    values.extend(
        [
            _json_dumps(record["metrics"]),
            _json_dumps(record["feature_importance"]),
            _json_dumps(record["model_comparison"]),
            _json_dumps(record["drift_report"]),
        ]
    )

    cursor.execute(
        f"""
        insert into {table_name} (
            {columns},
            metrics,
            feature_importance,
            model_comparison,
            drift_report
        )
        select
            {placeholders},
            parse_json(%s),
            parse_json(%s),
            parse_json(%s),
            parse_json(%s)
        """,
        values,
    )


def save_model_run_to_snowflake(
    metrics: dict[str, Any],
    feature_importance: pd.DataFrame,
    drift_report: dict[str, Any],
    model_path: Path,
    source_relation: str,
    settings: Settings,
    table_name: str | None = None,
    model_uri: str | None = None,
) -> dict[str, Any]:
    resolved_table_name = normalize_snowflake_table_identifier(
        table_name or settings.model_run_history_table
    )
    drift_summary_view = normalize_snowflake_table_identifier(settings.model_drift_summary_view)
    record = build_model_run_record(
        metrics=metrics,
        feature_importance=feature_importance,
        drift_report=drift_report,
        model_path=model_path,
        source_relation=source_relation,
        model_uri=model_uri,
    )

    connection = _build_snowflake_connection(settings)
    try:
        with connection.cursor() as cursor:
            _create_history_table(cursor, resolved_table_name)
            _create_drift_summary_view(cursor, resolved_table_name, drift_summary_view)
            _insert_history_record(cursor, resolved_table_name, record)
    finally:
        connection.close()

    return record
