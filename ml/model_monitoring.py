from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

TRACKED_METRICS = (
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "average_precision",
    "brier_score",
    "positive_rate",
)
LOWER_IS_BETTER = {"brier_score"}


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> int | None:
    number = _to_float(value)
    if number is None:
        return None
    return int(number)


def _top_feature(feature_importance: pd.DataFrame) -> tuple[str | None, float | None]:
    if feature_importance.empty or "feature" not in feature_importance or "importance" not in feature_importance:
        return None, None

    ranked = feature_importance.copy()
    ranked["importance"] = pd.to_numeric(ranked["importance"], errors="coerce")
    ranked = ranked.dropna(subset=["importance"]).sort_values("importance", ascending=False)
    if ranked.empty:
        return None, None

    top_row = ranked.iloc[0]
    return str(top_row["feature"]), float(top_row["importance"])


def build_evaluation_record(
    metrics: dict[str, Any],
    feature_importance: pd.DataFrame,
    run_timestamp: str | None = None,
) -> dict[str, Any]:
    top_feature, top_feature_importance = _top_feature(feature_importance)
    record: dict[str, Any] = {
        "run_timestamp": run_timestamp or datetime.now(UTC).isoformat(),
        "selected_model": str(metrics.get("selected_model") or "unknown"),
        "train_rows": _to_int(metrics.get("train_rows")),
        "test_rows": _to_int(metrics.get("test_rows")),
        "threshold": _to_float(metrics.get("threshold") or metrics.get("recommended_threshold")),
        "top_feature": top_feature,
        "top_feature_importance": top_feature_importance,
    }

    for metric_name in TRACKED_METRICS:
        record[metric_name] = _to_float(metrics.get(metric_name))

    return record


def load_evaluation_history(history_path: Path) -> pd.DataFrame:
    if not history_path.exists():
        return pd.DataFrame()

    history = pd.read_csv(history_path)
    if "run_timestamp" in history.columns:
        history = history.sort_values("run_timestamp").reset_index(drop=True)
    return history


def _metric_has_drift(metric_name: str, delta: float, threshold: float) -> bool:
    if metric_name in LOWER_IS_BETTER:
        return delta > threshold
    return delta < -threshold


def _metric_finding(metric_name: str, previous_value: float, current_value: float, delta: float) -> dict[str, Any]:
    direction = "increased" if delta > 0 else "decreased"
    severity = "high" if abs(delta) >= 0.08 else "medium"
    return {
        "type": "metric_drift",
        "metric": metric_name,
        "previous_value": previous_value,
        "current_value": current_value,
        "delta": delta,
        "severity": severity,
        "description": f"{metric_name} {direction} by {abs(delta):.3f} compared with the previous run.",
    }


def _feature_finding(
    previous_feature: str,
    current_feature: str,
    previous_importance: float | None = None,
    current_importance: float | None = None,
) -> dict[str, Any]:
    return {
        "type": "feature_drift",
        "metric": "top_feature",
        "previous_value": previous_feature,
        "current_value": current_feature,
        "delta": None
        if previous_importance is None or current_importance is None
        else current_importance - previous_importance,
        "severity": "medium",
        "description": "The most influential feature changed compared with the previous run.",
    }


def build_drift_report(
    current_record: dict[str, Any],
    previous_history: pd.DataFrame,
    metric_drop_threshold: float = 0.03,
    feature_shift_threshold: float = 0.10,
) -> dict[str, Any]:
    if previous_history.empty:
        return {
            "status": "baseline",
            "summary": "First tracked training run saved as the monitoring baseline.",
            "current_run_timestamp": current_record.get("run_timestamp"),
            "previous_run_timestamp": None,
            "selected_model": current_record.get("selected_model"),
            "findings": [],
            "tracked_metrics": {},
            "top_feature": current_record.get("top_feature"),
            "top_feature_importance": current_record.get("top_feature_importance"),
        }

    previous_record = previous_history.iloc[-1].to_dict()
    findings: list[dict[str, Any]] = []
    tracked_metrics: dict[str, dict[str, float | None]] = {}

    for metric_name in TRACKED_METRICS:
        previous_value = _to_float(previous_record.get(metric_name))
        current_value = _to_float(current_record.get(metric_name))
        if previous_value is None or current_value is None:
            continue

        delta = current_value - previous_value
        tracked_metrics[metric_name] = {
            "previous": previous_value,
            "current": current_value,
            "delta": delta,
        }
        if _metric_has_drift(metric_name, delta, metric_drop_threshold):
            findings.append(_metric_finding(metric_name, previous_value, current_value, delta))

    previous_feature = previous_record.get("top_feature")
    current_feature = current_record.get("top_feature")
    previous_importance = _to_float(previous_record.get("top_feature_importance"))
    current_importance = _to_float(current_record.get("top_feature_importance"))

    if previous_feature and current_feature and str(previous_feature) != str(current_feature):
        findings.append(
            _feature_finding(str(previous_feature), str(current_feature), previous_importance, current_importance)
        )
    elif previous_importance is not None and current_importance is not None:
        feature_delta = current_importance - previous_importance
        if abs(feature_delta) >= feature_shift_threshold:
            findings.append(
                {
                    "type": "feature_drift",
                    "metric": "top_feature_importance",
                    "previous_value": previous_importance,
                    "current_value": current_importance,
                    "delta": feature_delta,
                    "severity": "medium",
                    "description": "The leading feature importance changed materially compared with the previous run.",
                }
            )

    status = "attention" if findings else "stable"
    summary = (
        f"Model monitoring found {len(findings)} signal(s) requiring review."
        if findings
        else "Model monitoring is stable against the previous training run."
    )

    return {
        "status": status,
        "summary": summary,
        "current_run_timestamp": current_record.get("run_timestamp"),
        "previous_run_timestamp": previous_record.get("run_timestamp"),
        "selected_model": current_record.get("selected_model"),
        "findings": findings,
        "tracked_metrics": tracked_metrics,
        "top_feature": current_feature,
        "top_feature_importance": current_importance,
    }


def save_monitoring_artifacts(
    metrics: dict[str, Any],
    feature_importance: pd.DataFrame,
    history_path: Path,
    drift_report_path: Path,
) -> dict[str, Any]:
    history_path.parent.mkdir(parents=True, exist_ok=True)
    previous_history = load_evaluation_history(history_path)
    current_record = build_evaluation_record(metrics, feature_importance)
    drift_report = build_drift_report(current_record, previous_history)

    updated_history = pd.concat([previous_history, pd.DataFrame([current_record])], ignore_index=True)
    updated_history.to_csv(history_path, index=False)

    drift_report_path.parent.mkdir(parents=True, exist_ok=True)
    drift_report_path.write_text(json.dumps(drift_report, indent=2), encoding="utf-8")

    logger.info("Saved model evaluation history to %s", history_path)
    logger.info("Saved model drift report to %s", drift_report_path)
    return drift_report
