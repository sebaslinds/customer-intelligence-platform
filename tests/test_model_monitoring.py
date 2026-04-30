import json

import pandas as pd

from ml.model_monitoring import build_drift_report, build_evaluation_record, save_monitoring_artifacts


def _metrics(**overrides: float | str) -> dict[str, float | str]:
    metrics: dict[str, float | str] = {
        "selected_model": "random_forest",
        "accuracy": 0.80,
        "balanced_accuracy": 0.70,
        "precision": 0.82,
        "recall": 0.64,
        "f1": 0.72,
        "roc_auc": 0.76,
        "average_precision": 0.84,
        "brier_score": 0.12,
        "positive_rate": 0.50,
    }
    metrics.update(overrides)
    return metrics


def _feature_importance(feature: str = "total_orders", importance: float = 0.42) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature": [feature, "days_between_orders"],
            "importance": [importance, 0.25],
        }
    )


def test_build_drift_report_creates_baseline_for_first_run() -> None:
    record = build_evaluation_record(
        _metrics(),
        _feature_importance(),
        run_timestamp="2026-01-01T00:00:00+00:00",
    )

    report = build_drift_report(record, pd.DataFrame())

    assert report["status"] == "baseline"
    assert report["top_feature"] == "total_orders"
    assert report["findings"] == []


def test_build_drift_report_flags_metric_drop() -> None:
    previous_history = pd.DataFrame(
        [
            build_evaluation_record(
                _metrics(balanced_accuracy=0.75),
                _feature_importance(),
                run_timestamp="2026-01-01T00:00:00+00:00",
            )
        ]
    )
    current_record = build_evaluation_record(
        _metrics(balanced_accuracy=0.68),
        _feature_importance(),
        run_timestamp="2026-01-02T00:00:00+00:00",
    )

    report = build_drift_report(current_record, previous_history)

    assert report["status"] == "attention"
    assert any(finding["metric"] == "balanced_accuracy" for finding in report["findings"])


def test_save_monitoring_artifacts_appends_history(tmp_path) -> None:
    history_path = tmp_path / "model_evaluation_history.csv"
    drift_path = tmp_path / "model_drift_report.json"

    save_monitoring_artifacts(_metrics(), _feature_importance(), history_path, drift_path)
    save_monitoring_artifacts(_metrics(roc_auc=0.73), _feature_importance(), history_path, drift_path)

    history = pd.read_csv(history_path)
    report = json.loads(drift_path.read_text(encoding="utf-8"))

    assert len(history) == 2
    assert report["status"] in {"stable", "attention"}
