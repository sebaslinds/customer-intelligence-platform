import argparse
import json
import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config.logging_config import configure_logging
from config.settings import get_settings
from ingestion.snowflake_client import build_snowflake_engine
from ml.features import FEATURE_COLUMNS, TARGET_COLUMN, build_reorder_training_query
from ml.model_monitoring import save_monitoring_artifacts

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = Path("ml/artifacts/random_forest_reorder_model.joblib")
DEFAULT_METRICS_PATH = Path("ml/artifacts/training_metrics.json")
DEFAULT_FEATURE_IMPORTANCE_PATH = Path("ml/artifacts/feature_importance.csv")
DEFAULT_MODEL_HISTORY_PATH = Path("ml/artifacts/model_evaluation_history.csv")
DEFAULT_MODEL_DRIFT_PATH = Path("ml/artifacts/model_drift_report.json")
DEFAULT_SOURCE_RELATION = "fct_orders"
MAX_CURVE_POINTS = 200
MODEL_SELECTION_METRIC = "roc_auc"
THRESHOLD_SELECTION_METRIC = "balanced_accuracy"


def load_feature_store(table_name: str) -> pd.DataFrame:
    settings = get_settings()
    engine = build_snowflake_engine(settings)

    try:
        query = f"select * from {table_name}"
        logger.info("Loading feature store from Snowflake table: %s", table_name)
        frame = pd.read_sql(query, engine)
    except Exception:
        logger.exception("Failed to load feature store from Snowflake")
        raise
    finally:
        engine.dispose()

    frame.columns = [column.lower() for column in frame.columns]
    logger.info("Loaded feature store with %s rows and %s columns", *frame.shape)
    return frame


def load_training_dataset(
    source_relation: str = DEFAULT_SOURCE_RELATION,
    row_limit: int | None = 250_000,
) -> pd.DataFrame:
    settings = get_settings()
    engine = build_snowflake_engine(settings)

    try:
        query = build_reorder_training_query(source_relation=source_relation, row_limit=row_limit)
        logger.info("Loading time-aware training dataset from Snowflake relation: %s", source_relation)
        frame = pd.read_sql(query, engine)
    except Exception:
        logger.exception("Failed to load time-aware training dataset from Snowflake")
        raise
    finally:
        engine.dispose()

    frame.columns = [column.lower() for column in frame.columns]
    logger.info("Loaded training dataset with %s rows and %s columns", *frame.shape)
    return frame


def validate_training_data(frame: pd.DataFrame, target_column: str = TARGET_COLUMN) -> None:
    if frame.empty:
        raise ValueError("Feature store is empty. Run dbt models before training.")

    if target_column not in frame.columns:
        raise ValueError(
            f"Target column '{target_column}' is missing from feature_store. "
            "Add this label before running model training."
        )

    if frame[target_column].nunique(dropna=True) < 2:
        raise ValueError(f"Target column '{target_column}' must contain at least two classes.")


def build_training_matrix(
    frame: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
) -> tuple[pd.DataFrame, pd.Series]:
    validate_training_data(frame, target_column)

    available_feature_columns = [column for column in FEATURE_COLUMNS if column in frame.columns]
    if not available_feature_columns:
        raise ValueError("No numeric feature columns found for model training.")

    non_numeric_columns = [
        column for column in available_feature_columns if not pd.api.types.is_numeric_dtype(frame[column])
    ]
    if non_numeric_columns:
        raise ValueError(f"Feature columns must be numeric: {', '.join(non_numeric_columns)}")

    features = frame.reindex(columns=FEATURE_COLUMNS, fill_value=0).fillna(0)
    target = frame[target_column].astype(int)
    logger.info("Training with feature columns: %s", ", ".join(FEATURE_COLUMNS))
    return features, target


def build_threshold_analysis(target: pd.Series, probabilities: pd.Series | list[float]) -> list[dict[str, float]]:
    rows = []
    for threshold in [round(value / 20, 2) for value in range(1, 20)]:
        predictions = [1 if probability >= threshold else 0 for probability in probabilities]
        true_negative, false_positive, false_negative, true_positive = confusion_matrix(
            target,
            predictions,
            labels=[0, 1],
        ).ravel()
        specificity_denominator = true_negative + false_positive
        specificity = true_negative / specificity_denominator if specificity_denominator else 0
        rows.append(
            {
                "threshold": threshold,
                "precision": float(precision_score(target, predictions, zero_division=0)),
                "recall": float(recall_score(target, predictions, zero_division=0)),
                "f1": float(f1_score(target, predictions, zero_division=0)),
                "balanced_accuracy": float(balanced_accuracy_score(target, predictions)),
                "specificity": float(specificity),
                "false_positive_rate": float(1 - specificity),
                "positive_prediction_rate": float(sum(predictions) / len(predictions)),
            }
        )
    return rows


def select_recommended_threshold(threshold_analysis: list[dict[str, float]]) -> float:
    if not threshold_analysis:
        return 0.5

    selected_row = max(
        threshold_analysis,
        key=lambda row: (
            row[THRESHOLD_SELECTION_METRIC],
            row["f1"],
            -abs(row["positive_prediction_rate"] - 0.5),
        ),
    )
    return float(selected_row["threshold"])


def build_curve_rows(
    target: pd.Series,
    probabilities: pd.Series | list[float],
) -> tuple[list[dict[str, float]], list[dict[str, float | None]]]:
    if target.nunique() != 2:
        return [], []

    fpr, tpr, roc_thresholds = roc_curve(target, probabilities)
    roc_rows = [
        {
            "false_positive_rate": float(false_positive_rate),
            "true_positive_rate": float(true_positive_rate),
            "threshold": None if threshold == float("inf") else float(threshold),
        }
        for false_positive_rate, true_positive_rate, threshold in zip(fpr, tpr, roc_thresholds, strict=False)
    ]

    precision, recall, pr_thresholds = precision_recall_curve(target, probabilities)
    pr_rows = []
    for index, (precision_value, recall_value) in enumerate(zip(precision, recall, strict=False)):
        threshold = float(pr_thresholds[index]) if index < len(pr_thresholds) else None
        pr_rows.append(
            {
                "precision": float(precision_value),
                "recall": float(recall_value),
                "threshold": threshold,
            }
        )

    return roc_rows, pr_rows


def sample_curve_rows(rows: list[dict[str, float | None]], max_points: int = MAX_CURVE_POINTS) -> list[dict[str, float | None]]:
    if len(rows) <= max_points:
        return rows

    step = (len(rows) - 1) / (max_points - 1)
    indexes = {round(index * step) for index in range(max_points)}
    return [row for index, row in enumerate(rows) if index in indexes]


def get_candidate_models(random_state: int = 42) -> dict[str, ClassifierMixin]:
    return {
        "random_forest": RandomForestClassifier(
            n_estimators=80,
            max_depth=12,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
            class_weight="balanced",
        ),
        "logistic_regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=1_000,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=120,
            learning_rate=0.05,
            max_depth=3,
            random_state=random_state,
        ),
    }


def get_probability_scores(model: ClassifierMixin, features: pd.DataFrame) -> list[float]:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(features)[:, 1].tolist()

    if hasattr(model, "decision_function"):
        scores = model.decision_function(features)
        minimum = min(scores)
        maximum = max(scores)
        if maximum == minimum:
            return [0.5 for _ in scores]
        return [float((score - minimum) / (maximum - minimum)) for score in scores]

    return [float(prediction) for prediction in model.predict(features)]


def evaluate_model(
    model_name: str,
    model: ClassifierMixin,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    target: pd.Series,
) -> dict[str, object]:
    probabilities = get_probability_scores(model, x_test)
    threshold_analysis = build_threshold_analysis(y_test, probabilities)
    recommended_threshold = select_recommended_threshold(threshold_analysis)
    predictions = [1 if probability >= recommended_threshold else 0 for probability in probabilities]
    matrix = confusion_matrix(y_test, predictions)
    roc_rows, precision_recall_rows = build_curve_rows(y_test, probabilities)

    return {
        "model_name": model_name,
        "accuracy": float(accuracy_score(y_test, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)) if y_test.nunique() == 2 else None,
        "average_precision": (
            float(average_precision_score(y_test, probabilities)) if y_test.nunique() == 2 else None
        ),
        "brier_score": float(brier_score_loss(y_test, probabilities)),
        "train_rows": int(len(target) - len(y_test)),
        "test_rows": int(len(y_test)),
        "positive_rate": float(target.mean()),
        "confusion_matrix": matrix.tolist(),
        "classification_report": classification_report(
            y_test,
            predictions,
            output_dict=True,
            zero_division=0,
        ),
        "threshold_analysis": threshold_analysis,
        "recommended_threshold": recommended_threshold,
        "threshold_selection_metric": THRESHOLD_SELECTION_METRIC,
        "positive_prediction_rate": float(sum(predictions) / len(predictions)),
        "roc_curve": sample_curve_rows(roc_rows),
        "precision_recall_curve": sample_curve_rows(precision_recall_rows),
    }


def build_model_comparison(results: dict[str, dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    for model_name, metrics in results.items():
        rows.append(
            {
                "model_name": model_name,
                "accuracy": metrics.get("accuracy"),
                "balanced_accuracy": metrics.get("balanced_accuracy"),
                "precision": metrics.get("precision"),
                "recall": metrics.get("recall"),
                "f1": metrics.get("f1"),
                "roc_auc": metrics.get("roc_auc"),
                "recommended_threshold": metrics.get("recommended_threshold"),
                "positive_prediction_rate": metrics.get("positive_prediction_rate"),
                "average_precision": metrics.get("average_precision"),
                "brier_score": metrics.get("brier_score"),
            }
        )

    return sorted(
        rows,
        key=lambda row: (
            row.get(MODEL_SELECTION_METRIC) is not None,
            float(row.get(MODEL_SELECTION_METRIC) or 0),
            float(row.get("f1") or 0),
        ),
        reverse=True,
    )


def extract_feature_importance(model: ClassifierMixin, feature_names: pd.Index) -> pd.DataFrame:
    estimator = model
    if isinstance(model, Pipeline):
        estimator = model.named_steps["classifier"]

    if hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        values = abs(estimator.coef_[0])
    else:
        values = [0 for _ in feature_names]

    total = sum(values)
    normalized_values = values if total == 0 else [float(value / total) for value in values]
    return pd.DataFrame(
        {
            "feature": feature_names,
            "importance": normalized_values,
        }
    ).sort_values("importance", ascending=False)


def build_model_recommendations(
    metrics: dict[str, object],
    feature_importance: pd.DataFrame,
) -> list[dict[str, str]]:
    recommendations = []
    positive_rate = float(metrics.get("positive_rate") or 0)
    roc_auc = metrics.get("roc_auc")
    recall = float(metrics.get("recall") or 0)
    precision = float(metrics.get("precision") or 0)
    balanced_accuracy = float(metrics.get("balanced_accuracy") or 0)

    if positive_rate >= 0.8:
        recommendations.append(
            {
                "priority": "high",
                "action": "Monitor class imbalance before using the model for automated targeting.",
                "expected_impact": (
                    "The dataset contains many reorder-positive examples, so accuracy alone can overstate "
                    "model quality. Use balanced accuracy, ROC AUC, and threshold analysis together."
                ),
            }
        )

    if balanced_accuracy < 0.6:
        recommendations.append(
            {
                "priority": "high",
                "action": "Treat the current classifier as a ranking signal, not a final automated decision.",
                "expected_impact": (
                    "Balanced accuracy is close to random because the no-reorder class is hard to detect. "
                    "Use the probability score for prioritization until richer negative-class features are added."
                ),
            }
        )

    if roc_auc is not None and float(roc_auc) < 0.72:
        recommendations.append(
            {
                "priority": "medium",
                "action": "Add richer behavioral features such as recency, product affinity, and department mix.",
                "expected_impact": "Improves separation between likely reorders and customers who may churn.",
            }
        )

    if recall < precision:
        recommendations.append(
            {
                "priority": "medium",
                "action": "Tune the prediction threshold based on campaign goals.",
                "expected_impact": (
                    "Lower thresholds catch more potential reorderers; higher thresholds reduce false positives "
                    "for expensive retention campaigns."
                ),
            }
        )

    if not feature_importance.empty and feature_importance["importance"].head(2).sum() > 0.85:
        recommendations.append(
            {
                "priority": "medium",
                "action": "Reduce dependence on only a few features.",
                "expected_impact": (
                    "A broader feature set usually makes the model more stable and easier to explain to business users."
                ),
            }
        )

    return recommendations


def train_random_forest(
    features: pd.DataFrame,
    target: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[ClassifierMixin, dict[str, object], pd.DataFrame]:
    stratify = target if target.value_counts().min() >= 2 else None
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    trained_models = {}
    model_results = {}
    for model_name, candidate_model in get_candidate_models(random_state).items():
        logger.info("Training candidate model: %s", model_name)
        candidate_model.fit(x_train, y_train)
        trained_models[model_name] = candidate_model
        model_results[model_name] = evaluate_model(model_name, candidate_model, x_test, y_test, target)

    comparison = build_model_comparison(model_results)
    best_model_name = str(comparison[0]["model_name"])
    model = trained_models[best_model_name]
    metrics = model_results[best_model_name]
    metrics["selected_model"] = best_model_name
    metrics["model_selection_metric"] = MODEL_SELECTION_METRIC
    metrics["model_comparison"] = comparison

    feature_importance = extract_feature_importance(model, features.columns)
    metrics["model_recommendations"] = build_model_recommendations(metrics, feature_importance)

    return model, metrics, feature_importance


def save_model(model: ClassifierMixin, model_path: Path = DEFAULT_MODEL_PATH) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    logger.info("Saved trained model to %s", model_path)


def save_training_artifacts(
    metrics: dict[str, object],
    feature_importance: pd.DataFrame,
    metrics_path: Path = DEFAULT_METRICS_PATH,
    feature_importance_path: Path = DEFAULT_FEATURE_IMPORTANCE_PATH,
) -> None:
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    feature_importance.to_csv(feature_importance_path, index=False)
    logger.info("Saved training metrics to %s", metrics_path)
    logger.info("Saved feature importance to %s", feature_importance_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a reorder classification model.")
    parser.add_argument(
        "--feature-table",
        default=None,
        help=(
            "Optional legacy Snowflake table containing prebuilt training data. "
            "If omitted, a time-aware training dataset is built from fct_orders."
        ),
    )
    parser.add_argument(
        "--source-relation",
        default=DEFAULT_SOURCE_RELATION,
        help="Snowflake relation used to build the time-aware training dataset.",
    )
    parser.add_argument(
        "--row-limit",
        type=int,
        default=250_000,
        help="Maximum number of training rows to load from Snowflake. Use 0 for no limit.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Output path for the trained model artifact.",
    )
    parser.add_argument(
        "--metrics-path",
        type=Path,
        default=DEFAULT_METRICS_PATH,
        help="Output path for training metrics JSON.",
    )
    parser.add_argument(
        "--feature-importance-path",
        type=Path,
        default=DEFAULT_FEATURE_IMPORTANCE_PATH,
        help="Output path for feature importance CSV.",
    )
    parser.add_argument(
        "--model-history-path",
        type=Path,
        default=DEFAULT_MODEL_HISTORY_PATH,
        help="Output path for historical model evaluation CSV.",
    )
    parser.add_argument(
        "--model-drift-path",
        type=Path,
        default=DEFAULT_MODEL_DRIFT_PATH,
        help="Output path for model drift report JSON.",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split size.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    args = parse_args()

    if args.feature_table:
        logger.warning(
            "Using legacy feature table '%s'. Prefer the default time-aware dataset to avoid leakage.",
            args.feature_table,
        )
        frame = load_feature_store(args.feature_table)
    else:
        row_limit = args.row_limit if args.row_limit > 0 else None
        frame = load_training_dataset(args.source_relation, row_limit=row_limit)

    features, target = build_training_matrix(frame)
    model, metrics, feature_importance = train_random_forest(
        features=features,
        target=target,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    save_model(model, args.model_path)
    save_training_artifacts(metrics, feature_importance, args.metrics_path, args.feature_importance_path)
    save_monitoring_artifacts(metrics, feature_importance, args.model_history_path, args.model_drift_path)

    logger.info("Model accuracy: %.4f", metrics["accuracy"])
    logger.info("Model precision: %.4f", metrics["precision"])
    logger.info("Model recall: %.4f", metrics["recall"])
    logger.info("Model F1: %.4f", metrics["f1"])
    if metrics["roc_auc"] is not None:
        logger.info("Model ROC AUC: %.4f", metrics["roc_auc"])
    logger.info("Confusion matrix: %s", metrics["confusion_matrix"])
    logger.info("Feature importance:\n%s", feature_importance.to_string(index=False))


if __name__ == "__main__":
    main()
