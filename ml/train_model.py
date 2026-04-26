import argparse
import json
import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from config.logging_config import configure_logging
from config.settings import get_settings
from ingestion.snowflake_client import build_snowflake_engine
from ml.features import FEATURE_COLUMNS, TARGET_COLUMN, build_reorder_training_query

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = Path("ml/artifacts/random_forest_reorder_model.joblib")
DEFAULT_METRICS_PATH = Path("ml/artifacts/training_metrics.json")
DEFAULT_FEATURE_IMPORTANCE_PATH = Path("ml/artifacts/feature_importance.csv")
DEFAULT_SOURCE_RELATION = "fct_orders"


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

    feature_columns = [column for column in FEATURE_COLUMNS if column in frame.columns]
    if not feature_columns:
        raise ValueError("No numeric feature columns found for model training.")

    non_numeric_columns = [
        column for column in feature_columns if not pd.api.types.is_numeric_dtype(frame[column])
    ]
    if non_numeric_columns:
        raise ValueError(f"Feature columns must be numeric: {', '.join(non_numeric_columns)}")

    features = frame[feature_columns].fillna(0)
    target = frame[target_column].astype(int)
    logger.info("Training with feature columns: %s", ", ".join(feature_columns))
    return features, target


def train_random_forest(
    features: pd.DataFrame,
    target: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[RandomForestClassifier, dict[str, object], pd.DataFrame]:
    stratify = target if target.value_counts().min() >= 2 else None
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    model = RandomForestClassifier(
        n_estimators=80,
        max_depth=12,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced",
    )
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)) if y_test.nunique() == 2 else None,
        "train_rows": len(x_train),
        "test_rows": len(x_test),
        "positive_rate": float(target.mean()),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "classification_report": classification_report(
            y_test,
            predictions,
            output_dict=True,
            zero_division=0,
        ),
    }

    feature_importance = pd.DataFrame(
        {
            "feature": features.columns,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    return model, metrics, feature_importance


def save_model(model: RandomForestClassifier, model_path: Path = DEFAULT_MODEL_PATH) -> None:
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
