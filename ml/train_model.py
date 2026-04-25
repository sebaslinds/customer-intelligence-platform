import argparse
import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from config.logging_config import configure_logging
from config.settings import get_settings
from ingestion.snowflake_client import build_snowflake_engine

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = Path("ml/artifacts/random_forest_reorder_model.joblib")
DEFAULT_FEATURE_TABLE = "feature_store"
TARGET_COLUMN = "will_reorder"
EXCLUDED_COLUMNS = {TARGET_COLUMN, "user_id"}


def load_feature_store(table_name: str = DEFAULT_FEATURE_TABLE) -> pd.DataFrame:
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

    feature_columns = [
        column
        for column in frame.columns
        if column not in EXCLUDED_COLUMNS and pd.api.types.is_numeric_dtype(frame[column])
    ]
    if not feature_columns:
        raise ValueError("No numeric feature columns found for model training.")

    features = frame[feature_columns].fillna(0)
    target = frame[target_column].astype(int)
    logger.info("Training with feature columns: %s", ", ".join(feature_columns))
    return features, target


def train_random_forest(
    features: pd.DataFrame,
    target: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[RandomForestClassifier, dict[str, float]]:
    stratify = target if target.value_counts().min() >= 2 else None
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced",
    )
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
    }
    return model, metrics


def save_model(model: RandomForestClassifier, model_path: Path = DEFAULT_MODEL_PATH) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    logger.info("Saved trained model to %s", model_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a reorder classification model.")
    parser.add_argument(
        "--feature-table",
        default=DEFAULT_FEATURE_TABLE,
        help="Snowflake table or fully qualified table containing feature_store data.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Output path for the trained model artifact.",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split size.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    args = parse_args()

    frame = load_feature_store(args.feature_table)
    features, target = build_training_matrix(frame)
    model, metrics = train_random_forest(
        features=features,
        target=target,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    save_model(model, args.model_path)

    logger.info("Model accuracy: %.4f", metrics["accuracy"])
    logger.info("Model precision: %.4f", metrics["precision"])
    logger.info("Model recall: %.4f", metrics["recall"])


if __name__ == "__main__":
    main()
