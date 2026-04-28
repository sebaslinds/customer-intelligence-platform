import logging
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import HTTPException, status

from schemas.prediction import PredictionRequest

logger = logging.getLogger(__name__)

DEFAULT_FEATURE_COLUMNS = [
    "total_orders",
    "avg_basket_size",
    "reorder_ratio",
    "unique_products",
    "days_between_orders",
]

model_state: dict[str, Any] = {"model": None, "feature_columns": DEFAULT_FEATURE_COLUMNS}


def load_model(model_path: str) -> Any:
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found: {path}")

    model = joblib.load(path)
    feature_columns = getattr(model, "feature_names_in_", DEFAULT_FEATURE_COLUMNS)
    model_state["model"] = model
    model_state["feature_columns"] = list(feature_columns)
    logger.info("Loaded model artifact from %s", path)
    return model


def is_model_loaded() -> bool:
    return model_state["model"] is not None


def build_prediction_frame(payload: PredictionRequest) -> pd.DataFrame:
    values = payload.model_dump()
    feature_columns = model_state["feature_columns"]
    missing_features = [column for column in feature_columns if column not in values]
    if missing_features:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Missing required model features: {', '.join(missing_features)}",
        )

    return pd.DataFrame([{column: values[column] for column in feature_columns}])


def predict_reorder_probability(payload: PredictionRequest) -> float:
    model = model_state["model"]
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded. Train the model before serving predictions.",
        )

    features = build_prediction_frame(payload)
    try:
        return float(model.predict_proba(features)[0][1])
    except Exception:
        logger.exception("Prediction failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prediction failed.",
        ) from None
