import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from api.copilot import router as copilot_router
from config.logging_config import configure_logging
from config.production import validate_production_settings
from config.settings import get_settings

settings = get_settings()
validate_production_settings(settings)
configure_logging(settings.log_level)
logger = logging.getLogger(__name__)

DEFAULT_FEATURE_COLUMNS = [
    "total_orders",
    "avg_basket_size",
    "reorder_ratio",
    "unique_products",
    "days_between_orders",
]

model_state: dict[str, Any] = {"model": None, "feature_columns": DEFAULT_FEATURE_COLUMNS}


class PredictionRequest(BaseModel):
    total_orders: float = Field(..., ge=0)
    avg_basket_size: float = Field(..., ge=0)
    reorder_ratio: float = Field(..., ge=0, le=1)
    unique_products: float = Field(..., ge=0)
    days_between_orders: float = Field(..., ge=0)


class PredictionResponse(BaseModel):
    reorder_probability: float
    model_version: str = "random_forest_reorder_model"


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


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        load_model(settings.model_path)
    except FileNotFoundError:
        logger.warning("Model artifact is not available yet: %s", settings.model_path)
    except Exception:
        logger.exception("Failed to load model artifact")
        raise
    yield

app = FastAPI(
    title="Customer Intelligence Platform API",
    version="0.1.0",
    description="API surface for customer analytics and intelligence workflows.",
    lifespan=lifespan,
)
app.include_router(copilot_router)


@app.get("/health", tags=["system"])
def health_check() -> dict[str, str | bool]:
    return {
        "status": "ok",
        "environment": settings.app_env,
        "model_loaded": model_state["model"] is not None,
    }


@app.post("/predict", response_model=PredictionResponse, tags=["predictions"])
def predict(payload: PredictionRequest) -> PredictionResponse:
    model = model_state["model"]
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded. Train the model before serving predictions.",
        )

    features = build_prediction_frame(payload)
    try:
        probability = float(model.predict_proba(features)[0][1])
    except Exception:
        logger.exception("Prediction failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prediction failed.",
        ) from None

    return PredictionResponse(reorder_probability=probability)
