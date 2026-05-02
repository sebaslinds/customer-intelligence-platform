import logging
import tempfile
from pathlib import Path
from urllib.parse import urlparse
from typing import Any

import joblib
import pandas as pd
import snowflake.connector
from fastapi import HTTPException, status

from config.settings import Settings, get_settings
from ml.features import FEATURE_COLUMNS
from schemas.prediction import PredictionRequest

logger = logging.getLogger(__name__)

DEFAULT_FEATURE_COLUMNS = FEATURE_COLUMNS

model_state: dict[str, Any] = {"model": None, "feature_columns": DEFAULT_FEATURE_COLUMNS}


def load_model(model_path: str, model_uri: str | None = None, settings: Settings | None = None) -> Any:
    if model_uri:
        path = download_model_from_snowflake(model_uri, settings or get_settings())
    else:
        path = Path(model_path)

    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found: {path}")

    model = joblib.load(path)
    feature_columns = getattr(model, "feature_names_in_", DEFAULT_FEATURE_COLUMNS)
    model_state["model"] = model
    model_state["feature_columns"] = list(feature_columns)
    logger.info("Loaded model artifact from %s", path)
    return model


def download_model_from_snowflake(model_uri: str, settings: Settings) -> Path:
    stage_uri = normalize_snowflake_stage_uri(model_uri)
    cache_dir = Path(settings.model_cache_dir or tempfile.gettempdir()) / "customer-intelligence-models"
    cache_dir.mkdir(parents=True, exist_ok=True)

    target_file = cache_dir / Path(stage_uri).name
    if target_file.exists():
        target_file.unlink()

    connection = snowflake.connector.connect(
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

    try:
        file_uri = f"file://{cache_dir.as_posix()}"
        with connection.cursor() as cursor:
            cursor.execute(f"GET {stage_uri} '{file_uri}' OVERWRITE = TRUE")
    finally:
        connection.close()

    if not target_file.exists():
        downloaded_files = sorted(cache_dir.glob(Path(stage_uri).name + "*"))
        if downloaded_files:
            target_file = downloaded_files[0]

    if not target_file.exists():
        raise FileNotFoundError(f"Snowflake model artifact was not downloaded: {stage_uri}")

    logger.info("Downloaded model artifact from %s to %s", stage_uri, target_file)
    return target_file


def normalize_snowflake_stage_uri(model_uri: str) -> str:
    if model_uri.startswith("@"):
        return model_uri

    parsed_uri = urlparse(model_uri)
    if parsed_uri.scheme != "snowflake":
        raise ValueError("MODEL_URI must be a Snowflake stage path starting with @ or snowflake://")

    stage_name = parsed_uri.netloc
    stage_path = parsed_uri.path.lstrip("/")
    if not stage_name or not stage_path:
        raise ValueError("MODEL_URI must include a stage name and file path.")

    return f"@{stage_name}/{stage_path}"


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
