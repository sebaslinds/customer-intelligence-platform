import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.v1.router import api_router
from config.logging_config import configure_logging
from config.production import validate_production_settings
from config.settings import get_settings
from services.prediction.model_service import is_model_loaded, load_model

settings = get_settings()
validate_production_settings(settings)
configure_logging(settings.log_level)
logger = logging.getLogger(__name__)


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
app.include_router(api_router)


@app.get("/health", tags=["system"])
def health_check() -> dict[str, str | bool]:
    return {
        "status": "ok",
        "environment": settings.app_env,
        "model_loaded": is_model_loaded(),
    }
