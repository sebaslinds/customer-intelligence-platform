import logging

from fastapi import APIRouter, HTTPException, status

from config.settings import get_settings
from schemas.decision import DecisionRequest, DecisionResponse
from services.decision.engine import run_decision_engine

logger = logging.getLogger(__name__)
router = APIRouter(tags=["decision-engine"])
settings = get_settings()


@router.post("", response_model=DecisionResponse)
def create_decision(payload: DecisionRequest) -> DecisionResponse:
    try:
        return run_decision_engine(payload, settings=settings)
    except Exception:
        logger.exception("Decision engine failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Decision engine failed.",
        ) from None
