from fastapi import APIRouter

from schemas.decision import DecisionRequest, DecisionResponse
from services.decision.engine import run_decision_engine

router = APIRouter(tags=["decision"])


@router.post("", response_model=DecisionResponse)
def create_decision(request: DecisionRequest) -> DecisionResponse:
    return run_decision_engine(request)
