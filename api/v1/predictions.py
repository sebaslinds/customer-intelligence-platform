from fastapi import APIRouter

from schemas.prediction import PredictionRequest, PredictionResponse
from services.prediction.model_service import predict_reorder_probability

router = APIRouter(tags=["predictions"])


@router.post("", response_model=PredictionResponse)
def predict(payload: PredictionRequest) -> PredictionResponse:
    probability = predict_reorder_probability(payload)
    return PredictionResponse(reorder_probability=probability)
