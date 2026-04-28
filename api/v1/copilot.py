from fastapi import APIRouter, HTTPException, status

from schemas.copilot import CopilotRequest, CopilotResponse
from services.ai.copilot_service import answer_business_question

router = APIRouter(prefix="/copilot", tags=["copilot"])


@router.post("/insights", response_model=CopilotResponse)
def create_copilot_insights(payload: CopilotRequest) -> CopilotResponse:
    try:
        return answer_business_question(payload.question)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate copilot insights.",
        ) from exc
