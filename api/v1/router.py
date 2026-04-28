from fastapi import APIRouter

from api.v1.decision import router as decision_router

api_router = APIRouter()
api_router.include_router(decision_router, prefix="/decision")
