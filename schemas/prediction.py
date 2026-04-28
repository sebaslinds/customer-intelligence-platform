from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    total_orders: float = Field(..., ge=0)
    avg_basket_size: float = Field(..., ge=0)
    reorder_ratio: float = Field(..., ge=0, le=1)
    unique_products: float = Field(..., ge=0)
    days_between_orders: float = Field(..., ge=0)


class PredictionResponse(BaseModel):
    reorder_probability: float
    model_version: str = "random_forest_reorder_model"
