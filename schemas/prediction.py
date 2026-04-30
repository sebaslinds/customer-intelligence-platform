from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    total_orders: float = Field(..., ge=0)
    observed_basket_orders: float = Field(0, ge=0)
    avg_basket_size: float = Field(..., ge=0)
    reorder_ratio: float = Field(..., ge=0, le=1)
    prior_reorder_order_ratio: float = Field(0, ge=0, le=1)
    unique_products: float = Field(..., ge=0)
    unique_departments: float = Field(0, ge=0)
    unique_aisles: float = Field(0, ge=0)
    produce_item_ratio: float = Field(0, ge=0, le=1)
    dairy_eggs_item_ratio: float = Field(0, ge=0, le=1)
    fresh_fruits_item_ratio: float = Field(0, ge=0, le=1)
    fresh_vegetables_item_ratio: float = Field(0, ge=0, le=1)
    days_between_orders: float = Field(..., ge=0)
    stddev_days_between_orders: float = Field(0, ge=0)
    days_since_last_order: float = Field(0, ge=0)
    customer_tenure_days: float = Field(0, ge=0)
    order_frequency_30d: float = Field(0, ge=0)
    avg_order_dow: float = Field(0, ge=0, le=6)
    avg_order_hour_of_day: float = Field(0, ge=0, le=23)
    weekend_order_ratio: float = Field(0, ge=0, le=1)
    evening_order_ratio: float = Field(0, ge=0, le=1)


class PredictionResponse(BaseModel):
    reorder_probability: float
    model_version: str = "random_forest_reorder_model"
