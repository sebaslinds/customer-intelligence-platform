import logging
from typing import Any, Literal

import pandas as pd
from fastapi import APIRouter, HTTPException, status
from openai import OpenAI
from pydantic import BaseModel, Field
from sqlalchemy import Engine, text

from config.settings import get_settings
from ingestion.snowflake_client import build_snowflake_engine

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/copilot", tags=["copilot"])

InsightCategory = Literal["churn", "reorder_trend", "top_products", "customer_behavior", "retention", "clv", "general"]
ImpactLevel = Literal["low", "medium", "high"]


class CopilotRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)


class Insight(BaseModel):
    title: str
    category: InsightCategory
    finding: str
    evidence: list[str]
    impact: ImpactLevel
    recommended_action: str


class ImpactedSegment(BaseModel):
    segment_name: str
    metric: str
    value: str
    why_it_matters: str


class Recommendation(BaseModel):
    action: str
    expected_impact: str
    priority: ImpactLevel


class CopilotResponse(BaseModel):
    question: str
    summary: str
    explanation: str
    impacted_segments: list[ImpactedSegment]
    recommendations: list[Recommendation]
    insights: list[Insight]
    follow_up_questions: list[str]
    data_sources: list[str]


def classify_question(question: str) -> InsightCategory:
    normalized = question.lower()
    if any(term in normalized for term in ("churn", "attrition", "drop off", "dropping")):
        return "churn"
    if any(term in normalized for term in ("cohort", "retention")):
        return "retention"
    if any(term in normalized for term in ("reorder", "reorders", "repeat")):
        return "reorder_trend"
    if any(term in normalized for term in ("lifetime value", "clv", "value segment")):
        return "clv"
    if any(term in normalized for term in ("top product", "products", "items", "sku")):
        return "top_products"
    if any(term in normalized for term in ("customer", "user", "basket", "order")):
        return "customer_behavior"
    return "general"


def run_query(engine: Engine, query: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    logger.info("Running copilot context query")
    with engine.begin() as connection:
        frame = pd.read_sql_query(text(query), connection, params=params)

    frame.columns = [column.lower() for column in frame.columns]
    return frame.to_dict(orient="records")


def get_reorder_context(engine: Engine) -> dict[str, Any]:
    query = """
        select
            order_number,
            count(*) as total_orders,
            avg(item_count) as avg_basket_size,
            avg(reordered_item_count / nullif(item_count, 0)) as reorder_ratio
        from fct_orders
        group by order_number
        order by order_number desc
        limit 20
    """
    return {"reorder_trend": run_query(engine, query)}


def get_top_products_context(engine: Engine) -> dict[str, Any]:
    query = """
        select
            product_id,
            product_name,
            department,
            aisle,
            order_line_count,
            reordered_line_count,
            reordered_line_count / nullif(order_line_count, 0) as reorder_ratio
        from dim_products
        order by order_line_count desc
        limit 20
    """
    return {"top_products": run_query(engine, query)}


def get_customer_behavior_context(engine: Engine) -> dict[str, Any]:
    query = """
        select
            count(*) as total_users,
            avg(total_orders) as avg_total_orders,
            avg(avg_basket_size) as avg_basket_size,
            avg(reorder_ratio) as avg_reorder_ratio,
            avg(unique_products) as avg_unique_products,
            avg(days_between_orders) as avg_days_between_orders
        from feature_store
    """
    return {"customer_behavior": run_query(engine, query)}


def get_churn_context(engine: Engine) -> dict[str, Any]:
    segment_query = """
        select
            churn_risk_segment,
            count(*) as users,
            avg(churn_probability) as avg_churn_probability,
            avg(total_orders) as avg_total_orders,
            avg(avg_basket_size) as avg_basket_size,
            avg(reorder_ratio) as avg_reorder_ratio,
            avg(unique_products) as avg_unique_products,
            avg(days_between_orders) as avg_days_between_orders
        from customer_churn_probability
        group by churn_risk_segment
        order by avg_churn_probability desc
    """
    driver_query = """
        select
            user_id,
            churn_probability,
            churn_risk_segment,
            total_orders,
            avg_basket_size,
            reorder_ratio,
            unique_products,
            days_between_orders
        from customer_churn_probability
        order by churn_probability desc
        limit 25
    """
    return {
        "churn_by_segment": run_query(engine, segment_query),
        "highest_churn_users": run_query(engine, driver_query),
    }


def get_retention_context(engine: Engine) -> dict[str, Any]:
    query = """
        select
            cohort_order_number,
            cohort_period,
            cohort_size,
            active_users,
            retention_rate,
            revenue_proxy
        from customer_cohort_retention
        order by cohort_order_number desc, cohort_period asc
        limit 50
    """
    return {"cohort_retention": run_query(engine, query)}


def get_lifetime_value_context(engine: Engine) -> dict[str, Any]:
    segment_query = """
        select
            value_segment,
            count(*) as users,
            avg(customer_lifetime_value_proxy) as avg_lifetime_value_proxy,
            avg(historical_value_proxy) as avg_historical_value_proxy,
            avg(predicted_future_value_proxy) as avg_future_value_proxy,
            avg(churn_probability) as avg_churn_probability
        from customer_lifetime_value
        group by value_segment
        order by avg_lifetime_value_proxy desc
    """
    top_customer_query = """
        select
            user_id,
            value_segment,
            customer_lifetime_value_proxy,
            historical_value_proxy,
            predicted_future_value_proxy,
            churn_probability,
            churn_risk_segment
        from customer_lifetime_value
        order by customer_lifetime_value_proxy desc
        limit 25
    """
    return {
        "lifetime_value_by_segment": run_query(engine, segment_query),
        "top_lifetime_value_customers": run_query(engine, top_customer_query),
    }


def collect_business_context(engine: Engine, category: InsightCategory) -> dict[str, Any]:
    context: dict[str, Any] = {}

    if category in ("churn", "general"):
        context.update(get_churn_context(engine))
        context.update(get_retention_context(engine))
        context.update(get_customer_behavior_context(engine))
    if category in ("reorder_trend", "general"):
        context.update(get_reorder_context(engine))
    if category in ("top_products", "general"):
        context.update(get_top_products_context(engine))
    if category in ("customer_behavior", "general"):
        context.update(get_customer_behavior_context(engine))
    if category in ("retention", "general"):
        context.update(get_retention_context(engine))
    if category in ("clv", "general"):
        context.update(get_lifetime_value_context(engine))

    return context


def generate_structured_insights(
    question: str,
    context: dict[str, Any],
    model: str | None = None,
) -> CopilotResponse:
    settings = get_settings()
    client = OpenAI()
    selected_model = model or settings.openai_model

    response = client.responses.parse(
        model=selected_model,
        input=[
            {
                "role": "system",
                "content": (
                    "You are a senior analytics copilot for an Instacart-style customer "
                    "intelligence platform. Use only the supplied Snowflake query results. "
                    "If the data is insufficient, say what is missing. Explain metric movements, "
                    "identify impacted customer or product segments, and recommend practical "
                    "business actions. Keep answers concise and evidence-based."
                ),
            },
            {
                "role": "user",
                "content": f"Question: {question}\n\nSnowflake context:\n{context}",
            },
        ],
        text_format=CopilotResponse,
    )

    parsed = response.output_parsed
    parsed.question = question
    return parsed


def answer_business_question(question: str) -> CopilotResponse:
    settings = get_settings()
    category = classify_question(question)
    engine = build_snowflake_engine(settings)

    try:
        context = collect_business_context(engine, category)
        return generate_structured_insights(question=question, context=context)
    except Exception:
        logger.exception("Copilot insight generation failed")
        raise
    finally:
        engine.dispose()


@router.post("/insights", response_model=CopilotResponse)
def create_copilot_insights(payload: CopilotRequest) -> CopilotResponse:
    try:
        return answer_business_question(payload.question)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate copilot insights.",
        ) from exc
