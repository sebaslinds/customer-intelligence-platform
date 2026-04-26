import json
import logging
from typing import Any, Literal

import pandas as pd
from fastapi import APIRouter, HTTPException, status
from openai import OpenAI, OpenAIError
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
    if not settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY is required to generate copilot insights.")

    client = OpenAI(api_key=settings.openai_api_key)
    selected_model = model or settings.openai_model

    response = client.chat.completions.create(
        model=selected_model,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a senior analytics copilot for an Instacart-style customer "
                    "intelligence platform. Use only the supplied Snowflake query results. "
                    "If the data is insufficient, say what is missing. Explain metric movements, "
                    "identify impacted customer or product segments, and recommend practical "
                    "business actions. Keep answers concise and evidence-based. Return only "
                    "valid JSON that matches this schema: "
                    f"{CopilotResponse.model_json_schema()}"
                ),
            },
            {
                "role": "user",
                "content": f"Question: {question}\n\nSnowflake context:\n{context}",
            },
        ],
    )

    content = response.choices[0].message.content
    if not content:
        raise ValueError("OpenAI returned an empty copilot response.")

    parsed = CopilotResponse.model_validate(json.loads(content))
    parsed.question = question
    return parsed


def get_metric(record: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = record.get(key, default)
    if value is None:
        return default
    return float(value)


def generate_local_insights(question: str, context: dict[str, Any]) -> CopilotResponse:
    churn_segments = context.get("churn_by_segment", [])
    high_churn = next(
        (segment for segment in churn_segments if segment.get("churn_risk_segment") == "high"),
        churn_segments[0] if churn_segments else {},
    )
    customer_behavior = (context.get("customer_behavior") or [{}])[0]
    retention_rows = context.get("cohort_retention", [])

    high_churn_probability = get_metric(high_churn, "avg_churn_probability")
    high_churn_users = int(get_metric(high_churn, "users"))
    avg_reorder_ratio = get_metric(customer_behavior, "avg_reorder_ratio")
    avg_days_between_orders = get_metric(customer_behavior, "avg_days_between_orders")

    latest_retention = retention_rows[0] if retention_rows else {}
    latest_retention_rate = get_metric(latest_retention, "retention_rate")

    explanation = (
        "Local fallback analysis suggests churn risk is concentrated among customers with weaker "
        "reorder behavior, fewer total orders, and longer gaps between purchases. "
        "OpenAI generation was unavailable, so this response was generated directly from Snowflake metrics."
    )

    return CopilotResponse(
        question=question,
        summary="Churn appears tied to lower reorder engagement and longer purchase intervals.",
        explanation=explanation,
        impacted_segments=[
            ImpactedSegment(
                segment_name=str(high_churn.get("churn_risk_segment", "high")),
                metric="avg_churn_probability",
                value=f"{high_churn_probability:.2%}",
                why_it_matters="This segment has the highest modeled churn risk and should be prioritized for retention actions.",
            ),
            ImpactedSegment(
                segment_name="overall_customer_base",
                metric="avg_days_between_orders",
                value=f"{avg_days_between_orders:.2f}",
                why_it_matters="Longer order gaps can indicate declining purchase intent.",
            ),
        ],
        recommendations=[
            Recommendation(
                action="Launch a targeted reorder campaign for high-risk customers.",
                expected_impact="Increase repeat purchase behavior among customers most likely to churn.",
                priority="high",
            ),
            Recommendation(
                action="Promote frequently reordered products in personalized offers.",
                expected_impact="Improve basket relevance and encourage faster next orders.",
                priority="medium",
            ),
            Recommendation(
                action="Monitor cohort retention and reorder ratio weekly.",
                expected_impact="Detect churn movement earlier and adjust campaigns faster.",
                priority="medium",
            ),
        ],
        insights=[
            Insight(
                title="High-risk customers need retention focus",
                category="churn",
                finding=f"The highest-risk segment contains {high_churn_users} users with average churn probability of {high_churn_probability:.2%}.",
                evidence=[
                    f"Average reorder ratio: {avg_reorder_ratio:.2%}",
                    f"Latest cohort retention rate: {latest_retention_rate:.2%}",
                ],
                impact="high",
                recommended_action="Prioritize retention campaigns for users with low reorder ratios and long order gaps.",
            )
        ],
        follow_up_questions=[
            "Which products are most associated with repeat purchases?",
            "Which customer segment has the longest days between orders?",
            "How does retention change by cohort period?",
        ],
        data_sources=[
            "customer_churn_probability",
            "feature_store",
            "customer_cohort_retention",
        ],
    )


def answer_business_question(question: str) -> CopilotResponse:
    settings = get_settings()
    category = classify_question(question)
    engine = build_snowflake_engine(settings)

    try:
        context = collect_business_context(engine, category)
        try:
            return generate_structured_insights(question=question, context=context)
        except (OpenAIError, ValueError, json.JSONDecodeError) as error:
            logger.warning("OpenAI copilot unavailable; using local metric fallback: %s", error)
            return generate_local_insights(question=question, context=context)
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
