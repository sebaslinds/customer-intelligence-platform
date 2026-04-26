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
AIResponseSource = Literal["openai", "local_fallback"]


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
    ai_source: AIResponseSource = "local_fallback"
    ai_source_detail: str | None = None
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
    if any(term in normalized for term in ("lifetime value", "clv", "value segment")):
        return "clv"
    if any(term in normalized for term in ("top product", "products", "items", "sku")):
        return "top_products"
    if any(term in normalized for term in ("reorder", "reorders", "repeat")):
        return "reorder_trend"
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
    summary_query = """
        select
            count(*) as total_users,
            avg(total_orders) as avg_total_orders,
            avg(avg_basket_size) as avg_basket_size,
            avg(reorder_ratio) as avg_reorder_ratio,
            avg(unique_products) as avg_unique_products,
            avg(days_between_orders) as avg_days_between_orders
        from feature_store
    """
    segment_query = """
        with segmented as (
            select
                case
                    when total_orders >= 80 then 'power_customers_80_plus_orders'
                    when total_orders >= 40 then 'frequent_customers_40_to_79_orders'
                    when total_orders >= 10 then 'developing_customers_10_to_39_orders'
                    else 'low_frequency_customers_under_10_orders'
                end as customer_segment,
                user_id,
                total_orders,
                avg_basket_size,
                reorder_ratio,
                unique_products,
                days_between_orders
            from feature_store
        )

        select
            customer_segment,
            count(*) as users,
            avg(total_orders) as avg_total_orders,
            avg(avg_basket_size) as avg_basket_size,
            avg(reorder_ratio) as avg_reorder_ratio,
            avg(unique_products) as avg_unique_products,
            avg(days_between_orders) as avg_days_between_orders,
            max(days_between_orders) as max_days_between_orders
        from segmented
        group by customer_segment
        order by avg_days_between_orders desc
    """
    longest_gap_query = """
        select
            user_id,
            total_orders,
            observed_basket_orders,
            avg_basket_size,
            reorder_ratio,
            unique_products,
            days_between_orders
        from feature_store
        order by days_between_orders desc
        limit 20
    """
    return {
        "customer_behavior": run_query(engine, summary_query),
        "customer_segments": run_query(engine, segment_query),
        "customers_with_longest_order_gaps": run_query(engine, longest_gap_query),
    }


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
    parsed.ai_source = "openai"
    parsed.ai_source_detail = f"Generated by OpenAI model {selected_model}."
    return parsed


def describe_openai_fallback_reason(error: Exception) -> str:
    if isinstance(error, ValueError):
        return str(error)
    return f"OpenAI request failed: {error.__class__.__name__}"


def get_metric(record: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = record.get(key, default)
    if value is None:
        return default
    return float(value)


def format_metric_percent(value: float) -> str:
    return f"{value:.2%}"


def format_metric_number(value: float) -> str:
    return f"{value:,.2f}"


def generate_churn_fallback(question: str, context: dict[str, Any]) -> CopilotResponse:
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
                value=format_metric_percent(high_churn_probability),
                why_it_matters="This segment has the highest modeled churn risk and should be prioritized for retention actions.",
            ),
            ImpactedSegment(
                segment_name="overall_customer_base",
                metric="avg_days_between_orders",
                value=format_metric_number(avg_days_between_orders),
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
                finding=f"The highest-risk segment contains {high_churn_users} users with average churn probability of {format_metric_percent(high_churn_probability)}.",
                evidence=[
                    f"Average reorder ratio: {format_metric_percent(avg_reorder_ratio)}",
                    f"Latest cohort retention rate: {format_metric_percent(latest_retention_rate)}",
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


def generate_retention_fallback(question: str, context: dict[str, Any]) -> CopilotResponse:
    retention_rows = context.get("cohort_retention", [])
    if not retention_rows:
        return generate_general_fallback(question, context)

    lowest_row = min(retention_rows, key=lambda row: get_metric(row, "retention_rate", 1.0))
    latest_row = retention_rows[0]
    lowest_rate = get_metric(lowest_row, "retention_rate")
    latest_rate = get_metric(latest_row, "retention_rate")
    cohort_period = int(get_metric(lowest_row, "cohort_period"))
    cohort_order_number = int(get_metric(lowest_row, "cohort_order_number"))

    return CopilotResponse(
        question=question,
        summary="Retention varies by cohort period, with the weakest periods showing where repeat-order behavior drops.",
        explanation=(
            "This fallback response uses the cohort retention mart directly. It compares cohort periods "
            "and highlights the period with the lowest observed retention rate."
        ),
        impacted_segments=[
            ImpactedSegment(
                segment_name=f"cohort_order_{cohort_order_number}_period_{cohort_period}",
                metric="retention_rate",
                value=format_metric_percent(lowest_rate),
                why_it_matters="This is the weakest observed cohort-period combination and should be reviewed for retention actions.",
            ),
            ImpactedSegment(
                segment_name="latest_observed_cohort_period",
                metric="retention_rate",
                value=format_metric_percent(latest_rate),
                why_it_matters="The latest cohort period is useful as a directional benchmark for current retention behavior.",
            ),
        ],
        recommendations=[
            Recommendation(
                action="Compare offers and reorder reminders around the weakest cohort period.",
                expected_impact="Identify where customers need a stronger nudge to place the next order.",
                priority="high",
            ),
            Recommendation(
                action="Track retention by cohort period weekly.",
                expected_impact="Separate normal cohort aging from actual retention deterioration.",
                priority="medium",
            ),
        ],
        insights=[
            Insight(
                title="Retention should be evaluated by cohort period",
                category="retention",
                finding=(
                    f"The lowest observed retention rate is {format_metric_percent(lowest_rate)} "
                    f"for cohort order {cohort_order_number}, period {cohort_period}."
                ),
                evidence=[
                    f"Latest observed retention rate: {format_metric_percent(latest_rate)}",
                    f"Lowest cohort period: {cohort_period}",
                ],
                impact="medium",
                recommended_action="Focus analysis on the cohort periods with the steepest retention decline.",
            )
        ],
        follow_up_questions=[
            "Which customer segment has the longest days between orders?",
            "Which products are most associated with repeat purchases?",
            "How can we improve retention?",
        ],
        data_sources=["customer_cohort_retention"],
    )


def generate_top_products_fallback(question: str, context: dict[str, Any]) -> CopilotResponse:
    products = context.get("top_products", [])
    if not products:
        return generate_general_fallback(question, context)

    top_product = products[0]
    reorder_leader = max(products, key=lambda row: get_metric(row, "reorder_ratio"))
    top_name = str(top_product.get("product_name", "Unknown product"))
    repeat_name = str(reorder_leader.get("product_name", "Unknown product"))

    return CopilotResponse(
        question=question,
        summary=f"{top_name} leads by order volume, while {repeat_name} has the strongest repeat-purchase signal among top products.",
        explanation=(
            "This fallback response compares product order volume and reorder ratio from the product mart. "
            "Volume leaders are useful for traffic, while high reorder-ratio products are stronger candidates for retention offers."
        ),
        impacted_segments=[
            ImpactedSegment(
                segment_name=top_name,
                metric="order_line_count",
                value=format_metric_number(get_metric(top_product, "order_line_count")),
                why_it_matters="High-volume products shape the largest share of basket activity.",
            ),
            ImpactedSegment(
                segment_name=repeat_name,
                metric="reorder_ratio",
                value=format_metric_percent(get_metric(reorder_leader, "reorder_ratio")),
                why_it_matters="High repeat behavior indicates a good candidate for personalized reorder prompts.",
            ),
        ],
        recommendations=[
            Recommendation(
                action="Use high reorder-ratio products in retention campaigns.",
                expected_impact="Increase relevance of reorder reminders and improve repeat purchase conversion.",
                priority="high",
            ),
            Recommendation(
                action="Merchandise high-volume products with complementary items.",
                expected_impact="Improve basket expansion while preserving popular product demand.",
                priority="medium",
            ),
        ],
        insights=[
            Insight(
                title="Separate volume leaders from repeat-purchase leaders",
                category="top_products",
                finding=f"{top_name} has the highest order volume, while {repeat_name} has the strongest reorder ratio in the sampled product set.",
                evidence=[
                    f"{top_name} order lines: {format_metric_number(get_metric(top_product, 'order_line_count'))}",
                    f"{repeat_name} reorder ratio: {format_metric_percent(get_metric(reorder_leader, 'reorder_ratio'))}",
                ],
                impact="medium",
                recommended_action="Build separate campaigns for high-volume acquisition products and high-repeat retention products.",
            )
        ],
        follow_up_questions=[
            "Which products drive repeat purchases?",
            "Why are reorders dropping?",
            "Which customer segments are at churn risk?",
        ],
        data_sources=["dim_products"],
    )


def generate_reorder_fallback(question: str, context: dict[str, Any]) -> CopilotResponse:
    reorder_rows = context.get("reorder_trend", [])
    if not reorder_rows:
        return generate_general_fallback(question, context)

    latest = reorder_rows[0]
    valid_rows = [row for row in reorder_rows if row.get("reorder_ratio") is not None]
    avg_reorder = sum(get_metric(row, "reorder_ratio") for row in valid_rows) / max(len(valid_rows), 1)
    latest_reorder = get_metric(latest, "reorder_ratio")

    direction = "above" if latest_reorder >= avg_reorder else "below"
    return CopilotResponse(
        question=question,
        summary=f"Latest reorder ratio is {direction} the recent average in the available order-number trend.",
        explanation=(
            "This fallback response compares the most recent order-number reorder ratio to the recent average. "
            "Because prior item-level baskets are partially sampled, treat this as a directional simulation."
        ),
        impacted_segments=[
            ImpactedSegment(
                segment_name=f"order_number_{int(get_metric(latest, 'order_number'))}",
                metric="reorder_ratio",
                value=format_metric_percent(latest_reorder),
                why_it_matters="The latest order-number bucket shows the most recent reorder behavior in the trend table.",
            ),
            ImpactedSegment(
                segment_name="recent_order_number_average",
                metric="reorder_ratio",
                value=format_metric_percent(avg_reorder),
                why_it_matters="The recent average provides context for whether the latest reorder level is unusually high or low.",
            ),
        ],
        recommendations=[
            Recommendation(
                action="Monitor reorder ratio alongside observed basket coverage.",
                expected_impact="Avoid overreacting to simulation artifacts from partially observed basket lines.",
                priority="high",
            ),
            Recommendation(
                action="Promote high-repeat products to customers with long order gaps.",
                expected_impact="Increase reorder likelihood among customers showing weaker recent engagement.",
                priority="medium",
            ),
        ],
        insights=[
            Insight(
                title="Reorder trend needs coverage context",
                category="reorder_trend",
                finding=f"Latest reorder ratio is {format_metric_percent(latest_reorder)} versus a recent average of {format_metric_percent(avg_reorder)}.",
                evidence=[
                    f"Latest order count: {format_metric_number(get_metric(latest, 'total_orders'))}",
                    f"Latest average basket size: {format_metric_number(get_metric(latest, 'avg_basket_size'))}",
                ],
                impact="medium",
                recommended_action="Interpret reorder movement together with product-line coverage and customer order gaps.",
            )
        ],
        follow_up_questions=[
            "Which products are most associated with repeat purchases?",
            "Which customer segment has the longest days between orders?",
            "How can we improve retention?",
        ],
        data_sources=["fct_orders"],
    )


def generate_customer_behavior_fallback(question: str, context: dict[str, Any]) -> CopilotResponse:
    customer_behavior = (context.get("customer_behavior") or [{}])[0]
    customer_segments = context.get("customer_segments") or []
    total_users = int(get_metric(customer_behavior, "total_users"))
    avg_total_orders = get_metric(customer_behavior, "avg_total_orders")
    avg_days_between_orders = get_metric(customer_behavior, "avg_days_between_orders")
    avg_reorder_ratio = get_metric(customer_behavior, "avg_reorder_ratio")
    longest_gap_segment = (
        max(customer_segments, key=lambda segment: get_metric(segment, "avg_days_between_orders"))
        if customer_segments
        else {}
    )
    segment_name = str(longest_gap_segment.get("customer_segment", "overall_customer_base"))
    segment_avg_gap = get_metric(longest_gap_segment, "avg_days_between_orders", avg_days_between_orders)
    segment_users = int(get_metric(longest_gap_segment, "users", total_users))

    return CopilotResponse(
        question=question,
        summary=f"{segment_name} has the longest average days between orders at {format_metric_number(segment_avg_gap)} days.",
        explanation=(
            "This fallback response segments customers by total order frequency, then compares average days "
            "between orders across those segments. This directly answers which segment has the longest purchase interval."
        ),
        impacted_segments=[
            ImpactedSegment(
                segment_name=segment_name,
                metric="avg_days_between_orders",
                value=format_metric_number(segment_avg_gap),
                why_it_matters="This segment waits the longest between orders, making it a priority for lifecycle nudges.",
            ),
            ImpactedSegment(
                segment_name="overall_customer_base",
                metric="avg_days_between_orders",
                value=format_metric_number(avg_days_between_orders),
                why_it_matters="This is the baseline used to compare the longest-gap segment.",
            ),
        ],
        recommendations=[
            Recommendation(
                action=f"Create a reactivation campaign for {segment_name}.",
                expected_impact="Reduce the longest purchase gaps and move customers toward more frequent repeat orders.",
                priority="high",
            ),
            Recommendation(
                action="Use reorder reminders and high-repeat products for customers with long gaps.",
                expected_impact="Improve relevance of lifecycle marketing and shorten time to next order.",
                priority="medium",
            ),
        ],
        insights=[
            Insight(
                title="Longest order gaps are segment-specific",
                category="customer_behavior",
                finding=(
                    f"{segment_name} includes {segment_users:,} users and has the longest average order gap "
                    f"at {format_metric_number(segment_avg_gap)} days."
                ),
                evidence=[
                    f"Overall average days between orders: {format_metric_number(avg_days_between_orders)}",
                    f"Average reorder ratio: {format_metric_percent(avg_reorder_ratio)}",
                    f"Average total orders across all customers: {format_metric_number(avg_total_orders)}",
                ],
                impact="medium",
                recommended_action="Prioritize segments with the longest order gaps for reactivation campaigns.",
            )
        ],
        follow_up_questions=[
            "Which specific customers have the longest days between orders?",
            "Which customer segments are at churn risk?",
            "Which products drive repeat purchases?",
        ],
        data_sources=["feature_store", "customer_segments", "customers_with_longest_order_gaps"],
    )


def generate_clv_fallback(question: str, context: dict[str, Any]) -> CopilotResponse:
    segments = context.get("lifetime_value_by_segment", [])
    if not segments:
        return generate_general_fallback(question, context)

    top_segment = segments[0]
    segment_name = str(top_segment.get("value_segment", "highest_value"))
    avg_clv = get_metric(top_segment, "avg_lifetime_value_proxy")
    avg_future_value = get_metric(top_segment, "avg_future_value_proxy")

    return CopilotResponse(
        question=question,
        summary=f"{segment_name} customers represent the strongest lifetime value opportunity.",
        explanation="This fallback response uses the customer lifetime value mart to compare value segments.",
        impacted_segments=[
            ImpactedSegment(
                segment_name=segment_name,
                metric="avg_lifetime_value_proxy",
                value=format_metric_number(avg_clv),
                why_it_matters="This segment has the strongest estimated lifetime value proxy.",
            )
        ],
        recommendations=[
            Recommendation(
                action="Protect high-value customers with personalized retention offers.",
                expected_impact="Preserve future value from the most commercially important segment.",
                priority="high",
            ),
            Recommendation(
                action="Create upgrade campaigns for medium-value customers.",
                expected_impact="Move customers toward higher future-value behavior.",
                priority="medium",
            ),
        ],
        insights=[
            Insight(
                title="Lifetime value is concentrated by segment",
                category="clv",
                finding=f"{segment_name} customers average {format_metric_number(avg_clv)} in lifetime value proxy.",
                evidence=[f"Average future value proxy: {format_metric_number(avg_future_value)}"],
                impact="high",
                recommended_action="Use CLV segments to prioritize retention and merchandising campaigns.",
            )
        ],
        follow_up_questions=[
            "Which customer segments are at churn risk?",
            "How can we improve retention?",
            "Which products drive repeat purchases?",
        ],
        data_sources=["customer_lifetime_value"],
    )


def generate_general_fallback(question: str, context: dict[str, Any]) -> CopilotResponse:
    customer_behavior = (context.get("customer_behavior") or [{}])[0]
    return CopilotResponse(
        question=question,
        summary="The strongest available signals are customer frequency, reorder behavior, product repeatability, and retention cohorts.",
        explanation=(
            "OpenAI generation was unavailable, so this local response summarizes the available Snowflake context. "
            "Ask a more specific question about churn, retention, products, reorders, CLV, or customer behavior for a targeted answer."
        ),
        impacted_segments=[
            ImpactedSegment(
                segment_name="overall_customer_base",
                metric="avg_reorder_ratio",
                value=format_metric_percent(get_metric(customer_behavior, "avg_reorder_ratio")),
                why_it_matters="Reorder behavior is a core signal for repeat purchase health.",
            )
        ],
        recommendations=[
            Recommendation(
                action="Ask a focused follow-up about churn, retention, products, or CLV.",
                expected_impact="Generate a more specific metric-backed recommendation.",
                priority="medium",
            )
        ],
        insights=[
            Insight(
                title="Use a focused question for sharper insight",
                category="general",
                finding="The local fallback can answer specific metric categories using Snowflake marts.",
                evidence=list(context.keys()),
                impact="low",
                recommended_action="Choose one business area: churn, retention, reorder trend, products, CLV, or customer behavior.",
            )
        ],
        follow_up_questions=[
            "Why is churn increasing?",
            "Which products are most associated with repeat purchases?",
            "How does retention change by cohort period?",
        ],
        data_sources=list(context.keys()),
    )


def generate_local_insights(question: str, context: dict[str, Any], category: InsightCategory) -> CopilotResponse:
    fallback_generators = {
        "churn": generate_churn_fallback,
        "retention": generate_retention_fallback,
        "top_products": generate_top_products_fallback,
        "reorder_trend": generate_reorder_fallback,
        "customer_behavior": generate_customer_behavior_fallback,
        "clv": generate_clv_fallback,
        "general": generate_general_fallback,
    }
    return fallback_generators[category](question, context)


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
            fallback_response = generate_local_insights(question=question, context=context, category=category)
            fallback_response.ai_source_detail = describe_openai_fallback_reason(error)
            return fallback_response
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
