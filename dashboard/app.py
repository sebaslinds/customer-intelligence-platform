import json
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import streamlit as st
from sqlalchemy import text

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.logging_config import configure_logging  # noqa: E402
from config.production import validate_production_settings  # noqa: E402
from config.settings import get_settings  # noqa: E402
from ingestion.snowflake_client import build_snowflake_engine  # noqa: E402

settings = get_settings()
validate_production_settings(settings)
configure_logging(settings.log_level)
logger = logging.getLogger(__name__)

MODEL_METRICS_PATH = PROJECT_ROOT / "ml" / "artifacts" / "training_metrics.json"
FEATURE_IMPORTANCE_PATH = PROJECT_ROOT / "ml" / "artifacts" / "feature_importance.csv"
COPILOT_TIMEOUT_SECONDS = 60
API_HEALTH_TIMEOUT_SECONDS = 45
STREAMLIT_APP_URL = "https://customer-intelligence-platform-d2pmcjetsrlgm2zwep7vgf.streamlit.app/"
RENDER_API_URL = "https://customer-intelligence-platform-3v6q.onrender.com"


st.set_page_config(
    page_title="Customer Intelligence Platform",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(ttl=600, show_spinner=False)
def query_snowflake(query: str) -> pd.DataFrame:
    engine = build_snowflake_engine(get_settings())
    try:
        with engine.begin() as connection:
            frame = pd.read_sql_query(text(query), connection)
    finally:
        engine.dispose()

    frame.columns = [column.lower() for column in frame.columns]
    return frame


def load_kpis() -> dict[str, Any]:
    query = """
        select
            count(distinct order_id) as total_orders,
            sum(item_count) as revenue_proxy,
            avg(reordered_item_count / nullif(item_count, 0)) as reorder_rate
        from fct_orders
    """
    frame = query_snowflake(query)
    if frame.empty:
        return {"total_orders": 0, "revenue_proxy": 0, "reorder_rate": 0}
    return frame.iloc[0].to_dict()


def load_customer_insights() -> pd.DataFrame:
    query = """
        select
            user_id,
            total_orders,
            observed_basket_orders,
            avg_basket_size,
            reorder_ratio,
            unique_products,
            days_between_orders
        from feature_store
        order by total_orders desc, user_id
        limit 25
    """
    return query_snowflake(query)


def load_product_trends() -> pd.DataFrame:
    query = """
        select
            product_name,
            department,
            aisle,
            order_line_count,
            reordered_line_count,
            reordered_line_count / nullif(order_line_count, 0) as reorder_rate
        from dim_products
        order by order_line_count desc
        limit 25
    """
    return query_snowflake(query)


def load_data_quality_results() -> pd.DataFrame:
    query = """
        with checks as (
            select
                'raw_instacart_orders' as table_name,
                'order_id_not_null' as check_name,
                count(*) as invalid_count
            from raw_instacart_orders
            where order_id is null

            union all

            select
                'raw_instacart_orders' as table_name,
                'order_id_unique' as check_name,
                count(*) as invalid_count
            from (
                select order_id
                from raw_instacart_orders
                group by order_id
                having count(*) > 1
            )

            union all

            select
                'raw_instacart_products' as table_name,
                'product_id_not_null' as check_name,
                count(*) as invalid_count
            from raw_instacart_products
            where product_id is null

            union all

            select
                'raw_instacart_products' as table_name,
                'product_id_unique' as check_name,
                count(*) as invalid_count
            from (
                select product_id
                from raw_instacart_products
                group by product_id
                having count(*) > 1
            )

            union all

            select
                'raw_instacart_order_products_train' as table_name,
                'order_id_not_null' as check_name,
                count(*) as invalid_count
            from raw_instacart_order_products_train
            where order_id is null

            union all

            select
                'raw_instacart_order_products_train' as table_name,
                'product_id_not_null' as check_name,
                count(*) as invalid_count
            from raw_instacart_order_products_train
            where product_id is null

            union all

            select
                'raw_instacart_order_products_train' as table_name,
                'order_product_unique' as check_name,
                count(*) as invalid_count
            from (
                select order_id, product_id
                from raw_instacart_order_products_train
                group by order_id, product_id
                having count(*) > 1
            )

            union all

            select
                'raw_instacart_order_products_train' as table_name,
                'product_id_valid' as check_name,
                count(*) as invalid_count
            from raw_instacart_order_products_train as order_products
            left join raw_instacart_products as products
                on order_products.product_id = products.product_id
            where order_products.product_id is not null
                and products.product_id is null
        )

        select
            table_name,
            check_name,
            invalid_count,
            case when invalid_count = 0 then 'Passed' else 'Failed' end as status
        from checks
        order by table_name, check_name
    """
    return query_snowflake(query)


def load_raw_table_counts() -> pd.DataFrame:
    query = """
        select 'raw_instacart_orders' as table_name, count(*) as row_count
        from raw_instacart_orders

        union all

        select 'raw_instacart_products' as table_name, count(*) as row_count
        from raw_instacart_products

        union all

        select 'raw_instacart_order_products_train' as table_name, count(*) as row_count
        from raw_instacart_order_products_train

        union all

        select 'raw_instacart_aisles' as table_name, count(*) as row_count
        from raw_instacart_aisles

        union all

        select 'raw_instacart_departments' as table_name, count(*) as row_count
        from raw_instacart_departments
    """
    return query_snowflake(query)


def load_mart_table_counts() -> pd.DataFrame:
    query = """
        select 'fct_orders' as table_name, count(*) as row_count
        from fct_orders

        union all

        select 'dim_products' as table_name, count(*) as row_count
        from dim_products

        union all

        select 'dim_users' as table_name, count(*) as row_count
        from dim_users

        union all

        select 'feature_store' as table_name, count(*) as row_count
        from feature_store

        union all

        select 'customer_cohort_retention' as table_name, count(*) as row_count
        from customer_cohort_retention

        union all

        select 'customer_churn_probability' as table_name, count(*) as row_count
        from customer_churn_probability

        union all

        select 'customer_lifetime_value' as table_name, count(*) as row_count
        from customer_lifetime_value
    """
    return query_snowflake(query)


def load_snowflake_connection_health() -> dict[str, Any]:
    query = """
        select
            current_account() as account_name,
            current_database() as database_name,
            current_schema() as schema_name,
            current_warehouse() as warehouse_name,
            current_role() as role_name
    """
    frame = query_snowflake(query)
    if frame.empty:
        return {"status": "Failed"}

    health = frame.iloc[0].to_dict()
    health["status"] = "Connected"
    return health


def load_api_health() -> dict[str, Any]:
    api_base_url = settings.api_base_url.rstrip("/")
    response = requests.get(f"{api_base_url}/health", timeout=API_HEALTH_TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.json()


def format_api_health_detail(api_detail: dict[str, Any]) -> str:
    if error := api_detail.get("error"):
        return str(error)
    if "model_loaded" in api_detail:
        return f"model_loaded={api_detail.get('model_loaded')}"
    return "No health payload returned"


@st.cache_data(show_spinner=False)
def load_model_metrics() -> dict[str, Any]:
    if not MODEL_METRICS_PATH.exists():
        return {}

    with MODEL_METRICS_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


@st.cache_data(show_spinner=False)
def load_feature_importance() -> pd.DataFrame:
    if not FEATURE_IMPORTANCE_PATH.exists():
        return pd.DataFrame(columns=["feature", "importance"])

    return pd.read_csv(FEATURE_IMPORTANCE_PATH)


def request_copilot_insights(question: str) -> dict[str, Any]:
    api_base_url = settings.api_base_url.rstrip("/")
    response = requests.post(
        f"{api_base_url}/copilot/insights",
        json={"question": question},
        timeout=COPILOT_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


def queue_copilot_question(question: str) -> None:
    st.session_state.pending_copilot_question = question
    st.rerun()


def format_number(value: Any) -> str:
    if pd.isna(value):
        return "0"
    return f"{float(value):,.0f}"


def format_percent(value: Any) -> str:
    if pd.isna(value):
        return "0.0%"
    return f"{float(value):.1%}"


def parse_metric_value(value: Any) -> float | None:
    if value is None or isinstance(value, dict | list | tuple):
        return None

    text_value = str(value).strip().replace(",", "")
    if not text_value:
        return None

    if text_value.endswith("%"):
        text_value = text_value[:-1].strip()

    try:
        return float(text_value)
    except ValueError:
        return None


def build_segments_chart_frame(segments: list[dict[str, Any]]) -> pd.DataFrame:
    chart_rows = []
    for segment in segments:
        metric_value = parse_metric_value(segment.get("value"))
        if metric_value is None:
            continue

        segment_name = str(segment.get("segment_name") or "segment")
        metric = str(segment.get("metric") or "metric")
        chart_rows.append(
            {
                "segment": f"{segment_name} | {metric}",
                "value": metric_value,
            }
        )

    return pd.DataFrame(chart_rows)


def build_recommendation_priority_frame(recommendations: list[dict[str, Any]]) -> pd.DataFrame:
    priorities = [
        str(recommendation.get("priority") or "medium").lower()
        for recommendation in recommendations
        if isinstance(recommendation, dict)
    ]
    if not priorities:
        return pd.DataFrame(columns=["priority", "count"])

    priority_order = {"high": 3, "medium": 2, "low": 1}
    priority_frame = (
        pd.Series(priorities, name="priority")
        .value_counts()
        .rename_axis("priority")
        .reset_index(name="count")
    )
    priority_frame["sort_order"] = priority_frame["priority"].map(priority_order).fillna(0)
    return priority_frame.sort_values("sort_order", ascending=False).drop(columns="sort_order")


def render_ai_source(response: dict[str, Any]) -> None:
    ai_source = response.get("ai_source") or "local_fallback"
    source_label = {
        "openai": "OpenAI",
        "local_fallback": "Local fallback",
    }.get(str(ai_source), str(ai_source).replace("_", " ").title())

    st.caption(f"AI Source: {source_label}")
    if source_detail := response.get("ai_source_detail"):
        st.caption(str(source_detail))


def render_header() -> None:
    st.title("Customer Intelligence Platform")
    st.caption("Snowflake-powered customer, order, and product intelligence")


def render_project_overview() -> None:
    st.subheader("Project Overview")
    st.write(
        "A production-style customer intelligence platform that turns Instacart order data into "
        "analytics marts, machine learning predictions, business dashboards, and AI-generated insights."
    )

    objective_column, capabilities_column = st.columns(2)
    with objective_column:
        st.markdown("**Business Objective**")
        st.write(
            "Help retail teams understand reorder behavior, identify churn risk, prioritize retention "
            "actions, and surface product trends from customer purchase history."
        )
    with capabilities_column:
        st.markdown("**Key Capabilities**")
        st.write(
            "Batch ingestion, Snowflake warehouse modeling, dbt transformations, feature engineering, "
            "RandomForest scoring, FastAPI predictions, Streamlit analytics, and OpenAI copilot insights."
        )

    st.markdown("**Architecture**")
    architecture_steps = pd.DataFrame(
        [
            {"step": "1", "layer": "Data Source", "component": "Instacart CSV files"},
            {"step": "2", "layer": "Ingestion", "component": "Python, pandas, Snowflake connector"},
            {"step": "3", "layer": "Warehouse", "component": "Snowflake raw, staging, and mart tables"},
            {"step": "4", "layer": "Transformations", "component": "dbt staging, facts, dimensions, feature store"},
            {"step": "5", "layer": "Machine Learning", "component": "scikit-learn RandomForest model"},
            {"step": "6", "layer": "Serving", "component": "FastAPI deployed on Render"},
            {"step": "7", "layer": "Experience", "component": "Streamlit Cloud dashboard and AI copilot"},
        ]
    )
    st.dataframe(architecture_steps, use_container_width=True, hide_index=True)

    st.markdown("**Tech Stack**")
    stack_frame = pd.DataFrame(
        [
            {"area": "Backend", "tools": "Python, FastAPI, Pydantic"},
            {"area": "Data Warehouse", "tools": "Snowflake"},
            {"area": "Transformations", "tools": "dbt"},
            {"area": "Machine Learning", "tools": "pandas, scikit-learn, joblib"},
            {"area": "Dashboard", "tools": "Streamlit"},
            {"area": "AI", "tools": "OpenAI API with Snowflake context"},
            {"area": "Deployment", "tools": "Render, Streamlit Cloud, GitHub Actions"},
        ]
    )
    st.dataframe(stack_frame, use_container_width=True, hide_index=True)

    st.markdown("**Live Services**")
    dashboard_url, api_docs_url, health_url = st.columns(3)
    dashboard_url.link_button("Dashboard", STREAMLIT_APP_URL, use_container_width=True)
    api_docs_url.link_button("API Docs", f"{RENDER_API_URL}/docs", use_container_width=True)
    health_url.link_button("Health Check", f"{RENDER_API_URL}/health", use_container_width=True)

    st.markdown("**Portfolio Notes**")
    st.info(
        "Revenue is represented as a proxy because the Instacart dataset does not include prices. "
        "The AI copilot is grounded in aggregate Snowflake metrics and includes a local fallback mode "
        "for resilience when OpenAI is unavailable."
    )


def render_kpis(kpis: dict[str, Any]) -> None:
    st.subheader("KPIs")
    revenue_proxy, total_orders, reorder_rate = st.columns(3)

    revenue_proxy.metric("Revenue Proxy", format_number(kpis.get("revenue_proxy", 0)))
    total_orders.metric("Orders", format_number(kpis.get("total_orders", 0)))
    reorder_rate.metric("Reorder Rate", format_percent(kpis.get("reorder_rate", 0)))


def render_customer_insights(frame: pd.DataFrame) -> None:
    st.subheader("Customer Insights")
    if frame.empty:
        st.info("No customer feature data found. Run the dbt feature_store model first.")
        return

    chart_data = frame.head(15).set_index("user_id")[["total_orders", "unique_products"]]
    st.bar_chart(chart_data)

    st.dataframe(
        frame,
        use_container_width=True,
        hide_index=True,
        column_config={
            "user_id": "User ID",
            "total_orders": st.column_config.NumberColumn("Total Orders", format="%d"),
            "observed_basket_orders": st.column_config.NumberColumn("Observed Baskets", format="%d"),
            "avg_basket_size": st.column_config.NumberColumn("Avg Basket Size", format="%.2f"),
            "reorder_ratio": st.column_config.ProgressColumn("Reorder Ratio", format="%.2f", min_value=0, max_value=1),
            "unique_products": st.column_config.NumberColumn("Unique Products", format="%d"),
            "days_between_orders": st.column_config.NumberColumn("Days Between Orders", format="%.2f"),
        },
    )


def render_product_trends(frame: pd.DataFrame) -> None:
    st.subheader("Product Trends")
    if frame.empty:
        st.info("No product data found. Run the dim_products dbt model first.")
        return

    chart_data = frame.head(15).set_index("product_name")["order_line_count"]
    st.bar_chart(chart_data)

    st.dataframe(
        frame,
        use_container_width=True,
        hide_index=True,
        column_config={
            "product_name": "Product",
            "department": "Department",
            "aisle": "Aisle",
            "order_line_count": st.column_config.NumberColumn("Order Lines", format="%d"),
            "reordered_line_count": st.column_config.NumberColumn("Reordered Lines", format="%d"),
            "reorder_rate": st.column_config.ProgressColumn("Reorder Rate", format="%.2f", min_value=0, max_value=1),
        },
    )


def render_data_quality() -> None:
    st.subheader("Data Quality")
    st.caption("Live Snowflake checks for raw Instacart tables before downstream analytics.")

    try:
        quality_results = load_data_quality_results()
        raw_counts = load_raw_table_counts()
    except Exception as exc:
        logger.exception("Failed to load data quality results")
        st.error("Unable to load data quality results from Snowflake.")
        st.exception(exc)
        return

    if quality_results.empty:
        st.info("No data quality results found.")
        return

    total_checks = len(quality_results)
    passed_checks = int((quality_results["status"] == "Passed").sum())
    failed_checks = total_checks - passed_checks

    total_column, passed_column, failed_column = st.columns(3)
    total_column.metric("Total Checks", format_number(total_checks))
    passed_column.metric("Passed", format_number(passed_checks))
    failed_column.metric("Failed", format_number(failed_checks))

    status_summary = quality_results["status"].value_counts().rename_axis("status").reset_index(name="checks")
    st.bar_chart(status_summary.set_index("status")["checks"])

    st.markdown("**Validation Results**")
    st.dataframe(
        quality_results,
        use_container_width=True,
        hide_index=True,
        column_config={
            "table_name": "Table",
            "check_name": "Check",
            "invalid_count": st.column_config.NumberColumn("Invalid Rows", format="%d"),
            "status": "Status",
        },
    )

    if not raw_counts.empty:
        st.markdown("**Raw Table Row Counts**")
        st.bar_chart(raw_counts.set_index("table_name")["row_count"])
        st.dataframe(
            raw_counts,
            use_container_width=True,
            hide_index=True,
            column_config={
                "table_name": "Table",
                "row_count": st.column_config.NumberColumn("Rows", format="%d"),
            },
        )


def render_pipeline_health() -> None:
    st.subheader("Pipeline Health")
    st.caption("Operational checks across Snowflake, marts, model artifacts, and the deployed API.")

    snowflake_status = "Failed"
    snowflake_detail: dict[str, Any] = {}
    try:
        snowflake_detail = load_snowflake_connection_health()
        snowflake_status = str(snowflake_detail.get("status", "Connected"))
    except Exception as exc:
        logger.exception("Snowflake health check failed")
        snowflake_detail = {"error": str(exc)}

    api_status = "Failed"
    api_detail: dict[str, Any] = {}
    try:
        api_detail = load_api_health()
        api_status = "OK" if api_detail.get("status") == "ok" else "Failed"
    except Exception as exc:
        logger.exception("API health check failed")
        api_detail = {"error": str(exc)}

    model_exists = Path(settings.model_path).exists()
    metrics_exists = MODEL_METRICS_PATH.exists()

    snowflake_column, api_column, model_column, metrics_column = st.columns(4)
    snowflake_column.metric("Snowflake", snowflake_status)
    api_column.metric("Render API", api_status)
    model_column.metric("Model Artifact", "Available" if model_exists else "Missing")
    metrics_column.metric("Model Metrics", "Available" if metrics_exists else "Missing")

    health_rows = [
        {
            "component": "Snowflake connection",
            "status": snowflake_status,
            "detail": snowflake_detail.get("database_name") or snowflake_detail.get("error", ""),
        },
        {
            "component": "Render API",
            "status": api_status,
            "detail": format_api_health_detail(api_detail),
        },
        {
            "component": "Model artifact",
            "status": "Available" if model_exists else "Missing",
            "detail": settings.model_path,
        },
        {
            "component": "Training metrics",
            "status": "Available" if metrics_exists else "Missing",
            "detail": str(MODEL_METRICS_PATH.relative_to(PROJECT_ROOT)),
        },
    ]
    st.markdown("**Service Checks**")
    st.dataframe(pd.DataFrame(health_rows), use_container_width=True, hide_index=True)

    if snowflake_status == "Connected":
        st.markdown("**Snowflake Session**")
        session_frame = pd.DataFrame([snowflake_detail])
        st.dataframe(session_frame, use_container_width=True, hide_index=True)

    try:
        raw_counts = load_raw_table_counts()
        mart_counts = load_mart_table_counts()
    except Exception as exc:
        logger.exception("Failed to load pipeline table counts")
        st.error("Unable to load pipeline table counts from Snowflake.")
        st.exception(exc)
        return

    raw_column, mart_column = st.columns(2)
    with raw_column:
        st.markdown("**Raw Tables**")
        st.bar_chart(raw_counts.set_index("table_name")["row_count"])
        st.dataframe(
            raw_counts,
            use_container_width=True,
            hide_index=True,
            column_config={
                "table_name": "Table",
                "row_count": st.column_config.NumberColumn("Rows", format="%d"),
            },
        )

    with mart_column:
        st.markdown("**Mart Tables**")
        st.bar_chart(mart_counts.set_index("table_name")["row_count"])
        st.dataframe(
            mart_counts,
            use_container_width=True,
            hide_index=True,
            column_config={
                "table_name": "Table",
                "row_count": st.column_config.NumberColumn("Rows", format="%d"),
            },
        )


def render_model_performance(metrics: dict[str, Any], feature_importance: pd.DataFrame) -> None:
    st.subheader("Model Performance")
    if not metrics:
        st.info("No model metrics found. Run the ML training pipeline first.")
        return

    accuracy, precision, recall, f1_score, roc_auc = st.columns(5)
    accuracy.metric("Accuracy", format_percent(metrics.get("accuracy", 0)))
    precision.metric("Precision", format_percent(metrics.get("precision", 0)))
    recall.metric("Recall", format_percent(metrics.get("recall", 0)))
    f1_score.metric("F1 Score", format_percent(metrics.get("f1", 0)))
    roc_auc.metric("ROC AUC", f"{float(metrics.get('roc_auc') or 0):.3f}")

    train_rows, test_rows, positive_rate = st.columns(3)
    train_rows.metric("Train Rows", format_number(metrics.get("train_rows", 0)))
    test_rows.metric("Test Rows", format_number(metrics.get("test_rows", 0)))
    positive_rate.metric("Positive Rate", format_percent(metrics.get("positive_rate", 0)))

    st.divider()

    confusion_matrix = metrics.get("confusion_matrix", [])
    if confusion_matrix:
        confusion_frame = pd.DataFrame(
            confusion_matrix,
            index=["Actual No Reorder", "Actual Reorder"],
            columns=["Predicted No Reorder", "Predicted Reorder"],
        )
        st.subheader("Confusion Matrix")
        st.dataframe(confusion_frame, use_container_width=True)

    if not feature_importance.empty:
        st.subheader("Feature Importance")
        chart_data = feature_importance.set_index("feature")["importance"]
        st.bar_chart(chart_data)
        st.dataframe(
            feature_importance,
            use_container_width=True,
            hide_index=True,
            column_config={
                "feature": "Feature",
                "importance": st.column_config.ProgressColumn(
                    "Importance",
                    format="%.3f",
                    min_value=0,
                    max_value=1,
                ),
            },
        )


def render_copilot_response(response: dict[str, Any], message_index: int | None = None) -> None:
    render_ai_source(response)

    if summary := response.get("summary"):
        st.markdown(f"**Summary:** {summary}")
    if explanation := response.get("explanation"):
        st.write(explanation)

    impacted_segments = response.get("impacted_segments") or []
    if impacted_segments:
        st.markdown("**Impacted Segments**")
        segment_frame = pd.DataFrame(impacted_segments)
        chart_frame = build_segments_chart_frame(impacted_segments)
        if chart_frame.empty:
            st.dataframe(segment_frame, use_container_width=True, hide_index=True)
        else:
            table_column, chart_column = st.columns([1.35, 1])
            with table_column:
                st.dataframe(segment_frame, use_container_width=True, hide_index=True)
            with chart_column:
                st.caption("Metric comparison")
                st.bar_chart(chart_frame.set_index("segment")["value"])

    recommendations = response.get("recommendations") or []
    if recommendations:
        st.markdown("**Recommendations**")
        for recommendation in recommendations:
            priority = recommendation.get("priority", "medium")
            action = recommendation.get("action", "")
            expected_impact = recommendation.get("expected_impact", "")
            st.markdown(f"- **{priority.upper()}**: {action}  \n  {expected_impact}")

        priority_frame = build_recommendation_priority_frame(recommendations)
        if not priority_frame.empty:
            st.caption("Recommendation priority mix")
            st.bar_chart(priority_frame.set_index("priority")["count"])

    insights = response.get("insights") or []
    if insights:
        with st.expander("Detailed Insights"):
            for insight in insights:
                st.markdown(f"**{insight.get('title', 'Insight')}**")
                st.write(insight.get("finding", ""))
                evidence = insight.get("evidence") or []
                if evidence:
                    st.caption("Evidence: " + " | ".join(str(item) for item in evidence))

    follow_up_questions = response.get("follow_up_questions") or []
    if follow_up_questions:
        st.markdown("**Follow-up questions**")
        for index, follow_up_question in enumerate(follow_up_questions):
            key_parts = ["follow_up", str(message_index if message_index is not None else "live"), str(index)]
            button_key = "_".join(key_parts)
            if st.button(follow_up_question, key=button_key):
                queue_copilot_question(follow_up_question)


def render_ai_copilot() -> None:
    st.subheader("AI Business Copilot")
    st.caption("Ask business questions about churn, reorders, products, retention, and customer behavior.")

    if "copilot_messages" not in st.session_state:
        st.session_state.copilot_messages = [
            {
                "role": "assistant",
                "content": "Ask me about churn, reorder trends, top products, retention, or customer segments.",
            }
        ]

    examples = [
        "Why is churn increasing?",
        "What products drive repeat purchases?",
        "Which customer segments are at churn risk?",
        "How can we improve retention?",
    ]
    st.markdown("**Quick prompts**")
    prompt_columns = st.columns(2)
    for index, example in enumerate(examples):
        with prompt_columns[index % 2]:
            if st.button(example, key=f"example_prompt_{index}"):
                queue_copilot_question(example)

    for message_index, message in enumerate(st.session_state.copilot_messages):
        with st.chat_message(message["role"]):
            content = message.get("content")
            if isinstance(content, dict):
                render_copilot_response(content, message_index=message_index)
            else:
                st.write(content)

    question = st.chat_input("Ask for AI insights")
    pending_question = st.session_state.pop("pending_copilot_question", None)
    question = question or pending_question
    if not question:
        return

    st.session_state.copilot_messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Generating insights from Snowflake metrics..."):
            try:
                response = request_copilot_insights(question)
            except requests.RequestException as exc:
                logger.exception("Copilot API request failed")
                error_message = (
                    "Unable to reach the copilot API. Check API_BASE_URL and the Render service status."
                )
                st.error(error_message)
                st.caption(str(exc))
                st.session_state.copilot_messages.append({"role": "assistant", "content": error_message})
                return

        st.session_state.copilot_messages.append({"role": "assistant", "content": response})
        render_copilot_response(response, message_index=len(st.session_state.copilot_messages) - 1)


def render_sidebar() -> None:
    st.sidebar.header("Controls")
    st.sidebar.write(f"Environment: `{settings.app_env}`")
    st.sidebar.write(f"Database: `{settings.snowflake_database or 'not configured'}`")
    st.sidebar.write(f"Schema: `{settings.snowflake_schema or 'not configured'}`")
    st.sidebar.write(f"API: `{settings.api_base_url}`")
    if st.sidebar.button("Refresh data"):
        st.cache_data.clear()
        st.rerun()


def main() -> None:
    render_sidebar()
    render_header()

    try:
        kpis = load_kpis()
        customers = load_customer_insights()
        products = load_product_trends()
        model_metrics = load_model_metrics()
        feature_importance = load_feature_importance()
    except Exception as exc:
        logger.exception("Failed to load dashboard data")
        st.error("Unable to load dashboard data from Snowflake.")
        st.exception(exc)
        return

    render_kpis(kpis)
    st.divider()

    overview_tab, customer_tab, product_tab, quality_tab, health_tab, model_tab, copilot_tab = st.tabs(
        [
            "Project Overview",
            "Customer Insights",
            "Product Trends",
            "Data Quality",
            "Pipeline Health",
            "Model Performance",
            "AI Copilot",
        ]
    )
    with overview_tab:
        render_project_overview()
    with customer_tab:
        render_customer_insights(customers)
    with product_tab:
        render_product_trends(products)
    with quality_tab:
        render_data_quality()
    with health_tab:
        render_pipeline_health()
    with model_tab:
        render_model_performance(model_metrics, feature_importance)
    with copilot_tab:
        render_ai_copilot()


if __name__ == "__main__":
    main()
