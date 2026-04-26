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


def render_header() -> None:
    st.title("Customer Intelligence Platform")
    st.caption("Snowflake-powered customer, order, and product intelligence")


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
    if summary := response.get("summary"):
        st.markdown(f"**Summary:** {summary}")
    if explanation := response.get("explanation"):
        st.write(explanation)

    impacted_segments = response.get("impacted_segments") or []
    if impacted_segments:
        st.markdown("**Impacted Segments**")
        st.dataframe(pd.DataFrame(impacted_segments), use_container_width=True, hide_index=True)

    recommendations = response.get("recommendations") or []
    if recommendations:
        st.markdown("**Recommendations**")
        for recommendation in recommendations:
            priority = recommendation.get("priority", "medium")
            action = recommendation.get("action", "")
            expected_impact = recommendation.get("expected_impact", "")
            st.markdown(f"- **{priority.upper()}**: {action}  \n  {expected_impact}")

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

    customer_tab, product_tab, model_tab, copilot_tab = st.tabs(
        ["Customer Insights", "Product Trends", "Model Performance", "AI Copilot"]
    )
    with customer_tab:
        render_customer_insights(customers)
    with product_tab:
        render_product_trends(products)
    with model_tab:
        render_model_performance(model_metrics, feature_importance)
    with copilot_tab:
        render_ai_copilot()


if __name__ == "__main__":
    main()
