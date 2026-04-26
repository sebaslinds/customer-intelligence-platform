import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from sqlalchemy import text

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.logging_config import configure_logging
from config.production import validate_production_settings
from config.settings import get_settings
from ingestion.snowflake_client import build_snowflake_engine

settings = get_settings()
validate_production_settings(settings)
configure_logging(settings.log_level)
logger = logging.getLogger(__name__)


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
            avg_basket_size,
            reorder_ratio,
            unique_products,
            days_between_orders
        from feature_store
        order by total_orders desc, reorder_ratio desc
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
            "avg_basket_size": st.column_config.NumberColumn("Avg Basket Size", format="%.2f"),
            "reorder_ratio": st.column_config.ProgressColumn("Reorder Ratio", format="%.1f", min_value=0, max_value=1),
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
            "reorder_rate": st.column_config.ProgressColumn("Reorder Rate", format="%.1f", min_value=0, max_value=1),
        },
    )


def render_sidebar() -> None:
    st.sidebar.header("Controls")
    st.sidebar.write(f"Environment: `{settings.app_env}`")
    st.sidebar.write(f"Database: `{settings.snowflake_database or 'not configured'}`")
    st.sidebar.write(f"Schema: `{settings.snowflake_schema or 'not configured'}`")
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
    except Exception as exc:
        logger.exception("Failed to load dashboard data")
        st.error("Unable to load dashboard data from Snowflake.")
        st.exception(exc)
        return

    render_kpis(kpis)
    st.divider()

    customer_tab, product_tab = st.tabs(["Customer Insights", "Product Trends"])
    with customer_tab:
        render_customer_insights(customers)
    with product_tab:
        render_product_trends(products)


if __name__ == "__main__":
    main()
