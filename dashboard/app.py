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

TRANSLATIONS = {
    "en": {
        "toggle_language": "Afficher en français",
        "title": "Customer Intelligence Platform",
        "subtitle": "Snowflake-powered customer, order, and product intelligence",
        "controls": "Controls",
        "language": "Language",
        "environment": "Environment",
        "database": "Database",
        "schema": "Schema",
        "api": "API",
        "refresh_data": "Refresh data",
        "kpis": "KPIs",
        "revenue_proxy": "Revenue Proxy",
        "orders": "Orders",
        "reorder_rate": "Reorder Rate",
        "project_overview": "Project Overview",
        "overview_intro": (
            "A production-style customer intelligence platform that turns Instacart order data into "
            "analytics marts, machine learning predictions, business dashboards, and AI-generated insights."
        ),
        "business_objective": "Business Objective",
        "business_objective_text": (
            "Help retail teams understand reorder behavior, identify churn risk, prioritize retention "
            "actions, and surface product trends from customer purchase history."
        ),
        "key_capabilities": "Key Capabilities",
        "key_capabilities_text": (
            "Batch ingestion, Snowflake warehouse modeling, dbt transformations, feature engineering, "
            "RandomForest scoring, FastAPI predictions, Streamlit analytics, and OpenAI copilot insights."
        ),
        "architecture": "Architecture",
        "tech_stack": "Tech Stack",
        "live_services": "Live Services",
        "dashboard": "Dashboard",
        "api_docs": "API Docs",
        "health_check": "Health Check",
        "portfolio_notes": "Portfolio Notes",
        "portfolio_notes_text": (
            "Revenue is represented as a proxy because the Instacart dataset does not include prices. "
            "The AI copilot is grounded in aggregate Snowflake metrics and includes a local fallback mode "
            "for resilience when OpenAI is unavailable."
        ),
        "customer_insights": "Customer Insights",
        "no_customer_data": "No customer feature data found. Run the dbt feature_store model first.",
        "product_trends": "Product Trends",
        "no_product_data": "No product data found. Run the dim_products dbt model first.",
        "data_quality": "Data Quality",
        "data_quality_caption": "Live Snowflake checks for raw Instacart tables before downstream analytics.",
        "data_quality_error": "Unable to load data quality results from Snowflake.",
        "no_data_quality": "No data quality results found.",
        "total_checks": "Total Checks",
        "passed": "Passed",
        "failed": "Failed",
        "validation_results": "Validation Results",
        "raw_table_row_counts": "Raw Table Row Counts",
        "pipeline_health": "Pipeline Health",
        "pipeline_health_caption": "Operational checks across Snowflake, marts, model artifacts, and the deployed API.",
        "snowflake": "Snowflake",
        "render_api": "Render API",
        "model_artifact": "Model Artifact",
        "model_metrics": "Model Metrics",
        "available": "Available",
        "missing": "Missing",
        "service_checks": "Service Checks",
        "snowflake_session": "Snowflake Session",
        "raw_tables": "Raw Tables",
        "mart_tables": "Mart Tables",
        "table_counts_error": "Unable to load pipeline table counts from Snowflake.",
        "model_performance": "Model Performance",
        "no_model_metrics": "No model metrics found. Run the ML training pipeline first.",
        "accuracy": "Accuracy",
        "precision": "Precision",
        "recall": "Recall",
        "f1_score": "F1 Score",
        "roc_auc": "ROC AUC",
        "train_rows": "Train Rows",
        "test_rows": "Test Rows",
        "positive_rate": "Positive Rate",
        "confusion_matrix": "Confusion Matrix",
        "feature_importance": "Feature Importance",
        "ai_copilot": "AI Copilot",
        "ai_business_copilot": "AI Business Copilot",
        "ai_copilot_caption": "Ask business questions about churn, reorders, products, retention, and customer behavior.",
        "quick_prompts": "Quick prompts",
        "ask_ai": "Ask for AI insights",
        "generating_insights": "Generating insights from Snowflake metrics...",
        "copilot_api_error": "Unable to reach the copilot API. Check API_BASE_URL and the Render service status.",
        "ai_source": "AI Source",
        "summary": "Summary",
        "impacted_segments": "Impacted Segments",
        "metric_comparison": "Metric comparison",
        "recommendations": "Recommendations",
        "priority_mix": "Recommendation priority mix",
        "detailed_insights": "Detailed Insights",
        "evidence": "Evidence",
        "follow_up_questions": "Follow-up questions",
    },
    "fr": {
        "toggle_language": "Show in English",
        "title": "Plateforme d'intelligence client",
        "subtitle": "Analyse clients, commandes et produits propulsée par Snowflake",
        "controls": "Contrôles",
        "language": "Langue",
        "environment": "Environnement",
        "database": "Base de données",
        "schema": "Schéma",
        "api": "API",
        "refresh_data": "Rafraîchir les données",
        "kpis": "Indicateurs",
        "revenue_proxy": "Proxy De Revenus",
        "orders": "Commandes",
        "reorder_rate": "Taux de recommande",
        "project_overview": "Vue d'ensemble",
        "overview_intro": (
            "Une plateforme de type production qui transforme les données de commandes Instacart en "
            "marts analytiques, prédictions machine learning, tableaux de bord et insights générés par IA."
        ),
        "business_objective": "Objectif Business",
        "business_objective_text": (
            "Aider les equipes retail a comprendre les comportements de recommande, identifier le risque "
            "de churn, prioriser la retention et faire ressortir les tendances produits."
        ),
        "key_capabilities": "Capacités clés",
        "key_capabilities_text": (
            "Ingestion batch, modelisation Snowflake, transformations dbt, feature engineering, scoring "
            "RandomForest, predictions FastAPI, analyses Streamlit et insights OpenAI."
        ),
        "architecture": "Architecture",
        "tech_stack": "Stack technique",
        "live_services": "Services Live",
        "dashboard": "Dashboard",
        "api_docs": "Docs API",
        "health_check": "Health Check",
        "portfolio_notes": "Notes Portfolio",
        "portfolio_notes_text": (
            "Les revenus sont représentés par un proxy, car le dataset Instacart ne contient pas les prix. "
            "Le copilot IA est ancré dans les métriques agrégées Snowflake et inclut un mode fallback local "
            "si OpenAI n'est pas disponible."
        ),
        "customer_insights": "Insights Clients",
        "no_customer_data": "Aucune donnee client trouvee. Lance d'abord le modele dbt feature_store.",
        "product_trends": "Tendances Produits",
        "no_product_data": "Aucune donnee produit trouvee. Lance d'abord le modele dbt dim_products.",
        "data_quality": "Qualité des données",
        "data_quality_caption": "Validations Snowflake live sur les tables brutes avant les analyses downstream.",
        "data_quality_error": "Impossible de charger les resultats de qualite depuis Snowflake.",
        "no_data_quality": "Aucun resultat de qualite trouve.",
        "total_checks": "Checks Totaux",
        "passed": "Reussis",
        "failed": "Echoues",
        "validation_results": "Resultats De Validation",
        "raw_table_row_counts": "Volumes Des Tables Raw",
        "pipeline_health": "Santé du pipeline",
        "pipeline_health_caption": "Checks operationnels sur Snowflake, les marts, les artefacts modele et l'API deployee.",
        "snowflake": "Snowflake",
        "render_api": "API Render",
        "model_artifact": "Artefact Modele",
        "model_metrics": "Metriques Modele",
        "available": "Disponible",
        "missing": "Manquant",
        "service_checks": "Checks Des Services",
        "snowflake_session": "Session Snowflake",
        "raw_tables": "Tables Raw",
        "mart_tables": "Tables Marts",
        "table_counts_error": "Impossible de charger les volumes des tables depuis Snowflake.",
        "model_performance": "Performance modèle",
        "no_model_metrics": "Aucune metrique modele trouvee. Lance d'abord le pipeline ML.",
        "accuracy": "Accuracy",
        "precision": "Precision",
        "recall": "Recall",
        "f1_score": "F1 Score",
        "roc_auc": "ROC AUC",
        "train_rows": "Lignes Train",
        "test_rows": "Lignes Test",
        "positive_rate": "Taux Positif",
        "confusion_matrix": "Matrice De Confusion",
        "feature_importance": "Importance des features",
        "ai_copilot": "Copilot IA",
        "ai_business_copilot": "Copilot Business IA",
        "ai_copilot_caption": "Pose des questions business sur le churn, les recommandes, les produits, la retention et les clients.",
        "quick_prompts": "Prompts Rapides",
        "ask_ai": "Demander des insights IA",
        "generating_insights": "Generation d'insights depuis les metriques Snowflake...",
        "copilot_api_error": "Impossible de joindre l'API copilot. Verifie API_BASE_URL et le statut Render.",
        "ai_source": "Source IA",
        "summary": "Résumé",
        "impacted_segments": "Segments impactés",
        "metric_comparison": "Comparaison des métriques",
        "recommendations": "Recommandations",
        "priority_mix": "Mix Des Priorites",
        "detailed_insights": "Insights détaillés",
        "evidence": "Evidence",
        "follow_up_questions": "Questions de suivi",
    },
}


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


def get_language() -> str:
    return str(st.session_state.get("language", "en"))


def translate(key: str) -> str:
    language = get_language()
    return TRANSLATIONS.get(language, TRANSLATIONS["en"]).get(key, TRANSLATIONS["en"].get(key, key))


def toggle_language() -> None:
    st.session_state.language = "fr" if get_language() == "en" else "en"
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

    st.caption(f"{translate('ai_source')}: {source_label}")
    if source_detail := response.get("ai_source_detail"):
        st.caption(str(source_detail))


def render_header() -> None:
    st.title(translate("title"))
    st.caption(translate("subtitle"))


def render_project_overview() -> None:
    st.subheader(translate("project_overview"))
    st.write(translate("overview_intro"))

    objective_column, capabilities_column = st.columns(2)
    with objective_column:
        st.markdown(f"**{translate('business_objective')}**")
        st.write(translate("business_objective_text"))
    with capabilities_column:
        st.markdown(f"**{translate('key_capabilities')}**")
        st.write(translate("key_capabilities_text"))

    st.markdown(f"**{translate('architecture')}**")
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

    st.markdown(f"**{translate('tech_stack')}**")
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

    st.markdown(f"**{translate('live_services')}**")
    dashboard_url, api_docs_url, health_url = st.columns(3)
    dashboard_url.link_button(translate("dashboard"), STREAMLIT_APP_URL, use_container_width=True)
    api_docs_url.link_button(translate("api_docs"), f"{RENDER_API_URL}/docs", use_container_width=True)
    health_url.link_button(translate("health_check"), f"{RENDER_API_URL}/health", use_container_width=True)

    st.markdown(f"**{translate('portfolio_notes')}**")
    st.info(translate("portfolio_notes_text"))


def render_kpis(kpis: dict[str, Any]) -> None:
    st.subheader(translate("kpis"))
    revenue_proxy, total_orders, reorder_rate = st.columns(3)

    revenue_proxy.metric(translate("revenue_proxy"), format_number(kpis.get("revenue_proxy", 0)))
    total_orders.metric(translate("orders"), format_number(kpis.get("total_orders", 0)))
    reorder_rate.metric(translate("reorder_rate"), format_percent(kpis.get("reorder_rate", 0)))


def render_customer_insights(frame: pd.DataFrame) -> None:
    st.subheader(translate("customer_insights"))
    if frame.empty:
        st.info(translate("no_customer_data"))
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
    st.subheader(translate("product_trends"))
    if frame.empty:
        st.info(translate("no_product_data"))
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
    st.subheader(translate("data_quality"))
    st.caption(translate("data_quality_caption"))

    try:
        quality_results = load_data_quality_results()
        raw_counts = load_raw_table_counts()
    except Exception as exc:
        logger.exception("Failed to load data quality results")
        st.error(translate("data_quality_error"))
        st.exception(exc)
        return

    if quality_results.empty:
        st.info(translate("no_data_quality"))
        return

    total_checks = len(quality_results)
    passed_checks = int((quality_results["status"] == "Passed").sum())
    failed_checks = total_checks - passed_checks

    total_column, passed_column, failed_column = st.columns(3)
    total_column.metric(translate("total_checks"), format_number(total_checks))
    passed_column.metric(translate("passed"), format_number(passed_checks))
    failed_column.metric(translate("failed"), format_number(failed_checks))

    status_summary = quality_results["status"].value_counts().rename_axis("status").reset_index(name="checks")
    st.bar_chart(status_summary.set_index("status")["checks"])

    st.markdown(f"**{translate('validation_results')}**")
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
        st.markdown(f"**{translate('raw_table_row_counts')}**")
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
    st.subheader(translate("pipeline_health"))
    st.caption(translate("pipeline_health_caption"))

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
    snowflake_column.metric(translate("snowflake"), snowflake_status)
    api_column.metric(translate("render_api"), api_status)
    model_column.metric(translate("model_artifact"), translate("available") if model_exists else translate("missing"))
    metrics_column.metric(translate("model_metrics"), translate("available") if metrics_exists else translate("missing"))

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
            "status": translate("available") if model_exists else translate("missing"),
            "detail": settings.model_path,
        },
        {
            "component": "Training metrics",
            "status": translate("available") if metrics_exists else translate("missing"),
            "detail": str(MODEL_METRICS_PATH.relative_to(PROJECT_ROOT)),
        },
    ]
    st.markdown(f"**{translate('service_checks')}**")
    st.dataframe(pd.DataFrame(health_rows), use_container_width=True, hide_index=True)

    if snowflake_status == "Connected":
        st.markdown(f"**{translate('snowflake_session')}**")
        session_frame = pd.DataFrame([snowflake_detail])
        st.dataframe(session_frame, use_container_width=True, hide_index=True)

    try:
        raw_counts = load_raw_table_counts()
        mart_counts = load_mart_table_counts()
    except Exception as exc:
        logger.exception("Failed to load pipeline table counts")
        st.error(translate("table_counts_error"))
        st.exception(exc)
        return

    raw_column, mart_column = st.columns(2)
    with raw_column:
        st.markdown(f"**{translate('raw_tables')}**")
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
        st.markdown(f"**{translate('mart_tables')}**")
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
    st.subheader(translate("model_performance"))
    if not metrics:
        st.info(translate("no_model_metrics"))
        return

    accuracy, precision, recall, f1_score, roc_auc = st.columns(5)
    accuracy.metric(translate("accuracy"), format_percent(metrics.get("accuracy", 0)))
    precision.metric(translate("precision"), format_percent(metrics.get("precision", 0)))
    recall.metric(translate("recall"), format_percent(metrics.get("recall", 0)))
    f1_score.metric(translate("f1_score"), format_percent(metrics.get("f1", 0)))
    roc_auc.metric(translate("roc_auc"), f"{float(metrics.get('roc_auc') or 0):.3f}")

    train_rows, test_rows, positive_rate = st.columns(3)
    train_rows.metric(translate("train_rows"), format_number(metrics.get("train_rows", 0)))
    test_rows.metric(translate("test_rows"), format_number(metrics.get("test_rows", 0)))
    positive_rate.metric(translate("positive_rate"), format_percent(metrics.get("positive_rate", 0)))

    st.divider()

    confusion_matrix = metrics.get("confusion_matrix", [])
    if confusion_matrix:
        confusion_frame = pd.DataFrame(
            confusion_matrix,
            index=["Actual No Reorder", "Actual Reorder"],
            columns=["Predicted No Reorder", "Predicted Reorder"],
        )
        st.subheader(translate("confusion_matrix"))
        st.dataframe(confusion_frame, use_container_width=True)

    if not feature_importance.empty:
        st.subheader(translate("feature_importance"))
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
        st.markdown(f"**{translate('summary')}:** {summary}")
    if explanation := response.get("explanation"):
        st.write(explanation)

    impacted_segments = response.get("impacted_segments") or []
    if impacted_segments:
        st.markdown(f"**{translate('impacted_segments')}**")
        segment_frame = pd.DataFrame(impacted_segments)
        chart_frame = build_segments_chart_frame(impacted_segments)
        if chart_frame.empty:
            st.dataframe(segment_frame, use_container_width=True, hide_index=True)
        else:
            table_column, chart_column = st.columns([1.35, 1])
            with table_column:
                st.dataframe(segment_frame, use_container_width=True, hide_index=True)
            with chart_column:
                st.caption(translate("metric_comparison"))
                st.bar_chart(chart_frame.set_index("segment")["value"])

    recommendations = response.get("recommendations") or []
    if recommendations:
        st.markdown(f"**{translate('recommendations')}**")
        for recommendation in recommendations:
            priority = recommendation.get("priority", "medium")
            action = recommendation.get("action", "")
            expected_impact = recommendation.get("expected_impact", "")
            st.markdown(f"- **{priority.upper()}**: {action}  \n  {expected_impact}")

        priority_frame = build_recommendation_priority_frame(recommendations)
        if not priority_frame.empty:
            st.caption(translate("priority_mix"))
            st.bar_chart(priority_frame.set_index("priority")["count"])

    insights = response.get("insights") or []
    if insights:
        with st.expander(translate("detailed_insights")):
            for insight in insights:
                st.markdown(f"**{insight.get('title', 'Insight')}**")
                st.write(insight.get("finding", ""))
                evidence = insight.get("evidence") or []
                if evidence:
                    st.caption(f"{translate('evidence')}: " + " | ".join(str(item) for item in evidence))

    follow_up_questions = response.get("follow_up_questions") or []
    if follow_up_questions:
        st.markdown(f"**{translate('follow_up_questions')}**")
        for index, follow_up_question in enumerate(follow_up_questions):
            key_parts = ["follow_up", str(message_index if message_index is not None else "live"), str(index)]
            button_key = "_".join(key_parts)
            if st.button(follow_up_question, key=button_key):
                queue_copilot_question(follow_up_question)


def render_ai_copilot() -> None:
    st.subheader(translate("ai_business_copilot"))
    st.caption(translate("ai_copilot_caption"))

    if "copilot_messages" not in st.session_state:
        st.session_state.copilot_messages = [
            {
                "role": "assistant",
                "content": translate("ai_copilot_caption"),
            }
        ]

    examples = [
        "Why is churn increasing?",
        "What products drive repeat purchases?",
        "Which customer segments are at churn risk?",
        "How can we improve retention?",
    ]
    st.markdown(f"**{translate('quick_prompts')}**")
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

    question = st.chat_input(translate("ask_ai"))
    pending_question = st.session_state.pop("pending_copilot_question", None)
    question = question or pending_question
    if not question:
        return

    st.session_state.copilot_messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner(translate("generating_insights")):
            try:
                response = request_copilot_insights(question)
            except requests.RequestException as exc:
                logger.exception("Copilot API request failed")
                error_message = translate("copilot_api_error")
                st.error(error_message)
                st.caption(str(exc))
                st.session_state.copilot_messages.append({"role": "assistant", "content": error_message})
                return

        st.session_state.copilot_messages.append({"role": "assistant", "content": response})
        render_copilot_response(response, message_index=len(st.session_state.copilot_messages) - 1)


def render_sidebar() -> None:
    st.sidebar.header(translate("controls"))
    st.sidebar.write(f"{translate('language')}: `{'Français' if get_language() == 'fr' else 'English'}`")
    if st.sidebar.button(translate("toggle_language")):
        toggle_language()
    st.sidebar.write(f"{translate('environment')}: `{settings.app_env}`")
    st.sidebar.write(f"{translate('database')}: `{settings.snowflake_database or 'not configured'}`")
    st.sidebar.write(f"{translate('schema')}: `{settings.snowflake_schema or 'not configured'}`")
    st.sidebar.write(f"{translate('api')}: `{settings.api_base_url}`")
    if st.sidebar.button(translate("refresh_data")):
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
            translate("project_overview"),
            translate("customer_insights"),
            translate("product_trends"),
            translate("data_quality"),
            translate("pipeline_health"),
            translate("model_performance"),
            translate("ai_copilot"),
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
