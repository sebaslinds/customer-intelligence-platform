import json
import logging
import html
import sys
from pathlib import Path
from typing import Any

import altair as alt
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
DECISION_TIMEOUT_SECONDS = 45
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
        "revenue_proxy_help": (
            "Estimated business volume based on total items ordered. The Instacart dataset does not include "
            "prices, so this is a directional revenue proxy, not actual sales."
        ),
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
        "model_explanation": "What The Model Predicts",
        "model_explanation_text": (
            "This RandomForest estimates whether a future order is likely to include reordered products. "
            "Because repeat purchases dominate the dataset, balanced metrics and threshold analysis matter "
            "more than accuracy alone."
        ),
        "metric_gap_warning": "Why Accuracy Looks Better Than Balanced Accuracy",
        "metric_gap_warning_text": (
            "The test set is heavily imbalanced: most examples are future reorders. A model can look accurate "
            "by predicting reorder too often, while still missing the no-reorder class. Balanced accuracy gives "
            "equal weight to both classes, so it is the better quality signal here."
        ),
        "balanced_accuracy": "Balanced Accuracy",
        "average_precision": "Average Precision",
        "brier_score": "Brier Score",
        "positive_prediction_rate": "Predicted Positive Rate",
        "recommended_threshold": "Recommended Threshold",
        "threshold_selection_metric": "Threshold Selection",
        "threshold_analysis": "Threshold Analysis",
        "roc_curve": "ROC Curve",
        "precision_recall_curve": "Precision-Recall Curve",
        "classification_report": "Classification Report",
        "prediction_errors": "Prediction Errors",
        "true_negatives": "True Negatives",
        "false_positives": "False Positives",
        "false_negatives": "False Negatives",
        "true_positives": "True Positives",
        "model_interpretation": "Model Interpretation",
        "feature_importance_help": (
            "Feature importance shows which inputs most influenced the RandomForest. If only one or two "
            "features dominate, the next improvement is to add richer customer and product behavior signals."
        ),
        "model_recommendations": "Model Recommendations",
        "no_curve_data": "Run the ML training pipeline again to generate curve and threshold data.",
        "model_card": "Model Card",
        "model_card_task": "Prediction task",
        "model_card_task_value": "Binary reorder classification",
        "model_card_target": "Target",
        "model_card_target_value": "will_reorder: whether a future order contains at least one reordered item",
        "model_card_model": "Model",
        "model_card_model_value": "Best model selected from RandomForest, LogisticRegression, and GradientBoosting",
        "model_card_use_case": "Business use case",
        "model_card_use_case_value": "Support retention campaigns, reorder nudges, and customer prioritization.",
        "model_card_features": "Main features",
        "model_card_features_value": "Order count, basket size, reorder ratio, product breadth, and order interval.",
        "model_card_limitations": "Limitations",
        "model_card_limitations_value": (
            "Instacart does not include prices, margins, demographics, or real campaign outcomes. "
            "Use the score as decision support, not as a fully automated targeting rule."
        ),
        "model_card_monitoring": "Monitoring guidance",
        "model_card_monitoring_value": (
            "Track recall, precision, balanced accuracy, positive rate, and threshold behavior after each retrain."
        ),
        "selected_model": "Selected Model",
        "model_comparison": "Model Comparison",
        "model_comparison_help": (
            "Models are compared on the same train/test split. ROC AUC is used for selection, then F1, "
            "precision, recall, and calibration are reviewed for business tradeoffs."
        ),
        "ml_glossary": "ML Metrics Glossary",
        "ml_glossary_caption": "Plain-English definitions for the model metrics shown in this tab.",
        "glossary_term": "Term",
        "glossary_definition": "Definition",
        "glossary_business_use": "How to use it",
        "ai_copilot": "AI Copilot",
        "ai_business_copilot": "AI Business Copilot",
        "ai_copilot_caption": "Ask business questions about churn, reorders, products, retention, and customer behavior.",
        "quick_prompts": "Quick prompts",
        "copilot_welcome": "Ask business questions about churn, reorders, products, retention, and customer behavior.",
        "prompt_churn": "Why is churn increasing?",
        "prompt_products": "What products drive repeat purchases?",
        "prompt_segments": "Which customer segments are at churn risk?",
        "prompt_retention": "How can we improve retention?",
        "ask_ai": "Ask for AI insights",
        "generating_insights": "Generating insights from Snowflake metrics...",
        "copilot_api_error": "Unable to reach the copilot API. Check API_BASE_URL and the Render service status.",
        "ai_source": "AI Source",
        "summary": "Summary",
        "impacted_segments": "Impacted Segments",
        "metric_comparison": "Metric comparison",
        "segment_snapshot": "Segment snapshot",
        "why_it_matters": "Why it matters",
        "supporting_table": "Supporting table",
        "recommendations": "Recommendations",
        "priority_mix": "Recommendation priority mix",
        "priority_mix_help": (
            "This chart counts recommendations by urgency. High means act first, medium means plan next, "
            "and low means monitor or improve later."
        ),
        "priority_level": "Priority level",
        "recommendation_count": "Recommendations",
        "priority_high_meaning": "Immediate action: highest expected business impact or risk reduction.",
        "priority_medium_meaning": "Near-term action: useful improvement after the urgent items are handled.",
        "priority_low_meaning": "Monitor or backlog: lower urgency but still useful for optimization.",
        "detailed_insights": "Detailed Insights",
        "evidence": "Evidence",
        "follow_up_questions": "Follow-up questions",
        "decision_engine": "Decision Engine",
        "decision_engine_caption": "Turn anomaly signals into prioritized alerts, recommendations, and AI explanations.",
        "scenario_inputs": "Scenario Inputs",
        "use_gemini": "Use Gemini explanation",
        "run_decision_engine": "Run Decision Engine",
        "decision_api_error": "Unable to reach the Decision Engine API. Check API_BASE_URL and Render service status.",
        "decision_explanation": "Decision Explanation",
        "explanation_source": "Explanation source",
        "gemini_status": "Gemini status",
        "gemini_requested": "Requested",
        "gemini_configured": "Configured",
        "gemini_yes": "Yes",
        "gemini_no": "No",
        "gemini_not_configured_help": (
            "Gemini was requested, but the Render API does not have GEMINI_API_KEY configured. "
            "Add it to the API service environment variables, then redeploy the API."
        ),
        "detected_anomalies": "Detected Anomalies",
        "no_anomalies": "No anomalies detected for this scenario.",
        "priority_alerts": "Priority Alerts",
        "no_alerts": "No priority alerts returned.",
        "decision_recommendations": "Decision Recommendations",
        "no_recommendations": "No recommendations returned.",
        "decision_type": "Type",
        "decision_title": "Decision",
        "decision_action": "Action",
        "decision_rationale": "Rationale",
        "scenario_reorder_rate": "Reorder Rate",
        "scenario_churn_rate": "Churn Rate",
        "scenario_days_between_orders": "Days Between Orders",
        "scenario_data_quality_failures": "Data Quality Failures",
        "scenario_api_health": "API Health",
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
        "revenue_proxy_help": (
            "Estimation du volume business basee sur le nombre total d'articles commandes. Le dataset "
            "Instacart ne contient pas les prix, donc ce KPI est un proxy directionnel, pas des ventes reelles."
        ),
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
        "model_explanation": "Ce que predit le modele",
        "model_explanation_text": (
            "Ce RandomForest estime si une future commande contient probablement des produits recommandes. "
            "Comme les rachats dominent le dataset, les metriques equilibrees et l'analyse des seuils sont "
            "plus importantes que l'accuracy seule."
        ),
        "metric_gap_warning": "Pourquoi l'accuracy semble meilleure que l'accuracy equilibree",
        "metric_gap_warning_text": (
            "Le test set est fortement desequilibre: la plupart des exemples sont des recommandes futures. "
            "Un modele peut sembler accurate en predisant trop souvent recommande, tout en detectant mal la "
            "classe no-reorder. L'accuracy equilibree donne le meme poids aux deux classes."
        ),
        "balanced_accuracy": "Accuracy equilibree",
        "average_precision": "Precision moyenne",
        "brier_score": "Score de Brier",
        "positive_prediction_rate": "Taux predit positif",
        "recommended_threshold": "Seuil recommande",
        "threshold_selection_metric": "Selection du seuil",
        "threshold_analysis": "Analyse des seuils",
        "roc_curve": "Courbe ROC",
        "precision_recall_curve": "Courbe precision-recall",
        "classification_report": "Rapport de classification",
        "prediction_errors": "Erreurs de prediction",
        "true_negatives": "Vrais negatifs",
        "false_positives": "Faux positifs",
        "false_negatives": "Faux negatifs",
        "true_positives": "Vrais positifs",
        "model_interpretation": "Interpretation du modele",
        "feature_importance_help": (
            "L'importance des features montre quelles variables influencent le plus le RandomForest. "
            "Si une ou deux features dominent, la prochaine amelioration est d'ajouter des signaux clients "
            "et produits plus riches."
        ),
        "model_recommendations": "Recommandations modele",
        "no_curve_data": "Relance le pipeline ML pour generer les courbes et l'analyse des seuils.",
        "model_card": "Fiche modele",
        "model_card_task": "Tache de prediction",
        "model_card_task_value": "Classification binaire de recommande",
        "model_card_target": "Target",
        "model_card_target_value": "will_reorder: indique si une future commande contient au moins un produit recommande",
        "model_card_model": "Modele",
        "model_card_model_value": "Meilleur modele choisi entre RandomForest, LogisticRegression et GradientBoosting",
        "model_card_use_case": "Usage business",
        "model_card_use_case_value": "Aider les campagnes de retention, les nudges de recommande et la priorisation client.",
        "model_card_features": "Features principales",
        "model_card_features_value": "Nombre de commandes, taille du panier, ratio de recommande, diversite produits et delai entre commandes.",
        "model_card_limitations": "Limites",
        "model_card_limitations_value": (
            "Instacart ne contient pas les prix, les marges, la demographie ni les resultats reels de campagnes. "
            "Le score doit servir d'aide a la decision, pas de regle de ciblage entierement automatisee."
        ),
        "model_card_monitoring": "Suivi recommande",
        "model_card_monitoring_value": (
            "Surveiller recall, precision, accuracy equilibree, taux positif et comportement des seuils apres chaque retraining."
        ),
        "selected_model": "Modele selectionne",
        "model_comparison": "Comparaison des modeles",
        "model_comparison_help": (
            "Les modeles sont compares sur le meme split train/test. ROC AUC sert a choisir le modele, puis F1, "
            "precision, recall et calibration sont analyses pour les compromis business."
        ),
        "ml_glossary": "Glossaire des metriques ML",
        "ml_glossary_caption": "Definitions simples des metriques affichees dans cet onglet.",
        "glossary_term": "Terme",
        "glossary_definition": "Definition",
        "glossary_business_use": "Comment l'utiliser",
        "ai_copilot": "Copilot IA",
        "ai_business_copilot": "Copilot Business IA",
        "ai_copilot_caption": "Pose des questions business sur le churn, les recommandes, les produits, la retention et les clients.",
        "quick_prompts": "Prompts Rapides",
        "copilot_welcome": "Pose-moi une question sur le churn, les tendances de recommande, les meilleurs produits, la retention ou les segments clients.",
        "prompt_churn": "Pourquoi le churn augmente-t-il?",
        "prompt_products": "Quels produits generent le plus de rachats?",
        "prompt_segments": "Quels segments clients sont a risque de churn?",
        "prompt_retention": "Comment pouvons-nous ameliorer la retention?",
        "ask_ai": "Demander des insights IA",
        "generating_insights": "Generation d'insights depuis les metriques Snowflake...",
        "copilot_api_error": "Impossible de joindre l'API copilot. Verifie API_BASE_URL et le statut Render.",
        "ai_source": "Source IA",
        "summary": "Résumé",
        "impacted_segments": "Segments impactés",
        "metric_comparison": "Comparaison des métriques",
        "segment_snapshot": "Portrait des segments",
        "why_it_matters": "Pourquoi c'est important",
        "supporting_table": "Table de support",
        "recommendations": "Recommandations",
        "priority_mix": "Mix Des Priorites",
        "priority_mix_help": (
            "Ce graphique compte les recommandations par niveau d'urgence. Eleve signifie a traiter en "
            "premier, moyen signifie a planifier ensuite, et faible signifie a surveiller ou optimiser plus tard."
        ),
        "priority_level": "Niveau de priorite",
        "recommendation_count": "Recommandations",
        "priority_high_meaning": "Action immediate: impact business ou reduction du risque la plus forte.",
        "priority_medium_meaning": "Action court terme: amelioration utile apres les points urgents.",
        "priority_low_meaning": "Suivi ou backlog: moins urgent, mais utile pour optimiser.",
        "detailed_insights": "Insights détaillés",
        "evidence": "Evidence",
        "follow_up_questions": "Questions de suivi",
        "decision_engine": "Moteur de decision",
        "decision_engine_caption": "Transforme les anomalies en alertes priorisees, recommandations et explications IA.",
        "scenario_inputs": "Scenario de test",
        "use_gemini": "Utiliser l'explication Gemini",
        "run_decision_engine": "Executer le moteur de decision",
        "decision_api_error": "Impossible de joindre l'API Decision Engine. Verifie API_BASE_URL et le statut Render.",
        "decision_explanation": "Explication de la decision",
        "explanation_source": "Source de l'explication",
        "gemini_status": "Statut Gemini",
        "gemini_requested": "Demande",
        "gemini_configured": "Configure",
        "gemini_yes": "Oui",
        "gemini_no": "Non",
        "gemini_not_configured_help": (
            "Gemini a ete demande, mais l'API Render n'a pas GEMINI_API_KEY configuree. "
            "Ajoute-la dans les variables d'environnement du service API, puis redeploie l'API."
        ),
        "detected_anomalies": "Anomalies detectees",
        "no_anomalies": "Aucune anomalie detectee pour ce scenario.",
        "priority_alerts": "Alertes prioritaires",
        "no_alerts": "Aucune alerte prioritaire retournee.",
        "decision_recommendations": "Recommandations de decision",
        "no_recommendations": "Aucune recommandation retournee.",
        "decision_type": "Type",
        "decision_title": "Decision",
        "decision_action": "Action",
        "decision_rationale": "Raison",
        "scenario_reorder_rate": "Taux de recommande",
        "scenario_churn_rate": "Taux de churn",
        "scenario_days_between_orders": "Jours entre commandes",
        "scenario_data_quality_failures": "Echecs qualite data",
        "scenario_api_health": "Sante API",
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


def request_decision_engine(payload: dict[str, Any]) -> dict[str, Any]:
    api_base_url = settings.api_base_url.rstrip("/")
    response = requests.post(
        f"{api_base_url}/decision",
        json=payload,
        timeout=DECISION_TIMEOUT_SECONDS,
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


def translate_copilot_question(question: str) -> str:
    if get_language() != "fr":
        return question

    question_map = {
        "Why is churn increasing?": translate("prompt_churn"),
        "What products drive repeat purchases?": translate("prompt_products"),
        "Which products are most associated with repeat purchases?": translate("prompt_products"),
        "Which customer segments are at churn risk?": translate("prompt_segments"),
        "How can we improve retention?": translate("prompt_retention"),
        "How does retention change by cohort period?": "Comment la retention evolue-t-elle par periode de cohorte?",
        "Which customer segment has the longest days between orders?": (
            "Quel segment client a le plus long delai entre les commandes?"
        ),
    }
    return question_map.get(question, question)


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


def render_bar_chart(
    frame: pd.DataFrame,
    x_column: str,
    y_column: str,
    *,
    title: str | None = None,
    y_title: str | None = None,
    x_tick_angle: int = -25,
    height: int = 360,
) -> None:
    if frame.empty:
        return

    tooltip = [
        alt.Tooltip(f"{x_column}:N", title=x_column.replace("_", " ").title()),
        alt.Tooltip(f"{y_column}:Q", title=(y_title or y_column).replace("_", " ").title(), format=",.2f"),
    ]
    bars = (
        alt.Chart(frame)
        .mark_bar(color="#2563eb", cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
        .encode(
            x=alt.X(
                f"{x_column}:N",
                sort=None,
                axis=alt.Axis(labelAngle=x_tick_angle, labelLimit=180, title=None),
            ),
            y=alt.Y(f"{y_column}:Q", axis=alt.Axis(title=y_title), scale=alt.Scale(zero=True)),
            tooltip=tooltip,
        )
    )
    labels = bars.mark_text(dy=-8, color="#475569").encode(text=alt.Text(f"{y_column}:Q", format=",.0f"))
    chart = (bars + labels).properties(height=height)
    if title:
        chart = chart.properties(title=title)
    st.altair_chart(chart, use_container_width=True)


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
                "segment": segment_name,
                "metric": metric,
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


def build_priority_explanation_frame(priority_frame: pd.DataFrame) -> pd.DataFrame:
    meanings = {
        "high": translate("priority_high_meaning"),
        "medium": translate("priority_medium_meaning"),
        "low": translate("priority_low_meaning"),
    }
    rows = []
    for priority in ("high", "medium", "low"):
        matching_rows = priority_frame.loc[priority_frame["priority"] == priority, "count"]
        rows.append(
            {
                translate("priority_level"): format_priority_label(priority),
                translate("recommendation_count"): int(matching_rows.iloc[0]) if not matching_rows.empty else 0,
                translate("why_it_matters"): meanings[priority],
            }
        )
    return pd.DataFrame(rows)


def format_segment_metric_value(segment: dict[str, Any]) -> str:
    value = str(segment.get("value") or "0")
    metric = str(segment.get("metric") or "")
    if metric and metric not in value:
        return f"{value}"
    return value


def format_priority_label(priority: str) -> str:
    labels = {
        "high": "HIGH" if get_language() == "en" else "ÉLEVÉ",
        "medium": "MEDIUM" if get_language() == "en" else "MOYEN",
        "low": "LOW" if get_language() == "en" else "FAIBLE",
    }
    return labels.get(priority.lower(), priority.upper())


def render_card_value(value: str) -> None:
    escaped_value = html.escape(value)
    st.markdown(
        (
            "<div style='font-size:2rem;line-height:1.15;font-weight:650;"
            "overflow-wrap:anywhere;margin:0.35rem 0 0.9rem 0;'>"
            f"{escaped_value}"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def render_segment_cards(segments: list[dict[str, Any]]) -> None:
    for row_start in range(0, len(segments), 3):
        row_segments = segments[row_start : row_start + 3]
        columns = st.columns(len(row_segments))
        for column, segment in zip(columns, row_segments, strict=False):
            with column.container(border=True):
                st.caption(str(segment.get("metric") or "metric").replace("_", " ").title())
                st.markdown(f"**{segment.get('segment_name') or 'Segment'}**")
                render_card_value(format_segment_metric_value(segment))
                if reason := segment.get("why_it_matters"):
                    st.write(str(reason))


def render_recommendation_cards(recommendations: list[dict[str, Any]]) -> None:
    for row_start in range(0, len(recommendations), 2):
        row_recommendations = recommendations[row_start : row_start + 2]
        columns = st.columns(len(row_recommendations))
        for column, recommendation in zip(columns, row_recommendations, strict=False):
            priority = str(recommendation.get("priority") or "medium")
            with column.container(border=True):
                st.caption(format_priority_label(priority))
                st.markdown(f"**{recommendation.get('action', '')}**")
                if expected_impact := recommendation.get("expected_impact"):
                    st.write(expected_impact)


def render_decision_cards(decisions: list[dict[str, Any]]) -> None:
    if not decisions:
        return

    for row_start in range(0, len(decisions), 2):
        row_decisions = decisions[row_start : row_start + 2]
        columns = st.columns(len(row_decisions))
        for column, decision in zip(columns, row_decisions, strict=False):
            priority = str(decision.get("priority") or "low")
            with column.container(border=True):
                st.caption(f"{format_priority_label(priority)} | {str(decision.get('decision_type') or '').title()}")
                st.markdown(f"**{decision.get('title', '')}**")
                if action := decision.get("action"):
                    st.write(action)
                if rationale := decision.get("rationale"):
                    st.caption(str(rationale))


def render_priority_mix(priority_frame: pd.DataFrame) -> None:
    if priority_frame.empty:
        return

    display_frame = priority_frame.copy()
    display_frame["priority_label"] = display_frame["priority"].map(format_priority_label)
    chart_column, context_column = st.columns([1.15, 1])
    with chart_column:
        render_bar_chart(
            display_frame,
            "priority_label",
            "count",
            title=translate("priority_mix"),
            y_title=translate("recommendation_count"),
            x_tick_angle=0,
            height=320,
        )
    with context_column:
        st.caption(translate("why_it_matters"))
        st.write(translate("priority_mix_help"))
        st.dataframe(
            build_priority_explanation_frame(priority_frame),
            use_container_width=True,
            hide_index=True,
        )


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
    revenue_proxy.caption(translate("revenue_proxy_help"))
    total_orders.metric(translate("orders"), format_number(kpis.get("total_orders", 0)))
    reorder_rate.metric(translate("reorder_rate"), format_percent(kpis.get("reorder_rate", 0)))


def render_customer_insights(frame: pd.DataFrame) -> None:
    st.subheader(translate("customer_insights"))
    if frame.empty:
        st.info(translate("no_customer_data"))
        return

    chart_data = frame.head(15).melt(
        id_vars="user_id",
        value_vars=["total_orders", "unique_products"],
        var_name="metric",
        value_name="value",
    )
    chart = (
        alt.Chart(chart_data)
        .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
        .encode(
            x=alt.X("user_id:N", axis=alt.Axis(labelAngle=-25, labelLimit=120, title=None)),
            xOffset="metric:N",
            y=alt.Y("value:Q", axis=alt.Axis(title=None), scale=alt.Scale(zero=True)),
            color=alt.Color("metric:N", scale=alt.Scale(range=["#2563eb", "#38bdf8"]), legend=alt.Legend(title=None)),
            tooltip=[
                alt.Tooltip("user_id:N", title="User ID"),
                alt.Tooltip("metric:N", title="Metric"),
                alt.Tooltip("value:Q", title="Value", format=",.0f"),
            ],
        )
        .properties(height=380)
    )
    st.altair_chart(chart, use_container_width=True)

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

    chart_data = frame.head(15)[["product_name", "order_line_count"]]
    render_bar_chart(
        chart_data,
        "product_name",
        "order_line_count",
        y_title="Order lines",
        x_tick_angle=-25,
        height=420,
    )

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
    render_bar_chart(status_summary, "status", "checks", y_title="Checks", x_tick_angle=0, height=300)

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
        render_bar_chart(raw_counts, "table_name", "row_count", y_title="Rows", x_tick_angle=-20, height=360)
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
        render_bar_chart(raw_counts, "table_name", "row_count", y_title="Rows", x_tick_angle=-20, height=340)
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
        render_bar_chart(mart_counts, "table_name", "row_count", y_title="Rows", x_tick_angle=-20, height=340)
        st.dataframe(
            mart_counts,
            use_container_width=True,
            hide_index=True,
            column_config={
                "table_name": "Table",
                "row_count": st.column_config.NumberColumn("Rows", format="%d"),
            },
        )


def build_confusion_summary(confusion_matrix: list[list[int]]) -> dict[str, int]:
    if len(confusion_matrix) < 2 or len(confusion_matrix[0]) < 2 or len(confusion_matrix[1]) < 2:
        return {}

    return {
        "true_negatives": int(confusion_matrix[0][0]),
        "false_positives": int(confusion_matrix[0][1]),
        "false_negatives": int(confusion_matrix[1][0]),
        "true_positives": int(confusion_matrix[1][1]),
    }


def build_classification_report_frame(metrics: dict[str, Any]) -> pd.DataFrame:
    report = metrics.get("classification_report") or {}
    rows = []
    labels = {
        "0": "No Reorder",
        "1": "Reorder",
        "macro avg": "Macro Avg",
        "weighted avg": "Weighted Avg",
    }
    for key, label in labels.items():
        values = report.get(key)
        if not isinstance(values, dict):
            continue

        rows.append(
            {
                "class": label,
                "precision": float(values.get("precision", 0)),
                "recall": float(values.get("recall", 0)),
                "f1_score": float(values.get("f1-score", 0)),
                "support": int(values.get("support", 0)),
            }
        )
    return pd.DataFrame(rows)


def build_default_ml_recommendations(
    metrics: dict[str, Any],
    feature_importance: pd.DataFrame,
) -> list[dict[str, str]]:
    recommendations = []
    positive_rate = float(metrics.get("positive_rate") or 0)
    roc_auc = metrics.get("roc_auc")
    precision = float(metrics.get("precision") or 0)
    recall = float(metrics.get("recall") or 0)

    if positive_rate >= 0.8:
        recommendations.append(
            {
                "priority": "HIGH",
                "action": "Monitor class imbalance before using the model for automated targeting.",
                "expected_impact": (
                    "Repeat-order examples dominate the training data, so use precision, recall, and threshold "
                    "tradeoffs instead of accuracy alone."
                ),
            }
        )

    if roc_auc is not None and float(roc_auc) < 0.72:
        recommendations.append(
            {
                "priority": "MEDIUM",
                "action": "Add richer features: recency, product affinity, department mix, and last-order behavior.",
                "expected_impact": "Better features should improve separation between reorder and churn-risk behavior.",
            }
        )

    if precision > recall:
        recommendations.append(
            {
                "priority": "MEDIUM",
                "action": "Tune the classification threshold before launching retention campaigns.",
                "expected_impact": (
                    "Lower thresholds improve coverage; higher thresholds reduce false positives for costly campaigns."
                ),
            }
        )

    if not feature_importance.empty and feature_importance["importance"].head(2).sum() > 0.85:
        recommendations.append(
            {
                "priority": "MEDIUM",
                "action": "Reduce reliance on only the top features.",
                "expected_impact": "A broader signal set makes predictions more stable and more explainable.",
            }
        )

    return recommendations


def render_line_chart(
    frame: pd.DataFrame,
    x_column: str,
    y_columns: list[str],
    *,
    title: str,
    y_title: str | None = None,
    height: int = 320,
) -> None:
    if frame.empty:
        return

    available_columns = [column for column in y_columns if column in frame.columns]
    if not available_columns:
        return

    chart_frame = frame[[x_column, *available_columns]].melt(
        id_vars=x_column,
        var_name="metric",
        value_name="value",
    )
    chart = (
        alt.Chart(chart_frame)
        .mark_line(point=True, strokeWidth=3)
        .encode(
            x=alt.X(f"{x_column}:Q", axis=alt.Axis(title=x_column.replace("_", " ").title())),
            y=alt.Y("value:Q", axis=alt.Axis(title=y_title or "Value"), scale=alt.Scale(zero=True)),
            color=alt.Color("metric:N", title="Metric"),
            tooltip=[
                alt.Tooltip(f"{x_column}:Q", format=".2f"),
                alt.Tooltip("metric:N"),
                alt.Tooltip("value:Q", format=".3f"),
            ],
        )
        .properties(title=title, height=height)
    )
    st.altair_chart(chart, use_container_width=True)


def render_model_summary_cards(metrics: dict[str, Any], feature_importance: pd.DataFrame) -> None:
    top_feature = "N/A"
    if not feature_importance.empty:
        top_feature = str(feature_importance.iloc[0]["feature"]).replace("_", " ").title()

    with st.container(border=True):
        st.caption(translate("model_explanation"))
        st.write(translate("model_explanation_text"))
        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric(translate("positive_rate"), format_percent(metrics.get("positive_rate", 0)))
        col_b.metric(translate("recommended_threshold"), f"{float(metrics.get('recommended_threshold') or 0.5):.2f}")
        col_c.metric(
            translate("positive_prediction_rate"),
            format_percent(metrics.get("positive_prediction_rate", metrics.get("positive_rate", 0))),
        )
        col_d.metric(translate("selected_model"), str(metrics.get("selected_model") or "random_forest"))
        st.caption(f"Top Driver: {top_feature}")


def render_metric_gap_explanation(metrics: dict[str, Any]) -> None:
    accuracy = float(metrics.get("accuracy") or 0)
    balanced_accuracy = float(metrics.get("balanced_accuracy") or 0)
    positive_rate = float(metrics.get("positive_rate") or 0)
    positive_prediction_rate = float(metrics.get("positive_prediction_rate", positive_rate) or 0)

    if accuracy - balanced_accuracy < 0.15 and positive_rate < 0.8:
        return

    with st.container(border=True):
        st.markdown(f"**{translate('metric_gap_warning')}**")
        st.write(translate("metric_gap_warning_text"))
        gap_a, gap_b, gap_c, gap_d = st.columns(4)
        gap_a.metric(translate("accuracy"), format_percent(accuracy))
        gap_b.metric(translate("balanced_accuracy"), format_percent(balanced_accuracy))
        gap_c.metric(translate("positive_rate"), format_percent(positive_rate))
        gap_d.metric(translate("positive_prediction_rate"), format_percent(positive_prediction_rate))


def build_ml_glossary_rows() -> list[dict[str, str]]:
    if get_language() == "fr":
        return [
            {
                "term": "Accuracy",
                "definition": "Part totale de predictions correctes.",
                "business_use": "Utile comme vue rapide, mais trompeuse si une classe domine fortement.",
            },
            {
                "term": "Balanced Accuracy",
                "definition": "Moyenne de la performance sur les deux classes: reorder et no-reorder.",
                "business_use": "Meilleure metrique ici, car le dataset contient beaucoup plus de reorders.",
            },
            {
                "term": "Precision",
                "definition": "Parmi les clients predits reorder, part qui reorder vraiment.",
                "business_use": "A privilegier quand une campagne coute cher et qu'il faut eviter les faux positifs.",
            },
            {
                "term": "Recall",
                "definition": "Parmi les vrais reorder, part detectee par le modele.",
                "business_use": "A privilegier quand on veut couvrir le plus grand nombre de clients pertinents.",
            },
            {
                "term": "F1 Score",
                "definition": "Equilibre entre precision et recall.",
                "business_use": "Bon score de synthese quand precision et recall sont tous les deux importants.",
            },
            {
                "term": "ROC AUC",
                "definition": "Capacite du modele a classer un reorder au-dessus d'un no-reorder.",
                "business_use": "Utile pour comparer les modeles independamment d'un seuil fixe.",
            },
            {
                "term": "Average Precision",
                "definition": "Resume la courbe precision-recall sur tous les seuils.",
                "business_use": "Utile avec des classes desequilibrees et pour evaluer la qualite du ranking.",
            },
            {
                "term": "Brier Score",
                "definition": "Mesure l'erreur des probabilites predites. Plus bas est meilleur.",
                "business_use": "Indique si les probabilites sont bien calibrees pour prioriser les clients.",
            },
            {
                "term": "Positive Rate",
                "definition": "Part de vrais cas reorder dans les donnees.",
                "business_use": "Explique pourquoi l'accuracy peut paraitre elevee meme si le modele est faible.",
            },
            {
                "term": "Predicted Positive Rate",
                "definition": "Part des clients que le modele classe comme reorder.",
                "business_use": "Aide a voir si le modele cible trop large ou trop strict.",
            },
            {
                "term": "Recommended Threshold",
                "definition": "Seuil de probabilite utilise pour convertir un score en prediction finale.",
                "business_use": "Ajuste le compromis entre couverture client et precision de ciblage.",
            },
            {
                "term": "Confusion Matrix",
                "definition": "Table qui separe vrais positifs, faux positifs, vrais negatifs et faux negatifs.",
                "business_use": "Montre concretement les types d'erreurs que le modele fait.",
            },
        ]

    return [
        {
            "term": "Accuracy",
            "definition": "Share of all predictions that are correct.",
            "business_use": "Useful as a quick view, but misleading when one class dominates.",
        },
        {
            "term": "Balanced Accuracy",
            "definition": "Average performance across both classes: reorder and no-reorder.",
            "business_use": "Better for this project because reorder examples dominate the dataset.",
        },
        {
            "term": "Precision",
            "definition": "Of customers predicted to reorder, the share that actually reorders.",
            "business_use": "Use when campaigns are expensive and false positives should be limited.",
        },
        {
            "term": "Recall",
            "definition": "Of actual reorder customers, the share detected by the model.",
            "business_use": "Use when the goal is broad coverage of relevant customers.",
        },
        {
            "term": "F1 Score",
            "definition": "Balance between precision and recall.",
            "business_use": "Good summary when precision and recall both matter.",
        },
        {
            "term": "ROC AUC",
            "definition": "How well the model ranks reorder cases above no-reorder cases.",
            "business_use": "Useful for comparing models without committing to one threshold.",
        },
        {
            "term": "Average Precision",
            "definition": "Summary of the precision-recall curve across thresholds.",
            "business_use": "Useful with imbalanced classes and ranking-oriented use cases.",
        },
        {
            "term": "Brier Score",
            "definition": "Error of predicted probabilities. Lower is better.",
            "business_use": "Shows whether probabilities are calibrated enough for prioritization.",
        },
        {
            "term": "Positive Rate",
            "definition": "Share of actual reorder cases in the data.",
            "business_use": "Explains why accuracy can look high even when the model is weak.",
        },
        {
            "term": "Predicted Positive Rate",
            "definition": "Share of customers the model classifies as reorder.",
            "business_use": "Shows whether the model targets too broadly or too narrowly.",
        },
        {
            "term": "Recommended Threshold",
            "definition": "Probability cutoff used to convert scores into final predictions.",
            "business_use": "Controls the tradeoff between customer coverage and targeting precision.",
        },
        {
            "term": "Confusion Matrix",
            "definition": "Table separating true positives, false positives, true negatives, and false negatives.",
            "business_use": "Makes the model's error types concrete.",
        },
    ]


def render_ml_glossary() -> None:
    with st.expander(translate("ml_glossary"), expanded=False):
        st.caption(translate("ml_glossary_caption"))
        glossary_frame = pd.DataFrame(build_ml_glossary_rows())
        st.dataframe(
            glossary_frame,
            use_container_width=True,
            hide_index=True,
            column_config={
                "term": st.column_config.TextColumn(translate("glossary_term"), width="medium"),
                "definition": st.column_config.TextColumn(translate("glossary_definition"), width="large"),
                "business_use": st.column_config.TextColumn(translate("glossary_business_use"), width="large"),
            },
        )


def build_model_comparison_frame(metrics: dict[str, Any]) -> pd.DataFrame:
    comparison = metrics.get("model_comparison") or []
    if not comparison:
        return pd.DataFrame()

    frame = pd.DataFrame(comparison)
    metric_columns = [
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "average_precision",
        "brier_score",
    ]
    for column in metric_columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def render_model_comparison(metrics: dict[str, Any]) -> None:
    comparison_frame = build_model_comparison_frame(metrics)
    if comparison_frame.empty:
        return

    st.subheader(translate("model_comparison"))
    st.caption(translate("model_comparison_help"))
    chart_frame = comparison_frame[["model_name", "roc_auc", "f1", "balanced_accuracy"]].melt(
        id_vars="model_name",
        var_name="metric",
        value_name="score",
    )
    chart = (
        alt.Chart(chart_frame)
        .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
        .encode(
            x=alt.X("model_name:N", axis=alt.Axis(title=None, labelAngle=-15), sort=None),
            y=alt.Y("score:Q", axis=alt.Axis(title="Score"), scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("metric:N", title="Metric"),
            xOffset="metric:N",
            tooltip=[
                alt.Tooltip("model_name:N", title="Model"),
                alt.Tooltip("metric:N", title="Metric"),
                alt.Tooltip("score:Q", title="Score", format=".3f"),
            ],
        )
        .properties(height=360)
    )
    st.altair_chart(chart, use_container_width=True)
    st.dataframe(
        comparison_frame,
        use_container_width=True,
        hide_index=True,
        column_config={
            "model_name": "Model",
            "accuracy": st.column_config.ProgressColumn("Accuracy", format="%.3f", min_value=0, max_value=1),
            "balanced_accuracy": st.column_config.ProgressColumn("Balanced Accuracy", format="%.3f", min_value=0, max_value=1),
            "precision": st.column_config.ProgressColumn("Precision", format="%.3f", min_value=0, max_value=1),
            "recall": st.column_config.ProgressColumn("Recall", format="%.3f", min_value=0, max_value=1),
            "f1": st.column_config.ProgressColumn("F1", format="%.3f", min_value=0, max_value=1),
            "roc_auc": st.column_config.ProgressColumn("ROC AUC", format="%.3f", min_value=0, max_value=1),
            "recommended_threshold": st.column_config.NumberColumn("Recommended Threshold", format="%.2f"),
            "positive_prediction_rate": st.column_config.ProgressColumn(
                "Predicted Positive Rate",
                format="%.3f",
                min_value=0,
                max_value=1,
            ),
            "average_precision": st.column_config.ProgressColumn("Avg Precision", format="%.3f", min_value=0, max_value=1),
            "brier_score": st.column_config.NumberColumn("Brier", format="%.3f"),
        },
    )


def render_model_card() -> None:
    model_card_items = [
        ("model_card_task", "model_card_task_value"),
        ("model_card_target", "model_card_target_value"),
        ("model_card_model", "model_card_model_value"),
        ("model_card_use_case", "model_card_use_case_value"),
        ("model_card_features", "model_card_features_value"),
        ("model_card_limitations", "model_card_limitations_value"),
        ("model_card_monitoring", "model_card_monitoring_value"),
    ]

    with st.container(border=True):
        st.markdown(f"**{translate('model_card')}**")
        left_column, right_column = st.columns(2)
        for index, (label_key, value_key) in enumerate(model_card_items):
            target_column = left_column if index % 2 == 0 else right_column
            with target_column:
                st.caption(translate(label_key))
                st.write(translate(value_key))


def render_model_performance(metrics: dict[str, Any], feature_importance: pd.DataFrame) -> None:
    st.subheader(translate("model_performance"))
    if not metrics:
        st.info(translate("no_model_metrics"))
        return

    render_model_card()
    render_model_summary_cards(metrics, feature_importance)
    render_metric_gap_explanation(metrics)
    render_model_comparison(metrics)

    accuracy, balanced_accuracy, precision, recall, f1_score, roc_auc = st.columns(6)
    accuracy.metric(translate("accuracy"), format_percent(metrics.get("accuracy", 0)))
    balanced_accuracy.metric(translate("balanced_accuracy"), format_percent(metrics.get("balanced_accuracy", 0)))
    precision.metric(translate("precision"), format_percent(metrics.get("precision", 0)))
    recall.metric(translate("recall"), format_percent(metrics.get("recall", 0)))
    f1_score.metric(translate("f1_score"), format_percent(metrics.get("f1", 0)))
    roc_auc.metric(translate("roc_auc"), f"{float(metrics.get('roc_auc') or 0):.3f}")

    train_rows, test_rows, positive_rate, avg_precision, brier_score = st.columns(5)
    train_rows.metric(translate("train_rows"), format_number(metrics.get("train_rows", 0)))
    test_rows.metric(translate("test_rows"), format_number(metrics.get("test_rows", 0)))
    positive_rate.metric(translate("positive_rate"), format_percent(metrics.get("positive_rate", 0)))
    avg_precision.metric(translate("average_precision"), f"{float(metrics.get('average_precision') or 0):.3f}")
    brier_score.metric(translate("brier_score"), f"{float(metrics.get('brier_score') or 0):.3f}")
    render_ml_glossary()

    st.divider()

    confusion_matrix = metrics.get("confusion_matrix", [])
    if confusion_matrix:
        st.subheader(translate("prediction_errors"))
        summary = build_confusion_summary(confusion_matrix)
        if summary:
            tn, fp, fn, tp = st.columns(4)
            tn.metric(translate("true_negatives"), format_number(summary["true_negatives"]))
            fp.metric(translate("false_positives"), format_number(summary["false_positives"]))
            fn.metric(translate("false_negatives"), format_number(summary["false_negatives"]))
            tp.metric(translate("true_positives"), format_number(summary["true_positives"]))

        confusion_frame = pd.DataFrame(
            confusion_matrix,
            index=["Actual No Reorder", "Actual Reorder"],
            columns=["Predicted No Reorder", "Predicted Reorder"],
        )
        with st.expander(translate("confusion_matrix")):
            st.dataframe(confusion_frame, use_container_width=True)

    threshold_rows = metrics.get("threshold_analysis") or []
    roc_rows = metrics.get("roc_curve") or []
    pr_rows = metrics.get("precision_recall_curve") or []

    if threshold_rows or roc_rows or pr_rows:
        curve_left, curve_right = st.columns(2)
        with curve_left:
            if threshold_rows:
                render_line_chart(
                    pd.DataFrame(threshold_rows),
                    "threshold",
                    ["precision", "recall", "f1", "balanced_accuracy"],
                    title=translate("threshold_analysis"),
                    y_title="Score",
                    height=330,
                )
            else:
                st.info(translate("no_curve_data"))

        with curve_right:
            if roc_rows:
                render_line_chart(
                    pd.DataFrame(roc_rows),
                    "false_positive_rate",
                    ["true_positive_rate"],
                    title=translate("roc_curve"),
                    y_title="True Positive Rate",
                    height=330,
                )
            elif pr_rows:
                render_line_chart(
                    pd.DataFrame(pr_rows),
                    "recall",
                    ["precision"],
                    title=translate("precision_recall_curve"),
                    y_title="Precision",
                    height=330,
                )
    else:
        st.info(translate("no_curve_data"))

    report_frame = build_classification_report_frame(metrics)
    if not report_frame.empty:
        st.subheader(translate("classification_report"))
        st.dataframe(
            report_frame,
            use_container_width=True,
            hide_index=True,
            column_config={
                "precision": st.column_config.ProgressColumn("Precision", format="%.3f", min_value=0, max_value=1),
                "recall": st.column_config.ProgressColumn("Recall", format="%.3f", min_value=0, max_value=1),
                "f1_score": st.column_config.ProgressColumn("F1", format="%.3f", min_value=0, max_value=1),
                "support": st.column_config.NumberColumn("Support", format="%d"),
            },
        )

    recommendations = metrics.get("model_recommendations") or build_default_ml_recommendations(
        metrics,
        feature_importance,
    )
    if recommendations:
        st.subheader(translate("model_recommendations"))
        render_recommendation_cards(recommendations)

    if not feature_importance.empty:
        st.subheader(translate("feature_importance"))
        st.caption(translate("feature_importance_help"))
        chart_data = feature_importance[["feature", "importance"]]
        render_bar_chart(chart_data, "feature", "importance", y_title="Importance", x_tick_angle=-15, height=390)
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
        with st.container(border=True):
            st.caption(translate("summary"))
            st.markdown(f"**{summary}**")
    if explanation := response.get("explanation"):
        st.write(explanation)

    impacted_segments = response.get("impacted_segments") or []
    if impacted_segments:
        st.markdown(f"**{translate('impacted_segments')}**")
        segment_frame = pd.DataFrame(impacted_segments)
        render_segment_cards(impacted_segments)

        chart_frame = build_segments_chart_frame(impacted_segments)
        if not chart_frame.empty:
            render_bar_chart(
                chart_frame,
                "segment",
                "value",
                title=translate("metric_comparison"),
                y_title="Value",
                x_tick_angle=-20,
                height=430,
            )

        with st.expander(translate("supporting_table")):
            st.dataframe(segment_frame, use_container_width=True, hide_index=True)

    recommendations = response.get("recommendations") or []
    if recommendations:
        st.markdown(f"**{translate('recommendations')}**")
        render_recommendation_cards(recommendations)

        priority_frame = build_recommendation_priority_frame(recommendations)
        if not priority_frame.empty:
            render_priority_mix(priority_frame)

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
            translated_follow_up = translate_copilot_question(follow_up_question)
            if st.button(translated_follow_up, key=button_key):
                queue_copilot_question(translated_follow_up)


def render_ai_copilot() -> None:
    st.subheader(translate("ai_business_copilot"))
    st.caption(translate("ai_copilot_caption"))

    if "copilot_messages" not in st.session_state:
        st.session_state.copilot_messages = [
            {
                "role": "assistant",
                "content": translate("copilot_welcome"),
            }
        ]
    elif st.session_state.copilot_messages:
        first_message = st.session_state.copilot_messages[0]
        welcome_messages = {
            TRANSLATIONS["en"]["copilot_welcome"],
            TRANSLATIONS["fr"]["copilot_welcome"],
            TRANSLATIONS["en"]["ai_copilot_caption"],
            TRANSLATIONS["fr"]["ai_copilot_caption"],
        }
        if first_message.get("role") == "assistant" and first_message.get("content") in welcome_messages:
            first_message["content"] = translate("copilot_welcome")

    examples = [
        translate("prompt_churn"),
        translate("prompt_products"),
        translate("prompt_segments"),
        translate("prompt_retention"),
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


def build_decision_payload(kpis: dict[str, Any], api_health: dict[str, Any] | None = None) -> dict[str, Any]:
    api_status = "ok"
    if api_health and api_health.get("status") != "ok":
        api_status = "failed"

    return {
        "reorder_rate": float(kpis.get("reorder_rate") or 0),
        "churn_rate": 0.50,
        "days_between_orders": 18.0,
        "data_quality_failures": 0,
        "api_health": api_status,
    }


def render_decision_engine(kpis: dict[str, Any]) -> None:
    st.subheader(translate("decision_engine"))
    st.caption(translate("decision_engine_caption"))

    default_payload = build_decision_payload(kpis)
    with st.form("decision_engine_form"):
        st.markdown(f"**{translate('scenario_inputs')}**")
        metric_columns = st.columns(3)
        with metric_columns[0]:
            reorder_rate = st.slider(
                translate("scenario_reorder_rate"),
                min_value=0.0,
                max_value=1.0,
                value=float(default_payload["reorder_rate"]),
                step=0.01,
                format="%.2f",
            )
            data_quality_failures = st.number_input(
                translate("scenario_data_quality_failures"),
                min_value=0,
                value=0,
                step=1,
            )
        with metric_columns[1]:
            churn_rate = st.slider(
                translate("scenario_churn_rate"),
                min_value=0.0,
                max_value=1.0,
                value=0.50,
                step=0.01,
                format="%.2f",
            )
            api_health = st.selectbox(translate("scenario_api_health"), ["ok", "failed"])
        with metric_columns[2]:
            days_between_orders = st.slider(
                translate("scenario_days_between_orders"),
                min_value=0.0,
                max_value=60.0,
                value=18.0,
                step=1.0,
            )
            use_gemini = st.checkbox(translate("use_gemini"), value=False)

        submitted = st.form_submit_button(translate("run_decision_engine"), use_container_width=True)

    if not submitted and "decision_engine_response" not in st.session_state:
        return

    if submitted:
        payload = {
            "data": {
                "reorder_rate": reorder_rate,
                "churn_rate": churn_rate,
                "days_between_orders": days_between_orders,
                "data_quality_failures": data_quality_failures,
                "api_health": api_health,
            },
            "anomalies": [],
            "use_gemini": use_gemini,
        }
        try:
            with st.spinner(translate("run_decision_engine")):
                st.session_state.decision_engine_response = request_decision_engine(payload)
        except requests.RequestException as exc:
            logger.exception("Decision Engine API request failed")
            st.error(translate("decision_api_error"))
            st.caption(str(exc))
            return

    response = st.session_state.get("decision_engine_response") or {}
    decisions = response.get("decisions") or []
    alerts = [decision for decision in decisions if decision.get("decision_type") == "alert"]
    recommendations = [decision for decision in decisions if decision.get("decision_type") == "recommendation"]
    anomalies = response.get("anomalies") or []

    st.divider()
    summary_columns = st.columns(4)
    summary_columns[0].metric(translate("detected_anomalies"), format_number(len(anomalies)))
    summary_columns[1].metric(translate("priority_alerts"), format_number(len(alerts)))
    summary_columns[2].metric(translate("decision_recommendations"), format_number(len(recommendations)))
    summary_columns[3].metric(translate("explanation_source"), str(response.get("explanation_source", "local_fallback")))

    gemini_requested = bool(response.get("gemini_requested"))
    gemini_configured = bool(response.get("gemini_configured"))
    explanation_detail = response.get("explanation_detail")
    with st.expander(translate("gemini_status"), expanded=gemini_requested):
        status_frame = pd.DataFrame(
            [
                {
                    translate("gemini_requested"): translate("gemini_yes")
                    if gemini_requested
                    else translate("gemini_no"),
                    translate("gemini_configured"): translate("gemini_yes")
                    if gemini_configured
                    else translate("gemini_no"),
                    translate("explanation_source"): response.get("explanation_source", "local_fallback"),
                }
            ]
        )
        st.dataframe(status_frame, use_container_width=True, hide_index=True)
        if explanation_detail:
            st.caption(str(explanation_detail))
        if gemini_requested and not gemini_configured:
            st.warning(translate("gemini_not_configured_help"))

    if explanation := response.get("explanation"):
        with st.container(border=True):
            st.caption(translate("decision_explanation"))
            st.write(explanation)

    st.markdown(f"**{translate('priority_alerts')}**")
    if alerts:
        render_decision_cards(alerts)
    else:
        st.info(translate("no_alerts"))

    st.markdown(f"**{translate('decision_recommendations')}**")
    if recommendations:
        render_decision_cards(recommendations)
    else:
        st.info(translate("no_recommendations"))

    st.markdown(f"**{translate('detected_anomalies')}**")
    if anomalies:
        st.dataframe(pd.DataFrame(anomalies), use_container_width=True, hide_index=True)
    else:
        st.info(translate("no_anomalies"))


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
    except Exception as exc:
        logger.exception("Failed to load dashboard KPIs")
        st.warning("Unable to load live Snowflake KPIs. Showing empty defaults while the app remains available.")
        st.caption(str(exc))
        kpis = {"total_orders": 0, "revenue_proxy": 0, "reorder_rate": 0}

    model_metrics = load_model_metrics()
    feature_importance = load_feature_importance()

    render_kpis(kpis)
    st.divider()

    (
        overview_tab,
        customer_tab,
        product_tab,
        quality_tab,
        health_tab,
        model_tab,
        decision_tab,
        copilot_tab,
    ) = st.tabs(
        [
            translate("project_overview"),
            translate("customer_insights"),
            translate("product_trends"),
            translate("data_quality"),
            translate("pipeline_health"),
            translate("model_performance"),
            translate("decision_engine"),
            translate("ai_copilot"),
        ]
    )
    with overview_tab:
        render_project_overview()
    with customer_tab:
        try:
            render_customer_insights(load_customer_insights())
        except Exception as exc:
            logger.exception("Failed to load customer insights")
            st.error("Unable to load customer insights from Snowflake.")
            st.caption(str(exc))
    with product_tab:
        try:
            render_product_trends(load_product_trends())
        except Exception as exc:
            logger.exception("Failed to load product trends")
            st.error("Unable to load product trends from Snowflake.")
            st.caption(str(exc))
    with quality_tab:
        render_data_quality()
    with health_tab:
        render_pipeline_health()
    with model_tab:
        render_model_performance(model_metrics, feature_importance)
    with decision_tab:
        render_decision_engine(kpis)
    with copilot_tab:
        render_ai_copilot()


if __name__ == "__main__":
    main()
