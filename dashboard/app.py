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
        "customer_insights_caption": (
            "This view summarizes customer behavior by segment instead of ranking only the highest-order users."
        ),
        "customer_data_context": (
            "Important context: Instacart order history is capped at 100 observed orders per user, so many users "
            "show exactly 100 orders. Basket and product metrics are also based on the available order-product "
            "detail, so the page highlights data coverage and behavior segments instead of treating 100 orders "
            "as a literal lifetime total."
        ),
        "total_customers": "Customers",
        "avg_orders": "Avg Orders",
        "observed_baskets": "Observed Baskets",
        "avg_basket_size": "Avg Basket Size",
        "reorder_order_ratio": "Reorder Order Ratio",
        "days_between_orders": "Days Between Orders",
        "order_cap_share": "At 100-Order Cap",
        "avg_observed_coverage": "Documented Baskets",
        "avg_reorder_ratio": "Avg Reorder Ratio",
        "unique_products": "Unique Products",
        "monthly_frequency": "Monthly Frequency",
        "customer_segment_summary": "Customer Segment Summary",
        "customer_segment_summary_caption": (
            "Segments combine reorder behavior, purchase interval, product breadth, and data coverage. "
            "They are more useful than a simple top-user ranking."
        ),
        "behavior_segments_definition": (
            "Behavioral segments group customers by how they buy, not only by how many orders they have. "
            "They combine reorder intensity, purchase gaps, product variety, and basket item data coverage to "
            "turn raw customer rows into business actions."
        ),
        "basket_detail_definition": (
            "Basket item data means the products attached to each order. When coverage is low, we may know "
            "that a customer placed orders, but not enough about the exact products inside those baskets."
        ),
        "segment_description": "How to read it",
        "segment_guide": "Segment Guide",
        "customer_segment_snapshot": "Segment Snapshot",
        "sample_customer_profiles": "Representative Customer Profiles",
        "customer_id": "Customer ID",
        "limited_basket_detail_description": (
            "Many orders do not have enough product-level basket data. Treat product and reorder-product conclusions cautiously."
        ),
        "loyal_reorder_description": (
            "Customer shows strong repeat-purchase behavior. Useful for loyalty, retention, and repeat-product analysis."
        ),
        "at_risk_customer_description": (
            "Customer has weak reorder behavior or long purchase gaps. This group is a priority for retention actions."
        ),
        "product_explorer_description": (
            "Customer buys across a wider product set. Good candidate for discovery offers and personalized recommendations."
        ),
        "steady_customer_description": (
            "Customer has regular behavior without a strong risk or loyalty signal. Monitor for movement into other segments."
        ),
        "customer_distribution": "Order Distribution",
        "customer_behavior_map": "Customer Behavior Map",
        "customer_behavior_map_caption": (
            "Each point is a representative customer sample. The x-axis shows observed order frequency, "
            "the y-axis shows reorder intensity, and point size reflects product breadth."
        ),
        "representative_customers": "Representative Customers",
        "customer_segment": "Customer Segment",
        "users": "Users",
        "observed_basket_coverage": "Documented Basket Coverage",
        "limited_basket_detail": "Low basket item coverage",
        "loyal_reorder": "Loyal reorder customers",
        "at_risk_customer": "At-risk customers",
        "product_explorer": "Product explorers",
        "steady_customer": "Steady customers",
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
        "business_scenario_levers": "Business scenario levers",
        "business_scenario_caption": (
            "These values let you test business scenarios. Operational signals below are detected automatically."
        ),
        "detected_operational_signals": "Detected operational signals",
        "detected_operational_caption": (
            "Data quality and API health are read from live checks; they cannot be manually selected here."
        ),
        "decision_engine_how_it_works": "How the decision engine works",
        "decision_engine_how_it_works_text": (
            "The engine reads business metrics and operational checks, detects anomalies against rule thresholds, "
            "prioritizes decisions by severity, then generates a Gemini explanation when available."
        ),
        "decision_rulebook": "Decision rulebook",
        "decision_rule": "Rule",
        "decision_trigger": "Trigger",
        "decision_output": "Output",
        "decision_signal_source": "Signal source",
        "observed_value": "Observed value",
        "signal_detail": "Detail",
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
        "customer_insights_caption": (
            "Cette vue resume le comportement client par segment au lieu de classer seulement les utilisateurs "
            "avec le plus de commandes."
        ),
        "customer_data_context": (
            "Contexte important: l'historique Instacart est plafonne a 100 commandes observees par utilisateur. "
            "Il est donc normal de voir plusieurs clients a exactement 100 commandes. Les metriques panier et "
            "produit reposent aussi sur le detail order-product disponible; la page met donc l'accent sur la "
            "couverture des donnees et les segments comportementaux."
        ),
        "total_customers": "Clients",
        "avg_orders": "Commandes moyennes",
        "observed_baskets": "Paniers observes",
        "avg_basket_size": "Taille moyenne du panier",
        "reorder_order_ratio": "Ratio de commandes avec recommande",
        "days_between_orders": "Jours entre commandes",
        "order_cap_share": "Au plafond de 100",
        "avg_observed_coverage": "Paniers documentes",
        "avg_reorder_ratio": "Taux moyen de recommande",
        "unique_products": "Produits uniques",
        "monthly_frequency": "Frequence mensuelle",
        "customer_segment_summary": "Resume des segments clients",
        "customer_segment_summary_caption": (
            "Les segments combinent comportement de recommande, delai entre commandes, diversite produits et "
            "couverture des donnees. C'est plus utile qu'un simple classement des meilleurs utilisateurs."
        ),
        "behavior_segments_definition": (
            "Les segments comportementaux regroupent les clients selon leur facon d'acheter, pas seulement selon "
            "leur nombre de commandes. Ils combinent intensite de recommande, delai entre commandes, diversite "
            "produits et couverture des produits dans les paniers pour transformer des lignes client brutes "
            "en actions business."
        ),
        "basket_detail_definition": (
            "Les donnees de panier designent les produits rattaches a chaque commande. Quand la couverture est "
            "faible, on sait qu'un client a passe des commandes, mais on connait mal le contenu exact de ses paniers."
        ),
        "segment_description": "Comment l'interpreter",
        "segment_guide": "Guide des segments",
        "customer_segment_snapshot": "Synthese par segment",
        "sample_customer_profiles": "Profils clients representatifs",
        "customer_id": "Client ID",
        "limited_basket_detail_description": (
            "Beaucoup de commandes n'ont pas assez de donnees produit rattachees au panier. Les conclusions "
            "sur les produits et les rachats produit doivent rester prudentes."
        ),
        "loyal_reorder_description": (
            "Le client montre un fort comportement de rachat. Utile pour analyser fidelite, retention et produits recurrents."
        ),
        "at_risk_customer_description": (
            "Le client a peu de recommandes ou de longs delais entre commandes. Segment prioritaire pour la retention."
        ),
        "product_explorer_description": (
            "Le client achete une variete plus large de produits. Bon candidat pour les offres de decouverte et recommandations."
        ),
        "steady_customer_description": (
            "Le comportement est regulier, sans signal fort de risque ou de fidelite intense. A surveiller dans le temps."
        ),
        "customer_distribution": "Distribution des commandes",
        "customer_behavior_map": "Carte comportementale client",
        "customer_behavior_map_caption": (
            "Chaque point est un client d'un echantillon representatif. L'axe x montre la frequence observee, "
            "l'axe y l'intensite de recommande, et la taille du point reflete la diversite produits."
        ),
        "representative_customers": "Clients representatifs",
        "customer_segment": "Segment client",
        "users": "Utilisateurs",
        "observed_basket_coverage": "Couverture des paniers documentes",
        "limited_basket_detail": "Paniers peu documentes",
        "loyal_reorder": "Clients fideles en recommande",
        "at_risk_customer": "Clients a risque",
        "product_explorer": "Explorateurs produits",
        "steady_customer": "Clients stables",
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
        "business_scenario_levers": "Leviers de scenario business",
        "business_scenario_caption": (
            "Ces valeurs servent a tester des scenarios business. Les signaux operationnels ci-dessous sont detectes automatiquement."
        ),
        "detected_operational_signals": "Signaux operationnels detectes",
        "detected_operational_caption": (
            "La qualite des donnees et la sante API viennent de checks live; elles ne sont pas selectionnees manuellement."
        ),
        "decision_engine_how_it_works": "Comment fonctionne le moteur de decision",
        "decision_engine_how_it_works_text": (
            "Le moteur lit les metriques business et les checks operationnels, detecte les anomalies avec des seuils, "
            "priorise les decisions par severite, puis genere une explication Gemini lorsque disponible."
        ),
        "decision_rulebook": "Regles du moteur",
        "decision_rule": "Regle",
        "decision_trigger": "Declencheur",
        "decision_output": "Sortie",
        "decision_signal_source": "Source du signal",
        "observed_value": "Valeur observee",
        "signal_detail": "Detail",
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


FEATURE_STORE_COLUMN_TYPES = {
    "user_id": "number",
    "total_orders": "number",
    "observed_basket_orders": "number",
    "avg_basket_size": "float",
    "stddev_basket_size": "float",
    "reorder_ratio": "float",
    "reorder_order_ratio": "float",
    "unique_products": "number",
    "unique_departments": "number",
    "unique_aisles": "number",
    "produce_item_ratio": "float",
    "dairy_eggs_item_ratio": "float",
    "fresh_fruits_item_ratio": "float",
    "fresh_vegetables_item_ratio": "float",
    "days_between_orders": "float",
    "stddev_days_between_orders": "float",
    "customer_tenure_days": "float",
    "max_days_between_orders": "float",
    "avg_order_dow": "float",
    "order_frequency_30d": "float",
    "avg_order_hour_of_day": "float",
    "weekend_order_ratio": "float",
    "evening_order_ratio": "float",
    "active_order_dow_count": "number",
    "observed_basket_coverage": "float",
    "will_reorder": "number",
}


def load_table_columns(table_name: str) -> set[str]:
    query = f"""
        select lower(column_name) as column_name
        from information_schema.columns
        where table_schema = current_schema()
            and lower(table_name) = lower('{table_name}')
    """
    frame = query_snowflake(query)
    if frame.empty:
        return set()
    return set(frame["column_name"].astype(str))


def build_feature_store_select_columns(existing_columns: set[str]) -> str:
    select_parts = []
    for column_name, column_type in FEATURE_STORE_COLUMN_TYPES.items():
        if column_name in existing_columns:
            select_parts.append(f"            {column_name}")
        else:
            select_parts.append(f"            cast(null as {column_type}) as {column_name}")

    return ",\n".join(select_parts)


def customer_segment_case() -> str:
    return """
        case
            when coalesce(observed_basket_coverage, 0) < 0.05 then 'limited_basket_detail'
            when coalesce(reorder_ratio, 0) >= 0.60
                and coalesce(total_orders, 0) >= 25 then 'loyal_reorder'
            when coalesce(days_between_orders, 0) >= 21
                or coalesce(reorder_ratio, 0) < 0.25 then 'at_risk_customer'
            when coalesce(unique_products, 0) >= 15 then 'product_explorer'
            else 'steady_customer'
        end
    """


def feature_store_segmented_cte(existing_columns: set[str]) -> str:
    select_columns = build_feature_store_select_columns(existing_columns)
    segment_expression = customer_segment_case()
    return f"""
        with selected as (
            select
{select_columns}
            from feature_store
        ),

        segmented as (
            select
                selected.*,
                {segment_expression} as customer_segment
            from selected
        )
    """


def load_customer_insights() -> pd.DataFrame:
    existing_columns = load_table_columns("feature_store")
    segmented_cte = feature_store_segmented_cte(existing_columns)
    query = f"""
{segmented_cte},

        sampled as (
            select
                segmented.*,
                row_number() over (
                    partition by customer_segment
                    order by abs(hash(user_id))
                ) as segment_sample_rank
            from segmented
        )

        select *
        from sampled
        where segment_sample_rank <= 50
        order by customer_segment, total_orders desc, user_id
    """
    return query_snowflake(query)


def load_customer_profile_summary() -> dict[str, Any]:
    existing_columns = load_table_columns("feature_store")
    segmented_cte = feature_store_segmented_cte(existing_columns)
    query = f"""
{segmented_cte}

        select
            count(*) as total_customers,
            avg(total_orders) as avg_total_orders,
            median(total_orders) as median_total_orders,
            avg(reorder_ratio) as avg_reorder_ratio,
            avg(observed_basket_coverage) as avg_observed_basket_coverage,
            count_if(total_orders >= 100) / nullif(count(*), 0) as order_cap_share,
            avg(days_between_orders) as avg_days_between_orders,
            avg(unique_products) as avg_unique_products
        from segmented
    """
    frame = query_snowflake(query)
    if frame.empty:
        return {}
    return frame.iloc[0].to_dict()


def load_customer_segment_summary() -> pd.DataFrame:
    existing_columns = load_table_columns("feature_store")
    segmented_cte = feature_store_segmented_cte(existing_columns)
    query = f"""
{segmented_cte}

        select
            customer_segment,
            count(*) as users,
            avg(total_orders) as avg_total_orders,
            avg(reorder_ratio) as avg_reorder_ratio,
            avg(observed_basket_coverage) as avg_observed_basket_coverage,
            avg(unique_products) as avg_unique_products,
            avg(days_between_orders) as avg_days_between_orders,
            avg(order_frequency_30d) as avg_order_frequency_30d
        from segmented
        group by customer_segment
        order by users desc
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


def load_decision_observed_signals() -> dict[str, Any]:
    signals: dict[str, Any] = {
        "data_quality_failures": 0,
        "data_quality_status": "unknown",
        "data_quality_detail": "Data quality checks have not been evaluated.",
        "api_health": "failed",
        "api_health_status": "failed",
        "api_health_detail": "API health has not been evaluated.",
    }

    try:
        quality_results = load_data_quality_results()
        total_checks = len(quality_results)
        failed_checks = int((quality_results["status"] != "Passed").sum()) if total_checks else 0
        signals.update(
            {
                "data_quality_failures": failed_checks,
                "data_quality_status": "passed" if failed_checks == 0 else "failed",
                "data_quality_detail": f"{failed_checks} failed checks out of {total_checks}",
            }
        )
    except Exception as exc:
        logger.exception("Decision Engine data quality signal check failed")
        signals.update(
            {
                "data_quality_failures": 1,
                "data_quality_status": "unavailable",
                "data_quality_detail": str(exc),
            }
        )

    try:
        api_detail = load_api_health()
        api_ok = api_detail.get("status") == "ok"
        signals.update(
            {
                "api_health": "ok" if api_ok else "failed",
                "api_health_status": "ok" if api_ok else "failed",
                "api_health_detail": format_api_health_detail(api_detail),
            }
        )
    except Exception as exc:
        logger.exception("Decision Engine API health signal check failed")
        signals.update(
            {
                "api_health": "failed",
                "api_health_status": "failed",
                "api_health_detail": str(exc),
            }
        )

    return signals


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


DECISION_OUTPUT_TRANSLATIONS_FR = {
    "alert": "Alerte",
    "recommendation": "Recommandation",
    "local_fallback": "Fallback local",
    "gemini": "Gemini",
    "Generated with gemini-2.5-flash.": "Genere avec gemini-2.5-flash.",
    "Gemini was not requested for this run.": "Gemini n'a pas ete demande pour cette execution.",
    "GEMINI_API_KEY is not configured on the API service.": (
        "GEMINI_API_KEY n'est pas configuree sur le service API."
    ),
    "Reorder rate is below the healthy operating threshold.": (
        "Le taux de recommande est sous le seuil operationnel sain."
    ),
    "Churn risk is critically high.": "Le risque de churn est critique.",
    "Churn risk is elevated.": "Le risque de churn est eleve.",
    "Average purchase gap is unusually long.": "Le delai moyen entre commandes est anormalement long.",
    "Data quality checks are failing before downstream decisions.": (
        "Des controles de qualite des donnees echouent avant les decisions downstream."
    ),
    "Serving API is unavailable or unhealthy.": "L'API de service est indisponible ou non saine.",
    "Pause downstream analytics refresh": "Suspendre le rafraichissement analytique downstream",
    "Investigate failing data quality checks before running dbt, training, or dashboard refreshes.": (
        "Analyser les controles qualite en echec avant de lancer dbt, le training ou le "
        "rafraichissement du dashboard."
    ),
    "Data quality failures can contaminate marts, model features, and AI insights.": (
        "Les echecs de qualite peuvent contaminer les marts, les features modele et les insights IA."
    ),
    "Restore prediction API availability": "Restaurer la disponibilite de l'API de prediction",
    "Check Render logs, health endpoint, model artifact availability, and environment variables.": (
        "Verifier les logs Render, le endpoint health, la disponibilite de l'artefact modele "
        "et les variables d'environnement."
    ),
    "The decision and dashboard layers depend on the API for live predictions and copilot workflows.": (
        "Les couches decision et dashboard dependent de l'API pour les predictions live et "
        "les workflows copilot."
    ),
    "Launch retention intervention": "Lancer une intervention de retention",
    "Target high-risk customers with reorder reminders, high-repeat products, and time-bound offers.": (
        "Cibler les clients a risque avec des rappels de recommande, des produits a fort rachat "
        "et des offres limitees dans le temps."
    ),
    "Churn risk, weak reorder behavior, or long purchase gaps indicate declining customer engagement.": (
        "Le risque de churn, la faiblesse des recommandes ou les longs delais entre commandes "
        "indiquent une baisse d'engagement client."
    ),
    "Review revenue proxy decline": "Analyser la baisse du proxy de revenus",
    "Compare order volume, basket size, and repeat purchase movement by cohort and product department.": (
        "Comparer le volume de commandes, la taille du panier et les rachats par cohorte "
        "et departement produit."
    ),
    "A declining revenue proxy can signal lower basket activity even without price-level data.": (
        "Une baisse du proxy de revenus peut indiquer une baisse d'activite panier meme sans "
        "donnees de prix."
    ),
    "Continue monitoring customer health": "Continuer le suivi de la sante client",
    "Track reorder rate, retention cohorts, churn risk, and data quality daily.": (
        "Suivre chaque jour le taux de recommande, les cohortes de retention, le risque de churn "
        "et la qualite des donnees."
    ),
    "No critical anomaly was detected, so the best decision is ongoing monitoring.": (
        "Aucune anomalie critique n'a ete detectee; la meilleure decision est donc de continuer "
        "la surveillance."
    ),
}


def translate_decision_output_text(value: Any) -> str:
    text_value = "" if value is None else str(value)
    if get_language() != "fr":
        return text_value
    if text_value.startswith("Gemini request failed with HTTP status "):
        status_code = text_value.removeprefix("Gemini request failed with HTTP status ").removesuffix(".")
        return f"La requete Gemini a echoue avec le statut HTTP {status_code}."
    return DECISION_OUTPUT_TRANSLATIONS_FR.get(text_value, text_value)


def translate_decision_record(record: dict[str, Any]) -> dict[str, Any]:
    if get_language() != "fr":
        return record

    localized_record = dict(record)
    for field_name in ("title", "action", "rationale", "description"):
        if field_name in localized_record:
            localized_record[field_name] = translate_decision_output_text(localized_record[field_name])
    evidence = localized_record.get("evidence")
    if isinstance(evidence, dict) and "description" in evidence:
        localized_record["evidence"] = {
            **evidence,
            "description": translate_decision_output_text(evidence.get("description")),
        }
    return localized_record


FEATURE_LABELS = {
    "en": {
        "total_orders": "Total Orders",
        "observed_basket_orders": "Documented Basket Orders",
        "avg_basket_size": "Average Basket Size",
        "reorder_ratio": "Reorder Ratio",
        "days_between_orders": "Days Between Orders",
        "stddev_days_between_orders": "Order Gap Variability",
        "days_since_last_order": "Days Since Previous Order",
        "customer_tenure_days": "Observed Customer Tenure",
        "order_frequency_30d": "30-Day Order Frequency",
        "avg_order_dow": "Average Order Day",
        "avg_order_hour_of_day": "Average Order Hour",
        "weekend_order_ratio": "Weekend Order Share",
        "evening_order_ratio": "Evening Order Share",
        "prior_reorder_order_ratio": "Prior Reorder Order Share",
        "unique_products": "Unique Products",
        "unique_departments": "Unique Departments",
        "unique_aisles": "Unique Aisles",
        "produce_item_ratio": "Produce Item Share",
        "dairy_eggs_item_ratio": "Dairy/Eggs Item Share",
        "fresh_fruits_item_ratio": "Fresh Fruits Item Share",
        "fresh_vegetables_item_ratio": "Fresh Vegetables Item Share",
    },
    "fr": {
        "total_orders": "Commandes",
        "observed_basket_orders": "Paniers documentes",
        "avg_basket_size": "Taille moyenne du panier",
        "reorder_ratio": "Taux de recommande",
        "days_between_orders": "Jours entre commandes",
        "stddev_days_between_orders": "Variabilite du delai",
        "days_since_last_order": "Jours depuis la commande precedente",
        "customer_tenure_days": "Anciennete observee",
        "order_frequency_30d": "Frequence sur 30 jours",
        "avg_order_dow": "Jour moyen de commande",
        "avg_order_hour_of_day": "Heure moyenne de commande",
        "weekend_order_ratio": "Part commandes week-end",
        "evening_order_ratio": "Part commandes soir/nuit",
        "prior_reorder_order_ratio": "Part commandes avec rachat",
        "unique_products": "Produits uniques",
        "unique_departments": "Departements uniques",
        "unique_aisles": "Rayons uniques",
        "produce_item_ratio": "Part articles fruits/legumes",
        "dairy_eggs_item_ratio": "Part laitier/oeufs",
        "fresh_fruits_item_ratio": "Part fruits frais",
        "fresh_vegetables_item_ratio": "Part legumes frais",
    },
}


ML_TEXT_TRANSLATIONS_FR = {
    "Monitor class imbalance before using the model for automated targeting.": (
        "Surveiller le desequilibre des classes avant d'utiliser le modele pour du ciblage automatise."
    ),
    "The dataset contains many reorder-positive examples, so accuracy alone can overstate model quality. Use balanced accuracy, ROC AUC, and threshold analysis together.": (
        "Le dataset contient beaucoup d'exemples positifs de recommande; l'accuracy seule peut donc surestimer la qualite du modele. Utilise ensemble l'accuracy equilibree, le ROC AUC et l'analyse des seuils."
    ),
    "Repeat-order examples dominate the training data, so use precision, recall, and threshold tradeoffs instead of accuracy alone.": (
        "Les exemples de recommande dominent les donnees d'entrainement; utilise les compromis precision, recall et seuil plutot que l'accuracy seule."
    ),
    "Treat the current classifier as a ranking signal, not a final automated decision.": (
        "Traiter le classifieur actuel comme un signal de priorisation, pas comme une decision automatisee finale."
    ),
    "Balanced accuracy is close to random because the no-reorder class is hard to detect. Use the probability score for prioritization until richer negative-class features are added.": (
        "L'accuracy equilibree est proche du hasard, car la classe sans recommande est difficile a detecter. Utilise le score de probabilite pour prioriser jusqu'a l'ajout de features plus riches pour la classe negative."
    ),
    "Add richer behavioral features such as recency, product affinity, and department mix.": (
        "Ajouter des features comportementales plus riches comme la recence, l'affinite produit et le mix departement."
    ),
    "Add richer features: recency, product affinity, department mix, and last-order behavior.": (
        "Ajouter des features plus riches: recence, affinite produit, mix departement et comportement de la derniere commande."
    ),
    "Improves separation between likely reorders and customers who may churn.": (
        "Ameliore la separation entre les clients susceptibles de recommander et ceux qui risquent de churner."
    ),
    "Better features should improve separation between reorder and churn-risk behavior.": (
        "De meilleures features devraient mieux separer les comportements de recommande et de risque de churn."
    ),
    "Tune the prediction threshold based on campaign goals.": (
        "Ajuster le seuil de prediction selon les objectifs de campagne."
    ),
    "Tune the classification threshold before launching retention campaigns.": (
        "Ajuster le seuil de classification avant de lancer des campagnes de retention."
    ),
    "Lower thresholds catch more potential reorderers; higher thresholds reduce false positives for expensive retention campaigns.": (
        "Un seuil plus bas couvre plus de clients susceptibles de recommander; un seuil plus haut reduit les faux positifs pour les campagnes couteuses."
    ),
    "Lower thresholds improve coverage; higher thresholds reduce false positives for costly campaigns.": (
        "Un seuil plus bas augmente la couverture; un seuil plus haut reduit les faux positifs pour les campagnes couteuses."
    ),
    "Reduce dependence on only a few features.": (
        "Reduire la dependance a quelques features seulement."
    ),
    "Reduce reliance on only the top features.": (
        "Reduire la dependance aux features dominantes."
    ),
    "A broader feature set usually makes the model more stable and easier to explain to business users.": (
        "Un ensemble de features plus large rend generalement le modele plus stable et plus facile a expliquer aux utilisateurs business."
    ),
    "A broader signal set makes predictions more stable and more explainable.": (
        "Un ensemble de signaux plus large rend les predictions plus stables et plus explicables."
    ),
}


def format_feature_label(feature_name: Any) -> str:
    text_value = "" if feature_name is None else str(feature_name)
    labels = FEATURE_LABELS.get(get_language(), FEATURE_LABELS["en"])
    return labels.get(text_value, text_value.replace("_", " ").title())


def translate_ml_text(value: Any) -> str:
    text_value = "" if value is None else str(value)
    if get_language() != "fr":
        return text_value
    return ML_TEXT_TRANSLATIONS_FR.get(text_value, text_value)


def metric_display_label(metric_name: str) -> str:
    labels = {
        "accuracy": translate("accuracy"),
        "balanced_accuracy": translate("balanced_accuracy"),
        "precision": translate("precision"),
        "recall": translate("recall"),
        "f1": translate("f1_score"),
        "roc_auc": translate("roc_auc"),
        "average_precision": translate("average_precision"),
        "brier_score": translate("brier_score"),
        "recommended_threshold": translate("recommended_threshold"),
        "positive_prediction_rate": translate("positive_prediction_rate"),
    }
    return labels.get(metric_name, metric_name.replace("_", " ").title())


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
    label_format: str = ",.0f",
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
    labels = bars.mark_text(dy=-8, color="#475569").encode(text=alt.Text(f"{y_column}:Q", format=label_format))
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
                st.markdown(f"**{translate_ml_text(recommendation.get('action', ''))}**")
                if expected_impact := recommendation.get("expected_impact"):
                    st.write(translate_ml_text(expected_impact))


def render_decision_cards(decisions: list[dict[str, Any]]) -> None:
    if not decisions:
        return

    decisions = [translate_decision_record(decision) for decision in decisions]
    for row_start in range(0, len(decisions), 2):
        row_decisions = decisions[row_start : row_start + 2]
        columns = st.columns(len(row_decisions))
        for column, decision in zip(columns, row_decisions, strict=False):
            priority = str(decision.get("priority") or "low")
            with column.container(border=True):
                decision_type = translate_decision_output_text(decision.get("decision_type"))
                st.caption(f"{format_priority_label(priority)} | {decision_type.title()}")
                st.markdown(f"**{decision.get('title', '')}**")
                if action := decision.get("action"):
                    st.write(action)
                if rationale := decision.get("rationale"):
                    st.caption(str(rationale))


def render_decision_engine_explainer() -> None:
    with st.expander(translate("decision_engine_how_it_works"), expanded=True):
        st.write(translate("decision_engine_how_it_works_text"))
        rules = [
            {
                translate("decision_rule"): "Low reorder rate",
                translate("decision_trigger"): "reorder_rate < 0.35",
                translate("priority_level"): "high",
                translate("decision_output"): "Retention recommendation",
            },
            {
                translate("decision_rule"): "Elevated churn",
                translate("decision_trigger"): "churn_rate >= 0.50",
                translate("priority_level"): "high / critical",
                translate("decision_output"): "Retention intervention",
            },
            {
                translate("decision_rule"): "Long order gap",
                translate("decision_trigger"): "days_between_orders >= 21",
                translate("priority_level"): "high",
                translate("decision_output"): "Reorder reminder action",
            },
            {
                translate("decision_rule"): "Data quality failure",
                translate("decision_trigger"): "failed_checks > 0",
                translate("priority_level"): "critical",
                translate("decision_output"): "Pause downstream refresh",
            },
            {
                translate("decision_rule"): "API unhealthy",
                translate("decision_trigger"): "health endpoint != ok",
                translate("priority_level"): "critical",
                translate("decision_output"): "Restore API availability",
            },
        ]
        st.caption(translate("decision_rulebook"))
        st.dataframe(pd.DataFrame(rules), use_container_width=True, hide_index=True)


def render_detected_operational_signals(signals: dict[str, Any]) -> None:
    st.markdown(f"**{translate('detected_operational_signals')}**")
    st.caption(translate("detected_operational_caption"))
    signal_rows = [
        {
            translate("decision_signal_source"): translate("scenario_data_quality_failures"),
            translate("observed_value"): signals.get("data_quality_failures", 0),
            translate("signal_detail"): signals.get("data_quality_detail", ""),
        },
        {
            translate("decision_signal_source"): translate("scenario_api_health"),
            translate("observed_value"): signals.get("api_health_status", "unknown"),
            translate("signal_detail"): signals.get("api_health_detail", ""),
        },
    ]
    st.dataframe(pd.DataFrame(signal_rows), use_container_width=True, hide_index=True)


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


def translate_customer_segment(segment: Any) -> str:
    segment_key = str(segment or "steady_customer")
    return translate(segment_key) if segment_key in TRANSLATIONS["en"] else segment_key.replace("_", " ").title()


def build_customer_summary_from_sample(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {}

    total_orders = pd.to_numeric(frame.get("total_orders"), errors="coerce")
    reorder_ratio = pd.to_numeric(frame.get("reorder_ratio"), errors="coerce")
    coverage = pd.to_numeric(frame.get("observed_basket_coverage"), errors="coerce")
    days_between_orders = pd.to_numeric(frame.get("days_between_orders"), errors="coerce")
    unique_products = pd.to_numeric(frame.get("unique_products"), errors="coerce")
    return {
        "total_customers": frame["user_id"].nunique() if "user_id" in frame else len(frame),
        "avg_total_orders": total_orders.mean(),
        "median_total_orders": total_orders.median(),
        "avg_reorder_ratio": reorder_ratio.mean(),
        "avg_observed_basket_coverage": coverage.mean(),
        "order_cap_share": (total_orders >= 100).mean(),
        "avg_days_between_orders": days_between_orders.mean(),
        "avg_unique_products": unique_products.mean(),
    }


def render_customer_summary_cards(summary: dict[str, Any]) -> None:
    total_customers, avg_orders, order_cap, basket_coverage, reorder_ratio = st.columns(5)
    total_customers.metric(translate("total_customers"), format_number(summary.get("total_customers", 0)))
    avg_orders.metric(translate("avg_orders"), format_number(summary.get("avg_total_orders", 0)))
    order_cap.metric(translate("order_cap_share"), format_percent(summary.get("order_cap_share", 0)))
    basket_coverage.metric(
        translate("avg_observed_coverage"),
        format_percent(summary.get("avg_observed_basket_coverage", 0)),
    )
    reorder_ratio.metric(translate("avg_reorder_ratio"), format_percent(summary.get("avg_reorder_ratio", 0)))


def get_row_value(row: pd.Series, column_name: str, default: Any = 0) -> Any:
    value = row.get(column_name, default)
    if pd.isna(value):
        return default
    return value


def render_segment_guide_cards() -> None:
    segment_rows = [
        ("limited_basket_detail", "limited_basket_detail_description"),
        ("loyal_reorder", "loyal_reorder_description"),
        ("at_risk_customer", "at_risk_customer_description"),
        ("product_explorer", "product_explorer_description"),
        ("steady_customer", "steady_customer_description"),
    ]

    st.markdown(f"**{translate('segment_guide')}**")
    for row_start in range(0, len(segment_rows), 2):
        row_segments = segment_rows[row_start : row_start + 2]
        columns = st.columns(len(row_segments))
        for column, (segment_key, description_key) in zip(columns, row_segments, strict=False):
            with column.container(border=True):
                st.markdown(f"**{translate(segment_key)}**")
                st.caption(translate(description_key))


def render_segment_metric_cards(segment_frame: pd.DataFrame) -> None:
    if segment_frame.empty:
        return

    st.markdown(f"**{translate('customer_segment_snapshot')}**")
    display_frame = segment_frame.sort_values("users", ascending=False)
    for _, row in display_frame.iterrows():
        with st.container(border=True):
            st.markdown(f"**{get_row_value(row, 'customer_segment_label', '')}**")
            st.caption(
                " | ".join(
                    [
                        f"{translate('users')}: {format_number(get_row_value(row, 'users'))}",
                        f"{translate('avg_orders')}: {format_number(get_row_value(row, 'avg_total_orders'))}",
                        f"{translate('avg_reorder_ratio')}: "
                        f"{format_percent(get_row_value(row, 'avg_reorder_ratio'))}",
                        f"{translate('avg_observed_coverage')}: "
                        f"{format_percent(get_row_value(row, 'avg_observed_basket_coverage'))}",
                    ]
                )
            )


def render_customer_segment_summary(segment_summary: pd.DataFrame) -> None:
    if segment_summary.empty:
        return

    segment_frame = segment_summary.copy()
    segment_frame["customer_segment_label"] = segment_frame["customer_segment"].map(translate_customer_segment)
    segment_frame = segment_frame.sort_values("users", ascending=True)

    st.markdown(f"**{translate('customer_segment_summary')}**")
    st.caption(translate("customer_segment_summary_caption"))
    st.info(f"{translate('behavior_segments_definition')} {translate('basket_detail_definition')}")
    render_segment_guide_cards()

    chart_column, table_column = st.columns([1.1, 1])
    with chart_column:
        chart = (
            alt.Chart(segment_frame)
            .mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4)
            .encode(
                y=alt.Y(
                    "customer_segment_label:N",
                    sort=None,
                    axis=alt.Axis(title=None, labelLimit=220),
                ),
                x=alt.X("users:Q", axis=alt.Axis(title=translate("users"))),
                color=alt.Color(
                    "customer_segment_label:N",
                    legend=None,
                    scale=alt.Scale(range=["#2563eb", "#38bdf8", "#16a34a", "#f59e0b", "#7c3aed"]),
                ),
                tooltip=[
                    alt.Tooltip("customer_segment_label:N", title=translate("customer_segment")),
                    alt.Tooltip("users:Q", title=translate("users"), format=",.0f"),
                    alt.Tooltip("avg_reorder_ratio:Q", title=translate("reorder_rate"), format=".1%"),
                    alt.Tooltip("avg_observed_basket_coverage:Q", title=translate("observed_basket_coverage"), format=".1%"),
                ],
            )
            .properties(height=320)
        )
        st.altair_chart(chart, use_container_width=True)
    with table_column:
        render_segment_metric_cards(segment_frame)


def render_customer_behavior_charts(frame: pd.DataFrame) -> None:
    chart_frame = frame.copy()
    chart_frame["customer_segment_label"] = chart_frame["customer_segment"].map(translate_customer_segment)
    numeric_columns = [
        "total_orders",
        "reorder_ratio",
        "unique_products",
        "observed_basket_coverage",
        "days_between_orders",
    ]
    for column_name in numeric_columns:
        chart_frame[column_name] = pd.to_numeric(chart_frame.get(column_name), errors="coerce")

    st.markdown(f"**{translate('customer_behavior_map')}**")
    st.caption(translate("customer_behavior_map_caption"))
    distribution_column, scatter_column = st.columns([0.9, 1.25])
    with distribution_column:
        distribution = (
            alt.Chart(chart_frame.dropna(subset=["total_orders"]))
            .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
            .encode(
                x=alt.X(
                    "total_orders:Q",
                    bin=alt.Bin(maxbins=18),
                    axis=alt.Axis(title=translate("orders"), labelAngle=0),
                ),
                y=alt.Y("count():Q", axis=alt.Axis(title=translate("users"))),
                tooltip=[
                    alt.Tooltip("count():Q", title=translate("users")),
                    alt.Tooltip("total_orders:Q", title=translate("orders"), bin=True),
                ],
            )
            .properties(title=translate("customer_distribution"), height=360)
        )
        st.altair_chart(distribution, use_container_width=True)
    with scatter_column:
        scatter = (
            alt.Chart(chart_frame.dropna(subset=["total_orders", "reorder_ratio"]))
            .mark_circle(opacity=0.75)
            .encode(
                x=alt.X(
                    "total_orders:Q",
                    axis=alt.Axis(title=translate("orders"), labelAngle=0),
                    scale=alt.Scale(zero=False),
                ),
                y=alt.Y(
                    "reorder_ratio:Q",
                    axis=alt.Axis(title=translate("reorder_rate"), format="%"),
                    scale=alt.Scale(domain=[0, 1]),
                ),
                size=alt.Size("unique_products:Q", title=translate("unique_products"), scale=alt.Scale(range=[45, 420])),
                color=alt.Color(
                    "customer_segment_label:N",
                    title=translate("customer_segment"),
                    scale=alt.Scale(range=["#2563eb", "#38bdf8", "#16a34a", "#f59e0b", "#7c3aed"]),
                ),
                tooltip=[
                    alt.Tooltip("user_id:N", title="User ID"),
                    alt.Tooltip("customer_segment_label:N", title=translate("customer_segment")),
                    alt.Tooltip("total_orders:Q", title=translate("orders"), format=",.0f"),
                    alt.Tooltip("reorder_ratio:Q", title=translate("reorder_rate"), format=".1%"),
                    alt.Tooltip("unique_products:Q", title=translate("unique_products"), format=",.0f"),
                    alt.Tooltip("observed_basket_coverage:Q", title=translate("observed_basket_coverage"), format=".1%"),
                ],
            )
            .properties(height=360)
        )
        st.altair_chart(scatter, use_container_width=True)


def render_representative_customer_cards(frame: pd.DataFrame) -> None:
    if frame.empty:
        return

    st.markdown(f"**{translate('sample_customer_profiles')}**")
    display_frame = frame.sort_values(["customer_segment_label", "total_orders"], ascending=[True, False]).head(12)
    for row_start in range(0, len(display_frame), 3):
        row_customers = display_frame.iloc[row_start : row_start + 3]
        columns = st.columns(len(row_customers))
        for column, (_, row) in zip(columns, row_customers.iterrows(), strict=False):
            with column.container(border=True):
                st.caption(get_row_value(row, "customer_segment_label", ""))
                st.markdown(f"**{translate('customer_id')} {get_row_value(row, 'user_id', '')}**")
                metric_columns = st.columns(2)
                metric_columns[0].metric(translate("orders"), format_number(get_row_value(row, "total_orders")))
                metric_columns[1].metric(
                    translate("avg_reorder_ratio"),
                    format_percent(get_row_value(row, "reorder_ratio")),
                )
                st.caption(
                    " | ".join(
                        [
                            f"{translate('unique_products')}: {format_number(get_row_value(row, 'unique_products'))}",
                            f"{translate('days_between_orders')}: {format_number(get_row_value(row, 'days_between_orders'))}",
                            f"{translate('avg_observed_coverage')}: "
                            f"{format_percent(get_row_value(row, 'observed_basket_coverage'))}",
                        ]
                    )
                )


def render_customer_insights(frame: pd.DataFrame) -> None:
    st.subheader(translate("customer_insights"))
    if frame.empty:
        st.info(translate("no_customer_data"))
        return

    st.caption(translate("customer_insights_caption"))
    st.info(translate("customer_data_context"))

    try:
        summary = load_customer_profile_summary()
    except Exception:
        logger.exception("Failed to load customer profile summary")
        summary = build_customer_summary_from_sample(frame)
    render_customer_summary_cards(summary or build_customer_summary_from_sample(frame))

    st.divider()

    try:
        segment_summary = load_customer_segment_summary()
    except Exception:
        logger.exception("Failed to load customer segment summary")
        segment_summary = pd.DataFrame()
    render_customer_segment_summary(segment_summary)
    render_customer_behavior_charts(frame)

    table_frame = frame.copy()
    table_frame["customer_segment_label"] = table_frame["customer_segment"].map(translate_customer_segment)
    render_representative_customer_cards(table_frame)


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
    if get_language() == "fr":
        labels = {
            "0": "Sans recommande",
            "1": "Avec recommande",
            "macro avg": "Moyenne macro",
            "weighted avg": "Moyenne ponderee",
        }
    else:
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
        top_feature = format_feature_label(feature_importance.iloc[0]["feature"])

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
        top_driver_label = "Top Driver" if get_language() == "en" else "Variable principale"
        st.caption(f"{top_driver_label}: {top_feature}")


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
    chart_frame["metric_label"] = chart_frame["metric"].map(metric_display_label)
    model_title = "Model" if get_language() == "en" else "Modele"
    metric_title = "Metric" if get_language() == "en" else "Metrique"
    score_title = "Score"
    chart = (
        alt.Chart(chart_frame)
        .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
        .encode(
            x=alt.X("model_name:N", axis=alt.Axis(title=None, labelAngle=-15), sort=None),
            y=alt.Y("score:Q", axis=alt.Axis(title="Score"), scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("metric_label:N", title=metric_title),
            xOffset="metric_label:N",
            tooltip=[
                alt.Tooltip("model_name:N", title=model_title),
                alt.Tooltip("metric_label:N", title=metric_title),
                alt.Tooltip("score:Q", title=score_title, format=".3f"),
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
            "model_name": model_title,
            "accuracy": st.column_config.ProgressColumn(translate("accuracy"), format="%.3f", min_value=0, max_value=1),
            "balanced_accuracy": st.column_config.ProgressColumn(translate("balanced_accuracy"), format="%.3f", min_value=0, max_value=1),
            "precision": st.column_config.ProgressColumn(translate("precision"), format="%.3f", min_value=0, max_value=1),
            "recall": st.column_config.ProgressColumn(translate("recall"), format="%.3f", min_value=0, max_value=1),
            "f1": st.column_config.ProgressColumn(translate("f1_score"), format="%.3f", min_value=0, max_value=1),
            "roc_auc": st.column_config.ProgressColumn(translate("roc_auc"), format="%.3f", min_value=0, max_value=1),
            "recommended_threshold": st.column_config.NumberColumn(translate("recommended_threshold"), format="%.2f"),
            "positive_prediction_rate": st.column_config.ProgressColumn(
                translate("positive_prediction_rate"),
                format="%.3f",
                min_value=0,
                max_value=1,
            ),
            "average_precision": st.column_config.ProgressColumn(translate("average_precision"), format="%.3f", min_value=0, max_value=1),
            "brier_score": st.column_config.NumberColumn(translate("brier_score"), format="%.3f"),
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


def render_static_table(frame: pd.DataFrame, *, show_index: bool = False) -> None:
    if frame.empty:
        return

    table_html = frame.to_html(index=show_index, escape=True, border=0)
    html_block = (
        "<style>"
        ".responsive-static-table table{width:100%;table-layout:fixed;border-collapse:collapse;font-size:0.92rem;}"
        ".responsive-static-table th,.responsive-static-table td{border:1px solid #e5e7eb;padding:0.65rem 0.75rem;"
        "text-align:left;vertical-align:top;white-space:normal;overflow-wrap:anywhere;word-break:break-word;}"
        ".responsive-static-table th{background:#f9fafb;color:#6b7280;font-weight:500;}"
        ".responsive-static-table tr:nth-child(even) td{background:#fcfcfd;}"
        "</style>"
        f'<div class="responsive-static-table">{table_html}</div>'
    )
    st.markdown(
        html_block,
        unsafe_allow_html=True,
    )


def format_static_metric(value: Any, *, decimals: int = 3) -> str:
    try:
        return f"{float(value):.{decimals}f}"
    except (TypeError, ValueError):
        return "0"


def render_classification_report_visuals(report_frame: pd.DataFrame) -> None:
    class_rows = report_frame[report_frame["class"].isin(["No Reorder", "Reorder", "Sans recommande", "Avec recommande"])]
    if class_rows.empty:
        class_rows = report_frame.head(2)

    columns = st.columns(len(class_rows)) if len(class_rows) > 1 else [st]
    for column, (_, row) in zip(columns, class_rows.iterrows(), strict=False):
        with column:
            with st.container(border=True):
                st.caption(str(row["class"]))
                score_columns = st.columns(3)
                score_columns[0].metric(translate("precision"), format_percent(row["precision"]))
                score_columns[1].metric(translate("recall"), format_percent(row["recall"]))
                score_columns[2].metric("F1", format_percent(row["f1_score"]))
                st.caption(
                    ("Support: " if get_language() == "en" else "Volume: ")
                    + format_number(row["support"])
                )

    chart_frame = report_frame.melt(
        id_vars=["class"],
        value_vars=["precision", "recall", "f1_score"],
        var_name="metric",
        value_name="score",
    )
    metric_labels = {
        "precision": translate("precision"),
        "recall": translate("recall"),
        "f1_score": "F1",
    }
    chart_frame["metric_label"] = chart_frame["metric"].map(metric_labels)
    chart = (
        alt.Chart(chart_frame)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X("class:N", axis=alt.Axis(title=None, labelAngle=-15, labelLimit=150)),
            xOffset=alt.XOffset("metric_label:N"),
            y=alt.Y("score:Q", axis=alt.Axis(title="Score", format="%"), scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("metric_label:N", title="Metric" if get_language() == "en" else "Metrique"),
            tooltip=[
                alt.Tooltip("class:N", title="Class" if get_language() == "en" else "Classe"),
                alt.Tooltip("metric_label:N", title="Metric" if get_language() == "en" else "Metrique"),
                alt.Tooltip("score:Q", title="Score", format=".1%"),
            ],
        )
        .properties(height=280)
    )
    st.altair_chart(chart, use_container_width=True)


def render_feature_driver_cards(feature_importance: pd.DataFrame) -> None:
    top_features = feature_importance.head(3).copy()
    if top_features.empty:
        return

    columns = st.columns(len(top_features))
    for rank, (column, (_, row)) in enumerate(zip(columns, top_features.iterrows(), strict=False), start=1):
        with column:
            with st.container(border=True):
                st.caption(f"#{rank}")
                st.markdown(f"**{format_feature_label(row['feature'])}**")
                st.metric("Importance", f"{float(row['importance']):.3f}")

    top_share = float(top_features["importance"].sum())
    explanation = (
        f"The top 3 drivers explain {top_share:.1%} of the model signal. "
        "If this concentration is high, monitor drift and add broader customer/product features."
        if get_language() == "en"
        else f"Les 3 premiers signaux representent {top_share:.1%} de l'importance du modele. "
        "Si cette concentration est elevee, surveille le drift et ajoute des features client/produit plus riches."
    )
    st.caption(explanation)


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
            index=(
                ["Actual No Reorder", "Actual Reorder"]
                if get_language() == "en"
                else ["Reel sans recommande", "Reel avec recommande"]
            ),
            columns=(
                ["Predicted No Reorder", "Predicted Reorder"]
                if get_language() == "en"
                else ["Predit sans recommande", "Predit avec recommande"]
            ),
        )
        with st.expander(translate("confusion_matrix")):
            confusion_display = confusion_frame.reset_index()
            confusion_display.columns = (
                ["Actual", "Predicted No Reorder", "Predicted Reorder"]
                if get_language() == "en"
                else ["Reel", "Predit sans recommande", "Predit avec recommande"]
            )
            render_static_table(confusion_display)

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
        st.caption(
            "The cards compare how the model behaves on each class; the table keeps the technical details."
            if get_language() == "en"
            else "Les cartes comparent le comportement du modele par classe; le tableau garde le detail technique."
        )
        render_classification_report_visuals(report_frame)
        report_display = report_frame.rename(
            columns={
                "class": "Class" if get_language() == "en" else "Classe",
                "precision": translate("precision"),
                "recall": translate("recall"),
                "f1_score": "F1",
                "support": "Support" if get_language() == "en" else "Volume",
            }
        )
        for column in [translate("precision"), translate("recall"), "F1"]:
            if column in report_display:
                report_display[column] = report_display[column].map(format_static_metric)
        support_column = "Support" if get_language() == "en" else "Volume"
        if support_column in report_display:
            report_display[support_column] = report_display[support_column].map(format_number)
        with st.expander("Technical report table" if get_language() == "en" else "Table technique du rapport"):
            render_static_table(report_display)

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
        render_feature_driver_cards(feature_importance)
        chart_data = feature_importance[["feature", "importance"]].copy()
        chart_data["feature_label"] = chart_data["feature"].map(format_feature_label)
        visible_chart_data = chart_data.head(12)
        render_bar_chart(
            visible_chart_data,
            "feature_label",
            "importance",
            y_title="Importance",
            x_tick_angle=-15,
            height=390,
            label_format=".3f",
        )
        table_data = chart_data.rename(
            columns={
                "feature_label": "feature_display",
                "feature": "technical_feature",
            }
        )
        importance_display = table_data[["feature_display", "technical_feature", "importance"]].rename(
            columns={
                "feature_display": "Feature" if get_language() == "en" else "Feature affichee",
                "technical_feature": "Technical name" if get_language() == "en" else "Nom technique",
                "importance": "Importance",
            }
        )
        importance_display["Importance"] = importance_display["Importance"].map(format_static_metric)
        with st.expander("Technical feature table" if get_language() == "en" else "Table technique des features"):
            render_static_table(importance_display)


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


def build_decision_payload(kpis: dict[str, Any], observed_signals: dict[str, Any]) -> dict[str, Any]:
    return {
        "reorder_rate": float(kpis.get("reorder_rate") or 0),
        "churn_rate": 0.50,
        "days_between_orders": 18.0,
        "data_quality_failures": int(observed_signals.get("data_quality_failures") or 0),
        "api_health": observed_signals.get("api_health", "failed"),
    }


def render_decision_engine(kpis: dict[str, Any]) -> None:
    st.subheader(translate("decision_engine"))
    st.caption(translate("decision_engine_caption"))

    render_decision_engine_explainer()
    observed_signals = load_decision_observed_signals()
    default_payload = build_decision_payload(kpis, observed_signals)

    with st.form("decision_engine_form"):
        st.markdown(f"**{translate('business_scenario_levers')}**")
        st.caption(translate("business_scenario_caption"))
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
        with metric_columns[1]:
            churn_rate = st.slider(
                translate("scenario_churn_rate"),
                min_value=0.0,
                max_value=1.0,
                value=0.50,
                step=0.01,
                format="%.2f",
            )
        with metric_columns[2]:
            days_between_orders = st.slider(
                translate("scenario_days_between_orders"),
                min_value=0.0,
                max_value=60.0,
                value=18.0,
                step=1.0,
            )
            use_gemini = st.checkbox(translate("use_gemini"), value=False)

        render_detected_operational_signals(observed_signals)
        submitted = st.form_submit_button(translate("run_decision_engine"), use_container_width=True)

    if not submitted and "decision_engine_response" not in st.session_state:
        return

    if submitted:
        payload = {
            "data": {
                "reorder_rate": reorder_rate,
                "churn_rate": churn_rate,
                "days_between_orders": days_between_orders,
                "data_quality_failures": observed_signals.get("data_quality_failures", 0),
                "api_health": observed_signals.get("api_health", "failed"),
            },
            "anomalies": [],
            "use_gemini": use_gemini,
            "language": get_language(),
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
    decisions = [translate_decision_record(decision) for decision in response.get("decisions") or []]
    alerts = [decision for decision in decisions if decision.get("decision_type") == "alert"]
    recommendations = [decision for decision in decisions if decision.get("decision_type") == "recommendation"]
    anomalies = [translate_decision_record(anomaly) for anomaly in response.get("anomalies") or []]
    explanation_source = translate_decision_output_text(response.get("explanation_source", "local_fallback"))

    st.divider()
    summary_columns = st.columns(4)
    summary_columns[0].metric(translate("detected_anomalies"), format_number(len(anomalies)))
    summary_columns[1].metric(translate("priority_alerts"), format_number(len(alerts)))
    summary_columns[2].metric(translate("decision_recommendations"), format_number(len(recommendations)))
    summary_columns[3].metric(translate("explanation_source"), explanation_source)

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
                    translate("explanation_source"): explanation_source,
                }
            ]
        )
        st.dataframe(status_frame, use_container_width=True, hide_index=True)
        if explanation_detail:
            st.caption(translate_decision_output_text(explanation_detail))
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
