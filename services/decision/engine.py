import logging
from typing import Any, Literal

import requests
from pydantic import BaseModel, Field

from config.settings import Settings, get_settings

logger = logging.getLogger(__name__)

DecisionPriority = Literal["low", "medium", "high", "critical"]
DecisionType = Literal["alert", "recommendation"]
DecisionLanguage = Literal["en", "fr"]


class Anomaly(BaseModel):
    metric: str
    value: float | str
    severity: DecisionPriority = "medium"
    description: str | None = None


class DecisionRequest(BaseModel):
    data: dict[str, Any] = Field(default_factory=dict)
    anomalies: list[Anomaly] = Field(default_factory=list)
    use_gemini: bool = True
    language: DecisionLanguage = "en"


class Decision(BaseModel):
    decision_type: DecisionType
    priority: DecisionPriority
    title: str
    action: str
    rationale: str
    metric: str | None = None
    evidence: dict[str, Any] = Field(default_factory=dict)


class DecisionResponse(BaseModel):
    decisions: list[Decision]
    explanation: str
    explanation_source: Literal["gemini", "local_fallback"]
    explanation_detail: str | None = None
    gemini_requested: bool = False
    gemini_configured: bool = False
    anomalies: list[Anomaly]


PRIORITY_SCORE = {"low": 1, "medium": 2, "high": 3, "critical": 4}
PRIORITY_LABELS_FR = {
    "low": "faible",
    "medium": "moyenne",
    "high": "elevee",
    "critical": "critique",
}

DECISION_TEXT_FR = {
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

DETAIL_TEXT_FR = {
    "Gemini was not requested for this run.": "Gemini n'a pas ete demande pour cette execution.",
    "GEMINI_API_KEY is not configured on the API service.": (
        "GEMINI_API_KEY n'est pas configuree sur le service API."
    ),
    "Gemini returned no text in the response.": "Gemini n'a retourne aucun texte.",
    "Gemini returned an unexpected response shape.": "Gemini a retourne une reponse au format inattendu.",
    "Gemini was requested but did not return text.": "Gemini a ete demande, mais n'a retourne aucun texte.",
}


def normalize_priority(*priorities: DecisionPriority) -> DecisionPriority:
    return max(priorities or ("low",), key=lambda priority: PRIORITY_SCORE[priority])


def translate_decision_text(value: str | None, language: DecisionLanguage) -> str | None:
    if value is None or language != "fr":
        return value
    return DECISION_TEXT_FR.get(value, value)


def translate_detail_text(value: str | None, language: DecisionLanguage) -> str | None:
    if value is None or language != "fr":
        return value
    if value.startswith("Generated with "):
        model_name = value.removeprefix("Generated with ").removesuffix(".")
        return f"Genere avec {model_name}."
    if value.startswith("Gemini request failed with HTTP status "):
        status_code = value.removeprefix("Gemini request failed with HTTP status ").removesuffix(".")
        return f"La requete Gemini a echoue avec le statut HTTP {status_code}."
    if value.startswith("Gemini request failed: "):
        error_name = value.removeprefix("Gemini request failed: ").removesuffix(".")
        return f"La requete Gemini a echoue: {error_name}."
    return DETAIL_TEXT_FR.get(value, value)


def localize_anomalies(anomalies: list[Anomaly], language: DecisionLanguage) -> list[Anomaly]:
    if language != "fr":
        return anomalies
    return [
        anomaly.model_copy(update={"description": translate_decision_text(anomaly.description, language)})
        for anomaly in anomalies
    ]


def localize_decisions(decisions: list[Decision], language: DecisionLanguage) -> list[Decision]:
    if language != "fr":
        return decisions

    localized_decisions: list[Decision] = []
    for decision in decisions:
        evidence = dict(decision.evidence)
        if "description" in evidence:
            evidence["description"] = translate_decision_text(str(evidence["description"]), language)
        localized_decisions.append(
            decision.model_copy(
                update={
                    "title": translate_decision_text(decision.title, language),
                    "action": translate_decision_text(decision.action, language),
                    "rationale": translate_decision_text(decision.rationale, language),
                    "evidence": evidence,
                }
            )
        )
    return localized_decisions


def as_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, str):
        value = value.strip().replace(",", "")
        if value.endswith("%"):
            value = float(value[:-1]) / 100
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def detect_anomalies(data: dict[str, Any]) -> list[Anomaly]:
    anomalies: list[Anomaly] = []

    reorder_rate = as_float(data.get("reorder_rate"))
    if reorder_rate and reorder_rate < 0.35:
        anomalies.append(
            Anomaly(
                metric="reorder_rate",
                value=reorder_rate,
                severity="high",
                description="Reorder rate is below the healthy operating threshold.",
            )
        )

    churn_rate = as_float(data.get("churn_rate") or data.get("avg_churn_probability"))
    if churn_rate >= 0.70:
        anomalies.append(
            Anomaly(
                metric="churn_rate",
                value=churn_rate,
                severity="critical",
                description="Churn risk is critically high.",
            )
        )
    elif churn_rate >= 0.50:
        anomalies.append(
            Anomaly(
                metric="churn_rate",
                value=churn_rate,
                severity="high",
                description="Churn risk is elevated.",
            )
        )

    days_between_orders = as_float(data.get("days_between_orders") or data.get("avg_days_between_orders"))
    if days_between_orders >= 21:
        anomalies.append(
            Anomaly(
                metric="days_between_orders",
                value=days_between_orders,
                severity="high",
                description="Average purchase gap is unusually long.",
            )
        )

    data_quality_failures = as_float(data.get("data_quality_failures") or data.get("failed_checks"))
    if data_quality_failures > 0:
        anomalies.append(
            Anomaly(
                metric="data_quality_failures",
                value=data_quality_failures,
                severity="critical",
                description="Data quality checks are failing before downstream decisions.",
            )
        )

    api_health = str(data.get("api_health") or data.get("render_api_status") or "").lower()
    if api_health in {"failed", "down", "unhealthy"}:
        anomalies.append(
            Anomaly(
                metric="api_health",
                value=api_health,
                severity="critical",
                description="Serving API is unavailable or unhealthy.",
            )
        )

    return anomalies


def build_decisions(data: dict[str, Any], anomalies: list[Anomaly]) -> list[Decision]:
    decisions: list[Decision] = []
    anomaly_by_metric = {anomaly.metric: anomaly for anomaly in anomalies}

    if anomaly := anomaly_by_metric.get("data_quality_failures"):
        decisions.append(
            Decision(
                decision_type="alert",
                priority="critical",
                title="Pause downstream analytics refresh",
                action="Investigate failing data quality checks before running dbt, training, or dashboard refreshes.",
                rationale="Data quality failures can contaminate marts, model features, and AI insights.",
                metric=anomaly.metric,
                evidence={"value": anomaly.value, "description": anomaly.description},
            )
        )

    if anomaly := anomaly_by_metric.get("api_health"):
        decisions.append(
            Decision(
                decision_type="alert",
                priority="critical",
                title="Restore prediction API availability",
                action="Check Render logs, health endpoint, model artifact availability, and environment variables.",
                rationale="The decision and dashboard layers depend on the API for live predictions and copilot workflows.",
                metric=anomaly.metric,
                evidence={"value": anomaly.value, "description": anomaly.description},
            )
        )

    churn_anomaly = anomaly_by_metric.get("churn_rate")
    reorder_anomaly = anomaly_by_metric.get("reorder_rate")
    gap_anomaly = anomaly_by_metric.get("days_between_orders")
    if churn_anomaly or reorder_anomaly or gap_anomaly:
        priority = normalize_priority(
            churn_anomaly.severity if churn_anomaly else "low",
            reorder_anomaly.severity if reorder_anomaly else "low",
            gap_anomaly.severity if gap_anomaly else "low",
        )
        decisions.append(
            Decision(
                decision_type="recommendation",
                priority=priority,
                title="Launch retention intervention",
                action="Target high-risk customers with reorder reminders, high-repeat products, and time-bound offers.",
                rationale="Churn risk, weak reorder behavior, or long purchase gaps indicate declining customer engagement.",
                metric="retention_risk",
                evidence={
                    "churn_rate": data.get("churn_rate") or data.get("avg_churn_probability"),
                    "reorder_rate": data.get("reorder_rate"),
                    "days_between_orders": data.get("days_between_orders") or data.get("avg_days_between_orders"),
                },
            )
        )

    revenue_delta = as_float(data.get("revenue_delta"))
    if revenue_delta < -0.10:
        decisions.append(
            Decision(
                decision_type="alert",
                priority="high",
                title="Review revenue proxy decline",
                action="Compare order volume, basket size, and repeat purchase movement by cohort and product department.",
                rationale="A declining revenue proxy can signal lower basket activity even without price-level data.",
                metric="revenue_delta",
                evidence={"value": revenue_delta},
            )
        )

    if not decisions:
        decisions.append(
            Decision(
                decision_type="recommendation",
                priority="low",
                title="Continue monitoring customer health",
                action="Track reorder rate, retention cohorts, churn risk, and data quality daily.",
                rationale="No critical anomaly was detected, so the best decision is ongoing monitoring.",
                metric="overall_health",
                evidence={"observed_metrics": list(data.keys())},
            )
        )

    return sorted(decisions, key=lambda decision: PRIORITY_SCORE[decision.priority], reverse=True)


def build_local_explanation(
    decisions: list[Decision],
    anomalies: list[Anomaly],
    language: DecisionLanguage = "en",
) -> str:
    highest_priority = decisions[0].priority if decisions else "low"
    anomaly_count = len(anomalies)
    top_decision = decisions[0].title if decisions else "Continue monitoring"
    if language == "fr":
        priority_label = PRIORITY_LABELS_FR.get(highest_priority, highest_priority)
        return (
            f"Le moteur de decision a produit {len(decisions)} decision(s) a partir de "
            f"{anomaly_count} signal(aux) d'anomalie. La priorite la plus elevee est "
            f"{priority_label}. Premiere action recommandee: {top_decision}."
        )
    return (
        f"Decision engine produced {len(decisions)} decision(s) from {anomaly_count} anomaly signal(s). "
        f"The highest priority is {highest_priority}. Recommended first action: {top_decision}."
    )


def build_gemini_prompt(
    data: dict[str, Any],
    anomalies: list[Anomaly],
    decisions: list[Decision],
    language: DecisionLanguage = "en",
) -> str:
    language_instruction = (
        "Respond in French. Keep metric names such as reorder_rate, churn_rate, and days_between_orders unchanged. "
        "Translate business explanations, recommendations, and actions into French."
        if language == "fr"
        else "Respond in English."
    )
    return (
        "You are a decision engine for a customer intelligence data platform. "
        "Explain the decisions concisely for a business stakeholder. Use only the supplied data. "
        "Mention the highest priority, why it matters, and what should happen next.\n\n"
        f"{language_instruction}\n\n"
        f"Data: {data}\n"
        f"Anomalies: {[anomaly.model_dump() for anomaly in anomalies]}\n"
        f"Decisions: {[decision.model_dump() for decision in decisions]}"
    )


def explain_with_gemini(
    data: dict[str, Any],
    anomalies: list[Anomaly],
    decisions: list[Decision],
    settings: Settings,
    language: DecisionLanguage = "en",
) -> tuple[str | None, str | None]:
    if not settings.gemini_api_key:
        return None, translate_detail_text("GEMINI_API_KEY is not configured on the API service.", language)

    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{settings.gemini_model}:generateContent"
    )
    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": build_gemini_prompt(
                            data=data,
                            anomalies=anomalies,
                            decisions=decisions,
                            language=language,
                        ),
                    }
                ]
            }
        ]
    }

    try:
        response = requests.post(
            url,
            params={"key": settings.gemini_api_key},
            json=payload,
            timeout=20,
        )
        response.raise_for_status()
        body = response.json()
        candidates = body.get("candidates") or []
        parts = candidates[0].get("content", {}).get("parts", []) if candidates else []
        if parts and parts[0].get("text"):
            return str(parts[0]["text"]).strip(), translate_detail_text(
                f"Generated with {settings.gemini_model}.",
                language,
            )
        return None, translate_detail_text("Gemini returned no text in the response.", language)
    except requests.HTTPError as exc:
        status_code = exc.response.status_code if exc.response is not None else "unknown"
        logger.exception("Gemini explanation request failed")
        return None, translate_detail_text(f"Gemini request failed with HTTP status {status_code}.", language)
    except requests.RequestException as exc:
        logger.exception("Gemini explanation request failed")
        return None, translate_detail_text(f"Gemini request failed: {exc.__class__.__name__}.", language)
    except (KeyError, IndexError, TypeError):
        logger.exception("Gemini response shape was unexpected")
        return None, translate_detail_text("Gemini returned an unexpected response shape.", language)

    return None, translate_detail_text("Gemini was requested but did not return text.", language)


def run_decision_engine(payload: DecisionRequest, settings: Settings | None = None) -> DecisionResponse:
    runtime_settings = settings or get_settings()
    detected_anomalies = detect_anomalies(payload.data)
    anomalies = payload.anomalies + detected_anomalies
    decisions = build_decisions(payload.data, anomalies)
    localized_anomalies = localize_anomalies(anomalies, payload.language)
    localized_decisions = localize_decisions(decisions, payload.language)

    explanation = None
    explanation_detail = translate_detail_text("Gemini was not requested for this run.", payload.language)
    explanation_source: Literal["gemini", "local_fallback"] = "local_fallback"
    gemini_configured = bool(runtime_settings.gemini_api_key)
    if payload.use_gemini:
        explanation, explanation_detail = explain_with_gemini(
            payload.data,
            localized_anomalies,
            localized_decisions,
            runtime_settings,
            language=payload.language,
        )
        if explanation:
            explanation_source = "gemini"

    return DecisionResponse(
        decisions=localized_decisions,
        explanation=explanation
        or build_local_explanation(localized_decisions, localized_anomalies, payload.language),
        explanation_source=explanation_source,
        explanation_detail=explanation_detail,
        gemini_requested=payload.use_gemini,
        gemini_configured=gemini_configured,
        anomalies=localized_anomalies,
    )
