from config.settings import Settings
from services.decision.engine import DecisionRequest, build_decisions, detect_anomalies, run_decision_engine


def test_detect_anomalies_from_customer_metrics() -> None:
    anomalies = detect_anomalies(
        {
            "reorder_rate": 0.22,
            "churn_rate": 0.72,
            "days_between_orders": 28,
        }
    )

    metrics = {anomaly.metric for anomaly in anomalies}

    assert "reorder_rate" in metrics
    assert "churn_rate" in metrics
    assert "days_between_orders" in metrics


def test_build_decisions_prioritizes_data_quality() -> None:
    anomalies = detect_anomalies({"data_quality_failures": 2, "api_health": "ok"})
    decisions = build_decisions({"data_quality_failures": 2}, anomalies)

    assert decisions[0].priority == "critical"
    assert decisions[0].decision_type == "alert"
    assert "data quality" in decisions[0].rationale.lower()


def test_decision_engine_uses_local_fallback_without_gemini() -> None:
    response = run_decision_engine(
        DecisionRequest(
            data={
                "reorder_rate": 0.20,
                "churn_rate": 0.60,
                "days_between_orders": 25,
            },
            use_gemini=False,
        ),
        settings=Settings(gemini_api_key=None),
    )

    assert response.explanation_source == "local_fallback"
    assert response.gemini_requested is False
    assert response.gemini_configured is False
    assert response.explanation_detail == "Gemini was not requested for this run."
    assert response.decisions[0].priority == "high"
    assert response.anomalies


def test_decision_engine_explains_missing_gemini_key() -> None:
    response = run_decision_engine(
        DecisionRequest(
            data={
                "reorder_rate": 0.20,
                "churn_rate": 0.60,
                "days_between_orders": 25,
            },
            use_gemini=True,
        ),
        settings=Settings(gemini_api_key=None),
    )

    assert response.explanation_source == "local_fallback"
    assert response.gemini_requested is True
    assert response.gemini_configured is False
    assert response.explanation_detail is not None
    assert "GEMINI_API_KEY" in response.explanation_detail


def test_decision_engine_localizes_french_response() -> None:
    response = run_decision_engine(
        DecisionRequest(
            data={
                "reorder_rate": 0.20,
                "churn_rate": 0.60,
                "days_between_orders": 25,
            },
            use_gemini=False,
            language="fr",
        ),
        settings=Settings(gemini_api_key=None),
    )

    assert response.explanation_source == "local_fallback"
    assert "moteur de decision" in response.explanation.lower()
    assert response.decisions[0].title == "Lancer une intervention de retention"
    assert "Cibler les clients a risque" in response.decisions[0].action
    assert response.anomalies[0].description is not None
    assert "Reorder rate is below" not in response.anomalies[0].description
