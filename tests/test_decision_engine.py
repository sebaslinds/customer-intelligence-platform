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
        )
    )

    assert response.explanation_source == "local_fallback"
    assert response.decisions[0].priority == "high"
    assert response.anomalies
