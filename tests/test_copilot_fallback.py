from api.copilot import classify_question, generate_local_insights


def test_retention_question_uses_retention_fallback() -> None:
    context = {
        "cohort_retention": [
            {
                "cohort_order_number": 10,
                "cohort_period": 2,
                "cohort_size": 100,
                "active_users": 62,
                "retention_rate": 0.62,
                "revenue_proxy": 500,
            },
            {
                "cohort_order_number": 9,
                "cohort_period": 3,
                "cohort_size": 100,
                "active_users": 41,
                "retention_rate": 0.41,
                "revenue_proxy": 300,
            },
        ]
    }

    category = classify_question("How does retention change by cohort period?")
    response = generate_local_insights("How does retention change by cohort period?", context, category)

    assert category == "retention"
    assert response.insights[0].category == "retention"
    assert "retention" in response.summary.lower()


def test_top_products_question_uses_product_fallback() -> None:
    context = {
        "top_products": [
            {
                "product_name": "Banana",
                "department": "produce",
                "aisle": "fresh fruits",
                "order_line_count": 1000,
                "reordered_line_count": 700,
                "reorder_ratio": 0.70,
            },
            {
                "product_name": "Organic Milk",
                "department": "dairy",
                "aisle": "milk",
                "order_line_count": 600,
                "reordered_line_count": 540,
                "reorder_ratio": 0.90,
            },
        ]
    }

    category = classify_question("Which products are most associated with repeat purchases?")
    response = generate_local_insights("Which products are most associated with repeat purchases?", context, category)

    assert category == "top_products"
    assert response.insights[0].category == "top_products"
    assert "Organic Milk" in response.summary
