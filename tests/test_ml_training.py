import pandas as pd

from ml.train_model import build_training_matrix


def test_build_training_matrix_uses_ordered_feature_columns() -> None:
    frame = pd.DataFrame(
        {
            "user_id": [1, 2, 3, 4],
            "order_id": [11, 12, 13, 14],
            "order_number": [2, 2, 3, 3],
            "total_orders": [1, 1, 2, 2],
            "avg_basket_size": [5.0, 3.0, 7.0, 4.0],
            "reorder_ratio": [0.2, 0.0, 0.6, 0.1],
            "unique_products": [5, 3, 12, 8],
            "days_between_orders": [7.0, 14.0, 6.0, 10.0],
            "will_reorder": [1, 0, 1, 0],
        }
    )

    features, target = build_training_matrix(frame)

    assert list(features.columns) == [
        "total_orders",
        "avg_basket_size",
        "reorder_ratio",
        "unique_products",
        "days_between_orders",
    ]
    assert "will_reorder" not in features.columns
    assert target.tolist() == [1, 0, 1, 0]
