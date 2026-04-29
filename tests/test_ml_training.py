import pandas as pd

from ml.features import FEATURE_COLUMNS
from ml.train_model import build_training_matrix


def test_build_training_matrix_uses_ordered_feature_columns() -> None:
    frame = pd.DataFrame(
        {
            "user_id": [1, 2, 3, 4],
            "order_id": [11, 12, 13, 14],
            "order_number": [2, 2, 3, 3],
            "total_orders": [1, 1, 2, 2],
            "observed_basket_orders": [1, 1, 2, 2],
            "avg_basket_size": [5.0, 3.0, 7.0, 4.0],
            "reorder_ratio": [0.2, 0.0, 0.6, 0.1],
            "days_between_orders": [7.0, 14.0, 6.0, 10.0],
            "stddev_days_between_orders": [0.0, 0.0, 1.5, 2.0],
            "days_since_last_order": [7.0, 14.0, 4.0, 9.0],
            "customer_tenure_days": [7.0, 14.0, 13.0, 24.0],
            "order_frequency_30d": [1.0, 1.0, 2.0, 2.0],
            "avg_order_dow": [2.0, 4.0, 3.0, 5.0],
            "avg_order_hour_of_day": [10.0, 14.0, 11.0, 16.0],
            "weekend_order_ratio": [0.0, 0.5, 0.0, 0.5],
            "evening_order_ratio": [0.0, 0.0, 0.5, 0.0],
            "prior_reorder_order_ratio": [0.0, 0.0, 0.5, 0.5],
            "will_reorder": [1, 0, 1, 0],
        }
    )

    features, target = build_training_matrix(frame)

    assert list(features.columns) == FEATURE_COLUMNS
    assert "will_reorder" not in features.columns
    assert target.tolist() == [1, 0, 1, 0]
