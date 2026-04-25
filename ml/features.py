import pandas as pd


def build_customer_features(customers: pd.DataFrame) -> pd.DataFrame:
    features = customers.copy()
    if "total_spend" in features and "order_count" in features:
        features["average_order_value"] = features["total_spend"] / features["order_count"].clip(lower=1)
    return features
