# Customer Reorder Model Card

## Purpose

This model estimates whether a customer order is likely to contain at least one reordered item. It is designed for decision support in a portfolio data platform, not for fully automated customer targeting.

## Prediction Target

The target is `will_reorder`:

- `1`: the evaluated order contains at least one reordered item
- `0`: the evaluated order does not contain reordered items

Features are calculated from prior customer behavior before the evaluated order. This reduces direct leakage from the order being predicted.

## Data Sources

The training dataset is built from Snowflake models created with dbt:

- `fct_orders`
- `stg_order_products`
- `dim_products`
- `feature_store`

Raw inputs come from the Instacart dataset:

- orders
- order products
- products
- aisles
- departments

## Feature Groups

| Group | Examples | Business Meaning |
| --- | --- | --- |
| Order history | `total_orders`, `customer_tenure_days`, `order_frequency_30d` | How often and how long a customer has ordered |
| Recency and gaps | `days_since_last_order`, `days_between_orders`, `stddev_days_between_orders` | Whether purchase rhythm is stable or slowing |
| Time behavior | `avg_order_dow`, `avg_order_hour_of_day`, `weekend_order_ratio`, `evening_order_ratio` | When the customer tends to shop |
| Basket coverage | `observed_basket_orders`, `avg_basket_size` | How much item-level basket data is available |
| Product mix | `unique_products`, `unique_departments`, `unique_aisles`, `produce_item_ratio`, `fresh_fruits_item_ratio` | Breadth and category affinity |

## Current Model

The training pipeline compares multiple candidate models and selects the strongest model by ranking metrics such as ROC AUC, average precision, and calibration.

Current selected model:

```text
Gradient Boosting
```

The deployed artifact path remains:

```text
ml/artifacts/random_forest_reorder_model.joblib
```

The filename is kept for API compatibility, even though the training metadata identifies the selected candidate model.

## Validation Metrics

| Metric | Value |
| --- | ---: |
| Accuracy | 0.544 |
| Balanced accuracy | 0.674 |
| Precision | 0.977 |
| Recall | 0.525 |
| F1 score | 0.683 |
| ROC AUC | 0.741 |
| Average precision | 0.974 |
| Brier score | 0.058 |
| Positive rate | 93.4% |
| Train rows | 104,967 |
| Test rows | 26,242 |
| Recommended threshold | 0.95 |

The dataset is highly imbalanced toward reorder-positive examples. For this reason, accuracy alone is not a good quality measure. Balanced accuracy, ROC AUC, precision, recall, and threshold behavior are more informative.

## Model Comparison

| Model | Accuracy | Balanced Accuracy | Precision | Recall | F1 | ROC AUC | Threshold |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Gradient Boosting | 0.544 | 0.674 | 0.977 | 0.525 | 0.683 | 0.741 | 0.95 |
| Random Forest | 0.630 | 0.677 | 0.971 | 0.622 | 0.758 | 0.729 | 0.60 |
| Logistic Regression | 0.626 | 0.663 | 0.968 | 0.621 | 0.756 | 0.715 | 0.50 |

## Confusion Matrix

| Actual / Predicted | Predicted No Reorder | Predicted Reorder |
| --- | ---: | ---: |
| Actual No Reorder | 1,416 | 304 |
| Actual Reorder | 11,650 | 12,872 |

## Interpretation

The model is conservative at the selected threshold. It has very high precision, meaning that customers predicted to reorder are usually true reorderers. Recall is lower, meaning the model intentionally misses some reorder-positive customers to reduce false positives.

This is useful for campaigns where false positives are costly. If the business goal is broader reach, the threshold should be lowered and monitored.

## Known Limitations

- The Instacart dataset does not include prices, promotions, inventory, or customer demographics.
- Revenue is a proxy based on order volume, not actual sales.
- Product and aisle features currently add limited predictive signal because item-level coverage is incomplete for many orders.
- The model is trained offline and should not be treated as real-time customer intent.
- A production deployment should store artifacts in S3, Snowflake stage, or a model registry instead of Git.

## Recommended Next Improvements

- Add richer customer features such as recency bands, reorder streaks, and category preferences.
- Add product and department affinity features once basket item coverage is improved.
- Track model drift and feature drift over time.
- Add SHAP or permutation importance for more transparent explanations.
- Store model versions and metrics in a model registry.
- Tune thresholds by campaign objective: retention, replenishment, win-back, or product discovery.
