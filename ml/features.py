from textwrap import dedent

import pandas as pd

FEATURE_COLUMNS = [
    "total_orders",
    "avg_basket_size",
    "reorder_ratio",
    "unique_products",
    "days_between_orders",
]

TARGET_COLUMN = "will_reorder"


def build_reorder_training_query(source_relation: str = "fct_orders", row_limit: int | None = 250_000) -> str:
    """Build a time-aware training dataset from order history.

    Each row represents a target order from the train split. Features are
    computed only from orders that happened before that target order, while the
    target indicates whether the target order contains at least one reordered
    item.
    """
    limit_clause = f"\nlimit {row_limit}" if row_limit else ""

    return dedent(
        f"""
        with ordered_history as (
            select
                user_id,
                order_id,
                order_number,
                count(*) over (
                    partition by user_id
                    order by order_number
                    rows between unbounded preceding and 1 preceding
                ) as total_orders,
                avg(item_count) over (
                    partition by user_id
                    order by order_number
                    rows between unbounded preceding and 1 preceding
                ) as avg_basket_size,
                coalesce(
                    sum(reordered_item_count) over (
                        partition by user_id
                        order by order_number
                        rows between unbounded preceding and 1 preceding
                    )
                    / nullif(
                        sum(item_count) over (
                            partition by user_id
                            order by order_number
                            rows between unbounded preceding and 1 preceding
                        ),
                        0
                    ),
                    0
                ) as reorder_ratio,
                coalesce(
                    sum(item_count) over (
                        partition by user_id
                        order by order_number
                        rows between unbounded preceding and 1 preceding
                    ),
                    0
                ) as unique_products,
                coalesce(
                    avg(days_since_prior_order) over (
                        partition by user_id
                        order by order_number
                        rows between unbounded preceding and 1 preceding
                    ),
                    0
                ) as days_between_orders,
                case
                    when reordered_item_count > 0 then 1
                    else 0
                end as will_reorder
            from {source_relation}
        )

        select
            user_id,
            order_id,
            order_number,
            total_orders,
            avg_basket_size,
            reorder_ratio,
            unique_products,
            days_between_orders,
            will_reorder
        from ordered_history
        where total_orders >= 1
            and will_reorder is not null
            and order_id in (
                select order_id
                from {source_relation}
                where eval_set = 'train'
                    and item_count > 0
            )
        order by user_id, order_number
        {limit_clause}
        """
    ).strip()


def build_customer_features(customers: pd.DataFrame) -> pd.DataFrame:
    features = customers.copy()
    if "total_spend" in features and "order_count" in features:
        features["average_order_value"] = features["total_spend"] / features["order_count"].clip(lower=1)
    return features
