from textwrap import dedent

import pandas as pd

FEATURE_COLUMNS = [
    "total_orders",
    "observed_basket_orders",
    "avg_basket_size",
    "reorder_ratio",
    "days_between_orders",
    "stddev_days_between_orders",
    "days_since_last_order",
    "customer_tenure_days",
    "order_frequency_30d",
    "avg_order_dow",
    "avg_order_hour_of_day",
    "weekend_order_ratio",
    "evening_order_ratio",
    "prior_reorder_order_ratio",
    "unique_products",
    "unique_departments",
    "unique_aisles",
    "produce_item_ratio",
    "dairy_eggs_item_ratio",
    "fresh_fruits_item_ratio",
    "fresh_vegetables_item_ratio",
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
        with order_timeline as (
            select
                user_id,
                order_id,
                eval_set,
                order_number,
                order_dow,
                order_hour_of_day,
                item_count,
                reordered_item_count,
                coalesce(days_since_prior_order, 0) as days_since_prior_order,
                case when order_dow in (0, 6) then 1 else 0 end as is_weekend_order,
                case when order_hour_of_day between 18 and 23 or order_hour_of_day between 0 and 5 then 1 else 0 end
                    as is_evening_order,
                case when reordered_item_count > 0 then 1 else 0 end as has_reordered_items,
                sum(coalesce(days_since_prior_order, 0)) over (
                    partition by user_id
                    order by order_number
                    rows between unbounded preceding and current row
                ) as customer_day_number
            from {source_relation}
        ),

        ordered_history as (
            select
                user_id,
                order_id,
                eval_set,
                order_number,
                count(*) over (
                    partition by user_id
                    order by order_number
                    rows between unbounded preceding and 1 preceding
                ) as total_orders,
                count_if(item_count > 0) over (
                    partition by user_id
                    order by order_number
                    rows between unbounded preceding and 1 preceding
                ) as observed_basket_orders,
                coalesce(
                    avg(nullif(item_count, 0)) over (
                        partition by user_id
                        order by order_number
                        rows between unbounded preceding and 1 preceding
                    ),
                    0
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
                    sum(has_reordered_items) over (
                        partition by user_id
                        order by order_number
                        rows between unbounded preceding and 1 preceding
                    ),
                    0
                ) / nullif(
                    count(*) over (
                        partition by user_id
                        order by order_number
                        rows between unbounded preceding and 1 preceding
                    ),
                    0
                ) as prior_reorder_order_ratio,
                coalesce(
                    avg(days_since_prior_order) over (
                        partition by user_id
                        order by order_number
                        rows between unbounded preceding and 1 preceding
                    ),
                    0
                ) as days_between_orders,
                coalesce(
                    stddev_samp(days_since_prior_order) over (
                        partition by user_id
                        order by order_number
                        rows between unbounded preceding and 1 preceding
                    ),
                    0
                ) as stddev_days_between_orders,
                days_since_prior_order as days_since_last_order,
                coalesce(
                    sum(days_since_prior_order) over (
                        partition by user_id
                        order by order_number
                        rows between unbounded preceding and 1 preceding
                    ),
                    0
                ) as customer_tenure_days,
                coalesce(
                    count(*) over (
                        partition by user_id
                        order by customer_day_number
                        range between 30 preceding and 1 preceding
                    ),
                    0
                ) as order_frequency_30d,
                coalesce(
                    avg(order_dow) over (
                        partition by user_id
                        order by order_number
                        rows between unbounded preceding and 1 preceding
                    ),
                    0
                ) as avg_order_dow,
                coalesce(
                    avg(order_hour_of_day) over (
                        partition by user_id
                        order by order_number
                        rows between unbounded preceding and 1 preceding
                    ),
                    0
                ) as avg_order_hour_of_day,
                coalesce(
                    avg(is_weekend_order) over (
                        partition by user_id
                        order by order_number
                        rows between unbounded preceding and 1 preceding
                    ),
                    0
                ) as weekend_order_ratio,
                coalesce(
                    avg(is_evening_order) over (
                        partition by user_id
                        order by order_number
                        rows between unbounded preceding and 1 preceding
                    ),
                    0
                ) as evening_order_ratio,
                case
                    when reordered_item_count > 0 then 1
                    else 0
                end as will_reorder
            from order_timeline
        ),

        product_events as (
            select
                order_timeline.user_id,
                order_timeline.order_id,
                order_timeline.order_number,
                stg_order_products.product_id,
                lower(coalesce(dim_products.department, 'unknown')) as department,
                lower(coalesce(dim_products.aisle, 'unknown')) as aisle
            from order_timeline
            inner join stg_order_products
                on order_timeline.order_id = stg_order_products.order_id
            inner join dim_products
                on stg_order_products.product_id = dim_products.product_id
        ),

        product_history as (
            select
                ordered_history.user_id,
                ordered_history.order_id,
                count(distinct product_events.product_id) as unique_products,
                count(distinct product_events.department) as unique_departments,
                count(distinct product_events.aisle) as unique_aisles,
                coalesce(
                    count_if(product_events.department = 'produce') / nullif(count(product_events.order_id), 0),
                    0
                ) as produce_item_ratio,
                coalesce(
                    count_if(product_events.department = 'dairy eggs') / nullif(count(product_events.order_id), 0),
                    0
                ) as dairy_eggs_item_ratio,
                coalesce(
                    count_if(product_events.aisle = 'fresh fruits') / nullif(count(product_events.order_id), 0),
                    0
                ) as fresh_fruits_item_ratio,
                coalesce(
                    count_if(product_events.aisle = 'fresh vegetables') / nullif(count(product_events.order_id), 0),
                    0
                ) as fresh_vegetables_item_ratio
            from ordered_history
            left join product_events
                on ordered_history.user_id = product_events.user_id
                and product_events.order_number < ordered_history.order_number
            group by ordered_history.user_id, ordered_history.order_id
        )

        select
            ordered_history.user_id,
            ordered_history.order_id,
            ordered_history.order_number,
            ordered_history.total_orders,
            ordered_history.observed_basket_orders,
            ordered_history.avg_basket_size,
            ordered_history.reorder_ratio,
            coalesce(ordered_history.prior_reorder_order_ratio, 0) as prior_reorder_order_ratio,
            ordered_history.days_between_orders,
            ordered_history.stddev_days_between_orders,
            ordered_history.days_since_last_order,
            ordered_history.customer_tenure_days,
            ordered_history.order_frequency_30d,
            ordered_history.avg_order_dow,
            ordered_history.avg_order_hour_of_day,
            ordered_history.weekend_order_ratio,
            ordered_history.evening_order_ratio,
            coalesce(product_history.unique_products, 0) as unique_products,
            coalesce(product_history.unique_departments, 0) as unique_departments,
            coalesce(product_history.unique_aisles, 0) as unique_aisles,
            coalesce(product_history.produce_item_ratio, 0) as produce_item_ratio,
            coalesce(product_history.dairy_eggs_item_ratio, 0) as dairy_eggs_item_ratio,
            coalesce(product_history.fresh_fruits_item_ratio, 0) as fresh_fruits_item_ratio,
            coalesce(product_history.fresh_vegetables_item_ratio, 0) as fresh_vegetables_item_ratio,
            ordered_history.will_reorder
        from ordered_history
        left join product_history
            on ordered_history.user_id = product_history.user_id
            and ordered_history.order_id = product_history.order_id
        where ordered_history.total_orders >= 1
            and ordered_history.will_reorder is not null
            and ordered_history.order_id in (
                select order_id
                from {source_relation}
                where eval_set = 'train'
                    and item_count > 0
            )
        order by ordered_history.user_id, ordered_history.order_number
        {limit_clause}
        """
    ).strip()


def build_customer_features(customers: pd.DataFrame) -> pd.DataFrame:
    features = customers.copy()
    if "total_spend" in features and "order_count" in features:
        features["average_order_value"] = features["total_spend"] / features["order_count"].clip(lower=1)
    return features
