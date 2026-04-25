{{
    config(
        materialized='table'
    )
}}

with features as (
    select
        user_id,
        total_orders,
        avg_basket_size,
        reorder_ratio,
        unique_products,
        days_between_orders
    from {{ ref('feature_store') }}
),

scored as (
    select
        user_id,
        total_orders,
        avg_basket_size,
        reorder_ratio,
        unique_products,
        days_between_orders,
        least(
            0.95,
            greatest(
                0.05,
                0.50
                + case when total_orders <= 2 then 0.20 else -0.10 end
                + case when reorder_ratio < 0.20 then 0.20 else -0.15 end
                + case when days_between_orders > 14 then 0.20 else -0.05 end
                + case when unique_products <= 5 then 0.10 else -0.05 end
            )
        ) as churn_probability
    from features
),

final as (
    select
        user_id,
        total_orders,
        avg_basket_size,
        reorder_ratio,
        unique_products,
        days_between_orders,
        churn_probability,
        case
            when churn_probability >= 0.70 then 'high'
            when churn_probability >= 0.40 then 'medium'
            else 'low'
        end as churn_risk_segment
    from scored
)

select * from final
