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

churn as (
    select
        user_id,
        churn_probability,
        churn_risk_segment
    from {{ ref('customer_churn_probability') }}
),

estimated_value as (
    select
        features.user_id,
        features.total_orders,
        features.avg_basket_size,
        features.reorder_ratio,
        features.unique_products,
        features.days_between_orders,
        churn.churn_probability,
        churn.churn_risk_segment,
        features.total_orders * features.avg_basket_size as historical_value_proxy,
        features.avg_basket_size * greatest(1, features.total_orders * (1 - churn.churn_probability)) as predicted_future_value_proxy
    from features
    inner join churn
        on features.user_id = churn.user_id
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
        churn_risk_segment,
        historical_value_proxy,
        predicted_future_value_proxy,
        historical_value_proxy + predicted_future_value_proxy as customer_lifetime_value_proxy,
        case
            when historical_value_proxy + predicted_future_value_proxy >= 500 then 'high_value'
            when historical_value_proxy + predicted_future_value_proxy >= 150 then 'growth'
            else 'emerging'
        end as value_segment
    from estimated_value
)

select * from final
