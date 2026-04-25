{{
    config(
        materialized='table'
    )
}}

with orders as (
    select
        order_id,
        user_id,
        item_count,
        reordered_item_count,
        days_since_prior_order
    from {{ ref('fct_orders') }}
),

order_products as (
    select
        order_id,
        product_id
    from {{ ref('stg_order_products') }}
),

products as (
    select product_id
    from {{ ref('dim_products') }}
),

user_order_features as (
    select
        user_id,
        count(distinct order_id) as total_orders,
        coalesce(avg(item_count), 0) as avg_basket_size,
        coalesce(sum(reordered_item_count) / nullif(sum(item_count), 0), 0) as reorder_ratio,
        coalesce(avg(days_since_prior_order), 0) as days_between_orders,
        case
            when sum(reordered_item_count) > 0 then 1
            else 0
        end as will_reorder
    from orders
    group by user_id
),

user_product_features as (
    select
        orders.user_id,
        count(distinct products.product_id) as unique_products
    from orders
    inner join order_products
        on orders.order_id = order_products.order_id
    inner join products
        on order_products.product_id = products.product_id
    group by orders.user_id
),

final as (
    select
        user_order_features.user_id,
        user_order_features.total_orders,
        user_order_features.avg_basket_size,
        user_order_features.reorder_ratio,
        coalesce(user_product_features.unique_products, 0) as unique_products,
        user_order_features.days_between_orders,
        user_order_features.will_reorder
    from user_order_features
    left join user_product_features
        on user_order_features.user_id = user_product_features.user_id
)

select * from final
