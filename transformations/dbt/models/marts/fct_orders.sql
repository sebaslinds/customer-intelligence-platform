{{
    config(
        materialized='incremental',
        unique_key='order_id',
        incremental_strategy='merge',
        on_schema_change='sync_all_columns'
    )
}}

with orders as (
    select *
    from {{ ref('stg_orders') }}

    {% if is_incremental() %}
        where order_id > (select coalesce(max(order_id), 0) from {{ this }})
    {% endif %}
),

order_products as (
    select order_products.*
    from {{ ref('stg_order_products') }} as order_products
    inner join orders
        on order_products.order_id = orders.order_id
),

order_metrics as (
    select
        order_id,
        count(*) as item_count,
        sum(reordered) as reordered_item_count,
        min(add_to_cart_order) as first_cart_position,
        max(add_to_cart_order) as last_cart_position
    from order_products
    group by order_id
),

final as (
    select
        orders.order_id,
        orders.user_id,
        orders.eval_set,
        orders.order_number,
        orders.order_dow,
        orders.order_hour_of_day,
        orders.days_since_prior_order,
        coalesce(order_metrics.item_count, 0) as item_count,
        coalesce(order_metrics.reordered_item_count, 0) as reordered_item_count,
        order_metrics.first_cart_position,
        order_metrics.last_cart_position
    from orders
    left join order_metrics
        on orders.order_id = order_metrics.order_id
)

select * from final
