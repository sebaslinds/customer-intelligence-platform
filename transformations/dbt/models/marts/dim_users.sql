with orders as (
    select *
    from {{ ref('stg_orders') }}
),

order_products as (
    select *
    from {{ ref('stg_order_products') }}
),

order_metrics as (
    select
        order_id,
        count(*) as item_count,
        sum(reordered) as reordered_item_count
    from order_products
    group by order_id
),

user_metrics as (
    select
        orders.user_id,
        count(distinct orders.order_id) as order_count,
        min(orders.order_number) as first_order_number,
        max(orders.order_number) as latest_order_number,
        avg(orders.days_since_prior_order) as avg_days_since_prior_order,
        sum(coalesce(order_metrics.item_count, 0)) as total_items_ordered,
        sum(coalesce(order_metrics.reordered_item_count, 0)) as total_reordered_items
    from orders
    left join order_metrics
        on orders.order_id = order_metrics.order_id
    group by orders.user_id
),

final as (
    select
        user_id,
        order_count,
        first_order_number,
        latest_order_number,
        avg_days_since_prior_order,
        total_items_ordered,
        total_reordered_items
    from user_metrics
)

select * from final
