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
        count_if(coalesce(order_metrics.item_count, 0) > 0) as observed_basket_orders,
        min(orders.order_number) as first_order_number,
        max(orders.order_number) as latest_order_number,
        coalesce(avg(orders.days_since_prior_order), 0) as avg_days_since_prior_order,
        coalesce(stddev_samp(orders.days_since_prior_order), 0) as stddev_days_since_prior_order,
        coalesce(sum(orders.days_since_prior_order), 0) as customer_tenure_days,
        coalesce(max(orders.days_since_prior_order), 0) as max_days_since_prior_order,
        coalesce(avg(orders.order_dow), 0) as avg_order_dow,
        coalesce(avg(orders.order_hour_of_day), 0) as avg_order_hour_of_day,
        coalesce(avg(case when orders.order_dow in (0, 6) then 1 else 0 end), 0) as weekend_order_ratio,
        coalesce(
            avg(
                case
                    when orders.order_hour_of_day between 18 and 23
                        or orders.order_hour_of_day between 0 and 5
                        then 1
                    else 0
                end
            ),
            0
        ) as evening_order_ratio,
        sum(coalesce(order_metrics.item_count, 0)) as total_items_ordered,
        sum(coalesce(order_metrics.reordered_item_count, 0)) as total_reordered_items,
        case
            when sum(coalesce(order_metrics.item_count, 0)) > 0
                then sum(coalesce(order_metrics.reordered_item_count, 0))
                    / sum(coalesce(order_metrics.item_count, 0))
            else 0
        end as reorder_item_ratio,
        case
            when coalesce(sum(orders.days_since_prior_order), 0) > 0
                then count(distinct orders.order_id) / nullif(sum(orders.days_since_prior_order) / 30, 0)
            else count(distinct orders.order_id)
        end as order_frequency_30d
    from orders
    left join order_metrics
        on orders.order_id = order_metrics.order_id
    group by orders.user_id
),

final as (
    select
        user_id,
        order_count,
        observed_basket_orders,
        first_order_number,
        latest_order_number,
        avg_days_since_prior_order,
        stddev_days_since_prior_order,
        customer_tenure_days,
        max_days_since_prior_order,
        avg_order_dow,
        avg_order_hour_of_day,
        weekend_order_ratio,
        evening_order_ratio,
        total_items_ordered,
        total_reordered_items,
        reorder_item_ratio,
        order_frequency_30d,
        case
            when order_count >= 25 then 'high_frequency'
            when order_count >= 10 then 'medium_frequency'
            else 'low_frequency'
        end as customer_frequency_segment
    from user_metrics
)

select * from final
