{{
    config(
        materialized='table'
    )
}}

with orders as (
    select
        order_id,
        user_id,
        order_number,
        order_dow,
        order_hour_of_day,
        item_count,
        reordered_item_count,
        days_since_prior_order,
        case when order_dow in (0, 6) then 1 else 0 end as is_weekend_order,
        case
            when order_hour_of_day between 18 and 23
                or order_hour_of_day between 0 and 5
                then 1
            else 0
        end as is_evening_order,
        case when reordered_item_count > 0 then 1 else 0 end as has_reordered_items
    from {{ ref('fct_orders') }}
),

order_products as (
    select
        order_id,
        product_id
    from {{ ref('stg_order_products') }}
),

products as (
    select
        product_id,
        lower(coalesce(department, 'unknown')) as department,
        lower(coalesce(aisle, 'unknown')) as aisle
    from {{ ref('dim_products') }}
),

user_order_features as (
    select
        user_id,
        count(distinct order_id) as total_orders,
        count_if(item_count > 0) as observed_basket_orders,
        coalesce(avg(nullif(item_count, 0)), 0) as avg_basket_size,
        coalesce(stddev_samp(nullif(item_count, 0)), 0) as stddev_basket_size,
        case
            when sum(item_count) > 0
                then (sum(reordered_item_count) + 0.5) / (sum(item_count) + 1)
            else 0
        end as reorder_ratio,
        coalesce(avg(has_reordered_items), 0) as reorder_order_ratio,
        coalesce(avg(days_since_prior_order), 0) as days_between_orders,
        coalesce(stddev_samp(days_since_prior_order), 0) as stddev_days_between_orders,
        coalesce(sum(days_since_prior_order), 0) as customer_tenure_days,
        coalesce(max(days_since_prior_order), 0) as max_days_between_orders,
        coalesce(avg(order_dow), 0) as avg_order_dow,
        coalesce(avg(order_hour_of_day), 0) as avg_order_hour_of_day,
        coalesce(avg(is_weekend_order), 0) as weekend_order_ratio,
        coalesce(avg(is_evening_order), 0) as evening_order_ratio,
        count(distinct order_dow) as active_order_dow_count,
        case
            when count(distinct order_id) > 0
                then count_if(item_count > 0) / count(distinct order_id)
            else 0
        end as observed_basket_coverage,
        case
            when coalesce(sum(days_since_prior_order), 0) > 0
                then count(distinct order_id) / nullif(sum(days_since_prior_order) / 30, 0)
            else count(distinct order_id)
        end as order_frequency_30d,
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
        count(distinct products.product_id) as unique_products,
        count(distinct products.department) as unique_departments,
        count(distinct products.aisle) as unique_aisles,
        coalesce(
            count_if(products.department = 'produce') / nullif(count(*), 0),
            0
        ) as produce_item_ratio,
        coalesce(
            count_if(products.department = 'dairy eggs') / nullif(count(*), 0),
            0
        ) as dairy_eggs_item_ratio,
        coalesce(
            count_if(products.aisle = 'fresh fruits') / nullif(count(*), 0),
            0
        ) as fresh_fruits_item_ratio,
        coalesce(
            count_if(products.aisle = 'fresh vegetables') / nullif(count(*), 0),
            0
        ) as fresh_vegetables_item_ratio
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
        user_order_features.observed_basket_orders,
        user_order_features.avg_basket_size,
        user_order_features.stddev_basket_size,
        user_order_features.reorder_ratio,
        user_order_features.reorder_order_ratio,
        coalesce(user_product_features.unique_products, 0) as unique_products,
        coalesce(user_product_features.unique_departments, 0) as unique_departments,
        coalesce(user_product_features.unique_aisles, 0) as unique_aisles,
        coalesce(user_product_features.produce_item_ratio, 0) as produce_item_ratio,
        coalesce(user_product_features.dairy_eggs_item_ratio, 0) as dairy_eggs_item_ratio,
        coalesce(user_product_features.fresh_fruits_item_ratio, 0) as fresh_fruits_item_ratio,
        coalesce(user_product_features.fresh_vegetables_item_ratio, 0) as fresh_vegetables_item_ratio,
        user_order_features.days_between_orders,
        user_order_features.stddev_days_between_orders,
        user_order_features.customer_tenure_days,
        user_order_features.max_days_between_orders,
        user_order_features.avg_order_dow,
        user_order_features.avg_order_hour_of_day,
        user_order_features.weekend_order_ratio,
        user_order_features.evening_order_ratio,
        user_order_features.active_order_dow_count,
        user_order_features.observed_basket_coverage,
        user_order_features.order_frequency_30d,
        user_order_features.will_reorder
    from user_order_features
    left join user_product_features
        on user_order_features.user_id = user_product_features.user_id
)

select * from final
