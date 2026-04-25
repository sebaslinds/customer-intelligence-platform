{{
    config(
        materialized='table'
    )
}}

with orders as (
    select
        user_id,
        order_id,
        order_number,
        item_count
    from {{ ref('fct_orders') }}
),

user_cohorts as (
    select
        user_id,
        min(order_number) as cohort_order_number
    from orders
    group by user_id
),

cohort_activity as (
    select
        user_cohorts.cohort_order_number,
        orders.order_number - user_cohorts.cohort_order_number as cohort_period,
        count(distinct orders.user_id) as active_users,
        count(distinct orders.order_id) as orders_count,
        sum(orders.item_count) as revenue_proxy
    from orders
    inner join user_cohorts
        on orders.user_id = user_cohorts.user_id
    group by
        user_cohorts.cohort_order_number,
        orders.order_number - user_cohorts.cohort_order_number
),

cohort_sizes as (
    select
        cohort_order_number,
        active_users as cohort_size
    from cohort_activity
    where cohort_period = 0
),

final as (
    select
        cohort_activity.cohort_order_number,
        cohort_activity.cohort_period,
        cohort_sizes.cohort_size,
        cohort_activity.active_users,
        cohort_activity.orders_count,
        cohort_activity.revenue_proxy,
        cohort_activity.active_users / nullif(cohort_sizes.cohort_size, 0) as retention_rate
    from cohort_activity
    inner join cohort_sizes
        on cohort_activity.cohort_order_number = cohort_sizes.cohort_order_number
)

select * from final
