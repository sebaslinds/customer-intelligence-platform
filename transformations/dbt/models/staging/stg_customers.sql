with orders as (
    select *
    from {{ ref('stg_orders') }}
),

customer_orders as (
    select
        user_id,
        count(distinct order_id) as order_count,
        min(order_number) as first_order_number,
        max(order_number) as last_order_number,
        min(order_dow) as first_observed_order_dow,
        max(order_dow) as last_observed_order_dow,
        avg(order_dow) as avg_order_dow,
        avg(order_hour_of_day) as avg_order_hour_of_day,
        avg(days_since_prior_order) as avg_days_since_prior_order,
        count_if(eval_set = 'prior') as prior_order_count,
        count_if(eval_set = 'train') as train_order_count,
        count_if(eval_set = 'test') as test_order_count
    from orders
    where user_id is not null
    group by user_id
)

select * from customer_orders
