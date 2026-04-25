with customers as (
    select *
    from {{ ref('stg_customers') }}
)

select
    customer_id,
    email,
    first_name,
    last_name,
    created_at,
    total_spend,
    order_count,
    total_spend / nullif(order_count, 0) as average_order_value,
    case
        when total_spend >= 1000 then 'high_value'
        when total_spend >= 250 then 'growth'
        else 'emerging'
    end as customer_segment
from customers
