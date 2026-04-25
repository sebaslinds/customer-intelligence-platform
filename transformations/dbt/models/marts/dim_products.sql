with products as (
    select *
    from {{ ref('stg_products') }}
),

order_products as (
    select *
    from {{ ref('stg_order_products') }}
),

product_metrics as (
    select
        product_id,
        count(*) as order_line_count,
        sum(reordered) as reordered_line_count
    from order_products
    group by product_id
),

final as (
    select
        products.product_id,
        products.product_name,
        products.aisle_id,
        products.aisle,
        products.department_id,
        products.department,
        coalesce(product_metrics.order_line_count, 0) as order_line_count,
        coalesce(product_metrics.reordered_line_count, 0) as reordered_line_count
    from products
    left join product_metrics
        on products.product_id = product_metrics.product_id
)

select * from final
