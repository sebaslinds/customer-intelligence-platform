with source as (
    select *
    from {{ source('raw', 'raw_instacart_order_products_train') }}
),

cleaned as (
    select
        cast(order_id as varchar) || '-' || cast(product_id as varchar) as order_product_id,
        cast(order_id as integer) as order_id,
        cast(product_id as integer) as product_id,
        cast(add_to_cart_order as integer) as add_to_cart_order,
        cast(reordered as integer) as reordered
    from source
)

select * from cleaned
