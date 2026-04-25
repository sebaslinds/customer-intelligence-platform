with products as (
    select *
    from {{ source('raw', 'raw_instacart_products') }}
),

aisles as (
    select *
    from {{ source('raw', 'raw_instacart_aisles') }}
),

departments as (
    select *
    from {{ source('raw', 'raw_instacart_departments') }}
),

cleaned as (
    select
        cast(products.product_id as integer) as product_id,
        cast(products.product_name as varchar) as product_name,
        cast(products.aisle_id as integer) as aisle_id,
        cast(aisles.aisle as varchar) as aisle,
        cast(products.department_id as integer) as department_id,
        cast(departments.department as varchar) as department
    from products
    left join aisles
        on products.aisle_id = aisles.aisle_id
    left join departments
        on products.department_id = departments.department_id
)

select * from cleaned
