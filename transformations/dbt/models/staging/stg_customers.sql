with source as (
    select *
    from {{ source('raw', 'raw_customers') }}
),

renamed as (
    select
        customer_id,
        email,
        first_name,
        last_name,
        created_at,
        total_spend,
        order_count
    from source
)

select * from renamed
