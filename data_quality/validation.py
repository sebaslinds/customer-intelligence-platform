import logging
from dataclasses import dataclass

from sqlalchemy import text

from config.logging_config import configure_logging
from config.settings import get_settings
from ingestion.snowflake_client import build_snowflake_engine

logger = logging.getLogger(__name__)

ORDERS_TABLE = "raw_instacart_orders"
ORDER_PRODUCTS_TABLE = "raw_instacart_order_products_train"
PRODUCTS_TABLE = "raw_instacart_products"


@dataclass(frozen=True)
class ValidationResult:
    check_name: str
    success: bool
    unexpected_count: int | None = None


def run_count_check(check_name: str, invalid_count: int) -> ValidationResult:
    validation_result = ValidationResult(
        check_name=check_name,
        success=invalid_count == 0,
        unexpected_count=invalid_count,
    )

    if validation_result.success:
        logger.info("Validation passed: %s", check_name)
    else:
        logger.error(
            "Validation failed: %s unexpected_count=%s",
            check_name,
            validation_result.unexpected_count,
        )

    return validation_result


def fetch_invalid_count(query: str) -> int:
    settings = get_settings()
    engine = build_snowflake_engine(settings)
    try:
        with engine.begin() as connection:
            return int(connection.execute(text(query)).scalar_one())
    finally:
        engine.dispose()


def validate_orders() -> list[ValidationResult]:
    return [
        run_count_check(
            "orders.order_id_not_null",
            fetch_invalid_count(f"select count(*) from {ORDERS_TABLE} where order_id is null"),
        ),
        run_count_check(
            "orders.order_id_unique",
            fetch_invalid_count(
                f"""
                select count(*)
                from (
                    select order_id
                    from {ORDERS_TABLE}
                    group by order_id
                    having count(*) > 1
                )
                """
            ),
        ),
    ]


def validate_product_id_references() -> ValidationResult:
    invalid_count = fetch_invalid_count(
        f"""
        select count(*) as invalid_product_id_count
        from {ORDER_PRODUCTS_TABLE} as order_products
        left join {PRODUCTS_TABLE} as products
            on order_products.product_id = products.product_id
        where order_products.product_id is not null
            and products.product_id is null
        """
    )
    return run_count_check("order_products.product_id_valid", invalid_count)


def validate_order_products() -> list[ValidationResult]:
    return [
        run_count_check(
            "order_products.order_id_not_null",
            fetch_invalid_count(f"select count(*) from {ORDER_PRODUCTS_TABLE} where order_id is null"),
        ),
        run_count_check(
            "order_products.product_id_not_null",
            fetch_invalid_count(f"select count(*) from {ORDER_PRODUCTS_TABLE} where product_id is null"),
        ),
        run_count_check(
            "order_products.order_product_unique",
            fetch_invalid_count(
                f"""
                select count(*)
                from (
                    select order_id, product_id
                    from {ORDER_PRODUCTS_TABLE}
                    group by order_id, product_id
                    having count(*) > 1
                )
                """
            ),
        ),
        validate_product_id_references(),
    ]


def validate_products() -> list[ValidationResult]:
    return [
        run_count_check(
            "products.product_id_not_null",
            fetch_invalid_count(f"select count(*) from {PRODUCTS_TABLE} where product_id is null"),
        ),
        run_count_check(
            "products.product_id_unique",
            fetch_invalid_count(
                f"""
                select count(*)
                from (
                    select product_id
                    from {PRODUCTS_TABLE}
                    group by product_id
                    having count(*) > 1
                )
                """
            ),
        ),
    ]


def run_data_quality_checks() -> list[ValidationResult]:
    logger.info("Starting Great Expectations data validation")
    results = [
        *validate_orders(),
        *validate_products(),
        *validate_order_products(),
    ]

    failed_checks = [result.check_name for result in results if not result.success]
    if failed_checks:
        failed = ", ".join(failed_checks)
        raise ValueError(f"Data validation failed for checks: {failed}")

    logger.info("All Great Expectations checks passed")
    return results


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    run_data_quality_checks()


if __name__ == "__main__":
    main()
