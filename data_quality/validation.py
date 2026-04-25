import logging
from dataclasses import dataclass

from great_expectations.dataset import SqlAlchemyDataset
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


def build_dataset(table_name: str) -> SqlAlchemyDataset:
    settings = get_settings()
    engine = build_snowflake_engine(settings)
    return SqlAlchemyDataset(table_name=table_name, engine=engine)


def extract_unexpected_count(result: dict) -> int | None:
    result_payload = result.get("result", {})
    return result_payload.get("unexpected_count")


def run_expectation(check_name: str, result: dict) -> ValidationResult:
    validation_result = ValidationResult(
        check_name=check_name,
        success=bool(result.get("success")),
        unexpected_count=extract_unexpected_count(result),
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


def validate_orders() -> list[ValidationResult]:
    orders = build_dataset(ORDERS_TABLE)
    return [
        run_expectation(
            "orders.order_id_not_null",
            orders.expect_column_values_to_not_be_null("order_id"),
        ),
        run_expectation(
            "orders.order_id_unique",
            orders.expect_column_values_to_be_unique("order_id"),
        ),
    ]


def validate_order_products() -> list[ValidationResult]:
    order_products = build_dataset(ORDER_PRODUCTS_TABLE)
    results = [
        run_expectation(
            "order_products.order_id_not_null",
            order_products.expect_column_values_to_not_be_null("order_id"),
        ),
        run_expectation(
            "order_products.product_id_not_null",
            order_products.expect_column_values_to_not_be_null("product_id"),
        ),
        run_expectation(
            "order_products.order_product_unique",
            order_products.expect_compound_columns_to_be_unique(["order_id", "product_id"]),
        ),
    ]
    results.append(validate_product_id_references())
    return results


def validate_product_id_references() -> ValidationResult:
    settings = get_settings()
    engine = build_snowflake_engine(settings)
    query = text(
        f"""
        select count(*) as invalid_product_id_count
        from {ORDER_PRODUCTS_TABLE} as order_products
        left join {PRODUCTS_TABLE} as products
            on order_products.product_id = products.product_id
        where order_products.product_id is not null
            and products.product_id is null
        """
    )

    try:
        with engine.begin() as connection:
            invalid_count = connection.execute(query).scalar_one()
    finally:
        engine.dispose()

    result = ValidationResult(
        check_name="order_products.product_id_valid",
        success=invalid_count == 0,
        unexpected_count=int(invalid_count),
    )
    if result.success:
        logger.info("Validation passed: %s", result.check_name)
    else:
        logger.error(
            "Validation failed: %s unexpected_count=%s",
            result.check_name,
            result.unexpected_count,
        )
    return result


def validate_products() -> list[ValidationResult]:
    products = build_dataset(PRODUCTS_TABLE)
    return [
        run_expectation(
            "products.product_id_not_null",
            products.expect_column_values_to_not_be_null("product_id"),
        ),
        run_expectation(
            "products.product_id_unique",
            products.expect_column_values_to_be_unique("product_id"),
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
