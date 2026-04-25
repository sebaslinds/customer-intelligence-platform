import argparse
import logging
import os
from collections.abc import Iterable
from pathlib import Path

import pandas as pd
from sqlalchemy import Engine

from config.logging_config import configure_logging
from config.settings import get_settings
from ingestion.snowflake_client import build_snowflake_engine
from utils.logger import PipelineTimer, log_rows_processed

logger = logging.getLogger(__name__)

DEFAULT_CHUNK_SIZE = 100_000
DEFAULT_DATA_DIR = Path("data/raw")

INSTACART_FILES = {
    "orders.csv": "raw_instacart_orders",
    "order_products__train.csv": "raw_instacart_order_products_train",
    "products.csv": "raw_instacart_products",
    "aisles.csv": "raw_instacart_aisles",
    "departments.csv": "raw_instacart_departments",
}


def normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame.columns = [
        column.strip().lower().replace(" ", "_").replace("-", "_")
        for column in frame.columns
    ]
    return frame


def iter_csv_chunks(csv_path: Path, chunksize: int) -> Iterable[pd.DataFrame]:
    return pd.read_csv(csv_path, chunksize=chunksize)


def load_csv_to_snowflake(
    engine: Engine,
    csv_path: Path,
    table_name: str,
    chunksize: int = DEFAULT_CHUNK_SIZE,
) -> int:
    if not csv_path.exists():
        raise FileNotFoundError(f"Required source file not found: {csv_path}")

    total_rows = 0

    with PipelineTimer(logger, "load_csv_to_snowflake", table_name=table_name, source_file=str(csv_path)):
        logger.info(
            "Starting CSV load",
            extra={"extra_fields": {"source_file": str(csv_path), "table_name": table_name}},
        )

        for chunk_number, chunk in enumerate(iter_csv_chunks(csv_path, chunksize), start=1):
            frame = normalize_columns(chunk)
            write_mode = "replace" if chunk_number == 1 else "append"

            frame.to_sql(
                name=table_name,
                con=engine,
                if_exists=write_mode,
                index=False,
                chunksize=chunksize,
                method="multi",
            )

            total_rows += len(frame)
            log_rows_processed(
                logger,
                len(frame),
                table_name=table_name,
                chunk_number=chunk_number,
            )
            logger.info(
                "Loaded CSV chunk",
                extra={
                    "extra_fields": {
                        "chunk_number": chunk_number,
                        "table_name": table_name,
                        "rows_loaded": len(frame),
                        "total_rows": total_rows,
                    }
                },
            )

    log_rows_processed(logger, total_rows, table_name=table_name, metric_scope="file_total")
    logger.info(
        "Completed CSV load",
        extra={"extra_fields": {"table_name": table_name, "total_rows": total_rows}},
    )
    return total_rows


def load_instacart_dataset(
    data_dir: Path = DEFAULT_DATA_DIR,
    chunksize: int = DEFAULT_CHUNK_SIZE,
) -> dict[str, int]:
    settings = get_settings()
    engine = build_snowflake_engine(settings)
    load_results: dict[str, int] = {}

    with PipelineTimer(logger, "load_instacart_dataset", data_dir=str(data_dir), chunksize=chunksize):
        try:
            for file_name, table_name in INSTACART_FILES.items():
                csv_path = data_dir / file_name
                load_results[table_name] = load_csv_to_snowflake(
                    engine=engine,
                    csv_path=csv_path,
                    table_name=table_name,
                    chunksize=chunksize,
                )
        except Exception:
            logger.exception("Instacart ingestion pipeline failed")
            raise
        finally:
            engine.dispose()

    return load_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load Instacart CSV files into Snowflake.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(os.getenv("INSTACART_DATA_DIR", DEFAULT_DATA_DIR)),
        help="Directory containing Instacart CSV files.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=int(os.getenv("INGESTION_CHUNK_SIZE", DEFAULT_CHUNK_SIZE)),
        help="Number of CSV rows to process per batch.",
    )
    return parser.parse_args()


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    args = parse_args()

    logger.info(
        "Running Instacart ingestion with data_dir=%s chunksize=%s",
        args.data_dir,
        args.chunksize,
    )
    results = load_instacart_dataset(data_dir=args.data_dir, chunksize=args.chunksize)

    for table_name, row_count in results.items():
        logger.info("Load summary: %s=%s rows", table_name, row_count)


if __name__ == "__main__":
    main()
