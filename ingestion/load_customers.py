import logging
from pathlib import Path

import pandas as pd

from config.logging_config import configure_logging
from config.settings import get_settings
from ingestion.snowflake_client import build_snowflake_engine

logger = logging.getLogger(__name__)


def load_customer_csv(path: Path, table_name: str = "raw_customers") -> int:
    settings = get_settings()
    engine = build_snowflake_engine(settings)
    frame = pd.read_csv(path)

    with engine.begin() as connection:
        frame.to_sql(table_name, connection, if_exists="append", index=False)

    logger.info("Loaded %s records into %s", len(frame), table_name)
    return len(frame)


if __name__ == "__main__":
    configure_logging(get_settings().log_level)
    load_customer_csv(Path("data/raw/customers.csv"))
