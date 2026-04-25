import argparse
import json
import logging
import time
from collections.abc import Iterable
from typing import Any

import pandas as pd
from confluent_kafka import Consumer, KafkaException
from sqlalchemy import Engine

from config.logging_config import configure_logging
from config.settings import get_settings
from ingestion.snowflake_client import build_snowflake_engine

logger = logging.getLogger(__name__)

DEFAULT_TABLE_NAME = "raw_streaming_orders"
REQUIRED_EVENT_FIELDS = {
    "event_id",
    "event_type",
    "order_id",
    "user_id",
    "order_number",
    "order_dow",
    "order_hour_of_day",
    "item_count",
    "reordered_item_count",
    "created_at",
}


def build_consumer_config(group_id: str) -> dict[str, str]:
    settings = get_settings()
    return {
        "bootstrap.servers": settings.kafka_bootstrap_servers,
        "group.id": group_id,
        "auto.offset.reset": "earliest",
        "enable.auto.commit": "false",
    }


def parse_event(raw_value: bytes) -> dict[str, Any]:
    try:
        event = json.loads(raw_value.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON event: {exc}") from exc

    missing_fields = REQUIRED_EVENT_FIELDS.difference(event)
    if missing_fields:
        missing = ", ".join(sorted(missing_fields))
        raise ValueError(f"Missing required event fields: {missing}")

    return event


def flush_events_to_snowflake(
    engine: Engine,
    events: Iterable[dict[str, Any]],
    table_name: str = DEFAULT_TABLE_NAME,
) -> int:
    frame = pd.DataFrame(events)
    if frame.empty:
        return 0

    frame.to_sql(table_name, engine, if_exists="append", index=False, method="multi")
    logger.info("Flushed %s streaming events into Snowflake table %s", len(frame), table_name)
    return len(frame)


def consume_order_events(
    topic: str,
    group_id: str,
    batch_size: int,
    flush_interval_seconds: int,
    table_name: str = DEFAULT_TABLE_NAME,
) -> None:
    consumer = Consumer(build_consumer_config(group_id))
    engine = build_snowflake_engine(get_settings())
    buffer: list[dict[str, Any]] = []
    last_flush_time = time.monotonic()

    try:
        consumer.subscribe([topic])
        logger.info("Consuming Kafka topic=%s group_id=%s", topic, group_id)

        while True:
            message = consumer.poll(1.0)
            if message is None:
                should_flush = buffer and time.monotonic() - last_flush_time >= flush_interval_seconds
                if should_flush:
                    flush_events_to_snowflake(engine, buffer, table_name)
                    consumer.commit(asynchronous=False)
                    buffer.clear()
                    last_flush_time = time.monotonic()
                continue

            if message.error():
                raise KafkaException(message.error())

            try:
                buffer.append(parse_event(message.value()))
            except ValueError:
                logger.exception("Skipping malformed Kafka message")
                consumer.commit(message=message, asynchronous=False)
                continue

            if len(buffer) >= batch_size:
                flush_events_to_snowflake(engine, buffer, table_name)
                consumer.commit(asynchronous=False)
                buffer.clear()
                last_flush_time = time.monotonic()
    except KeyboardInterrupt:
        logger.info("Stopping Kafka consumer")
    except Exception:
        logger.exception("Kafka consumer failed")
        raise
    finally:
        if buffer:
            flush_events_to_snowflake(engine, buffer, table_name)
            consumer.commit(asynchronous=False)
        consumer.close()
        engine.dispose()


def parse_args() -> argparse.Namespace:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Consume order events from Kafka into Snowflake.")
    parser.add_argument("--topic", default=settings.kafka_topic)
    parser.add_argument("--group-id", default=settings.kafka_consumer_group)
    parser.add_argument("--batch-size", type=int, default=settings.kafka_batch_size)
    parser.add_argument("--flush-interval-seconds", type=int, default=30)
    parser.add_argument("--table-name", default=DEFAULT_TABLE_NAME)
    return parser.parse_args()


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    args = parse_args()
    consume_order_events(
        topic=args.topic,
        group_id=args.group_id,
        batch_size=args.batch_size,
        flush_interval_seconds=args.flush_interval_seconds,
        table_name=args.table_name,
    )


if __name__ == "__main__":
    main()
