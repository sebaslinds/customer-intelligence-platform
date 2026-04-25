import argparse
import json
import logging
import random
import time
import uuid
from datetime import UTC, datetime
from typing import Any

from confluent_kafka import Producer

from config.logging_config import configure_logging
from config.settings import get_settings

logger = logging.getLogger(__name__)


def build_producer_config() -> dict[str, str]:
    settings = get_settings()
    return {"bootstrap.servers": settings.kafka_bootstrap_servers}


def build_order_event() -> dict[str, Any]:
    item_count = random.randint(1, 25)
    return {
        "event_id": str(uuid.uuid4()),
        "event_type": "order_created",
        "order_id": random.randint(1_000_000, 9_999_999),
        "user_id": random.randint(1, 250_000),
        "order_number": random.randint(1, 100),
        "order_dow": random.randint(0, 6),
        "order_hour_of_day": random.randint(0, 23),
        "item_count": item_count,
        "reordered_item_count": random.randint(0, item_count),
        "created_at": datetime.now(UTC).isoformat(),
    }


def delivery_report(error: Exception | None, message: Any) -> None:
    if error is not None:
        logger.error("Kafka delivery failed: %s", error)
        return

    logger.info(
        "Produced event to topic=%s partition=%s offset=%s",
        message.topic(),
        message.partition(),
        message.offset(),
    )


def produce_order_events(topic: str, event_count: int, interval_seconds: float) -> None:
    producer = Producer(build_producer_config())

    for _ in range(event_count):
        event = build_order_event()
        producer.produce(
            topic=topic,
            key=str(event["order_id"]),
            value=json.dumps(event),
            callback=delivery_report,
        )
        producer.poll(0)
        logger.info("Queued order event order_id=%s", event["order_id"])
        time.sleep(interval_seconds)

    producer.flush()


def parse_args() -> argparse.Namespace:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Simulate real-time Instacart order events.")
    parser.add_argument("--topic", default=settings.kafka_topic)
    parser.add_argument("--event-count", type=int, default=100)
    parser.add_argument("--interval-seconds", type=float, default=0.2)
    return parser.parse_args()


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    args = parse_args()
    produce_order_events(
        topic=args.topic,
        event_count=args.event_count,
        interval_seconds=args.interval_seconds,
    )


if __name__ == "__main__":
    main()
