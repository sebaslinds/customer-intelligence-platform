# Kafka Streaming Pipeline

This folder contains a simple real-time order event pipeline.

Install the Kafka-specific dependency before running the producer or consumer:

```bash
pip install -r requirements-streaming.txt
```

## Producer

Simulates JSON order events and publishes them to Kafka:

```bash
python -m streaming.producer --topic instacart.orders --event-count 100
```

## Consumer

Consumes JSON events in batches and appends them to Snowflake:

```bash
python -m streaming.consumer --topic instacart.orders --batch-size 100
```

Default Snowflake target table:

```text
raw_streaming_orders
```

## Required Environment Variables

- `KAFKA_BOOTSTRAP_SERVERS`
- `KAFKA_TOPIC`
- `KAFKA_CONSUMER_GROUP`
- `KAFKA_BATCH_SIZE`
- Snowflake variables from `.env.example`
