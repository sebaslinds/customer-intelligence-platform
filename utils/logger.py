import json
import logging
import sys
import time
from contextlib import ContextDecorator
from datetime import UTC, datetime
from typing import Any


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        extra_fields = getattr(record, "extra_fields", None)
        if isinstance(extra_fields, dict):
            payload.update(extra_fields)

        return json.dumps(payload, default=str)


def configure_structured_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(level.upper())


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def log_metric(logger: logging.Logger, metric_name: str, metric_value: Any, **dimensions: Any) -> None:
    logger.info(
        "pipeline_metric",
        extra={
            "extra_fields": {
                "metric_name": metric_name,
                "metric_value": metric_value,
                **dimensions,
            }
        },
    )


def log_rows_processed(logger: logging.Logger, rows: int, **dimensions: Any) -> None:
    log_metric(logger, "rows_processed", rows, **dimensions)


class PipelineTimer(ContextDecorator):
    def __init__(self, logger: logging.Logger, pipeline_name: str, **dimensions: Any) -> None:
        self.logger = logger
        self.pipeline_name = pipeline_name
        self.dimensions = dimensions
        self.started_at = 0.0

    def __enter__(self) -> "PipelineTimer":
        self.started_at = time.perf_counter()
        self.logger.info(
            "pipeline_started",
            extra={
                "extra_fields": {
                    "pipeline_name": self.pipeline_name,
                    **self.dimensions,
                }
            },
        )
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> bool:
        duration_seconds = round(time.perf_counter() - self.started_at, 4)
        status = "failed" if exc_type else "succeeded"
        self.logger.info(
            "pipeline_finished",
            extra={
                "extra_fields": {
                    "pipeline_name": self.pipeline_name,
                    "status": status,
                    "duration_seconds": duration_seconds,
                    **self.dimensions,
                }
            },
        )
        log_metric(
            self.logger,
            "execution_time_seconds",
            duration_seconds,
            pipeline_name=self.pipeline_name,
            status=status,
            **self.dimensions,
        )
        return False
