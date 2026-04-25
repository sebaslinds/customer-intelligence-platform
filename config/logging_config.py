from utils.logger import configure_structured_logging


def configure_logging(level: str = "INFO") -> None:
    configure_structured_logging(level)
