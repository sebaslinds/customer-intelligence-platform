from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "customer-intelligence-platform"
    app_env: str = "development"
    log_level: str = "INFO"

    snowflake_account: str | None = None
    snowflake_user: str | None = None
    snowflake_password: str | None = None
    snowflake_role: str | None = None
    snowflake_warehouse: str | None = None
    snowflake_database: str | None = None
    snowflake_schema: str | None = None
    snowflake_insecure_mode: bool = False

    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_base_url: str = "http://localhost:8000"
    model_path: str = "ml/artifacts/random_forest_reorder_model.joblib"
    openai_api_key: str | None = None
    openai_model: str = "gpt-4.1-mini"
    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_topic: str = "instacart.orders"
    kafka_consumer_group: str = "customer-intelligence-streaming"
    kafka_batch_size: int = 100

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()
