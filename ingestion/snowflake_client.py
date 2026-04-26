from sqlalchemy import Engine, create_engine

from config.settings import Settings


def build_snowflake_engine(settings: Settings) -> Engine:
    required = {
        "account": settings.snowflake_account,
        "user": settings.snowflake_user,
        "password": settings.snowflake_password,
        "warehouse": settings.snowflake_warehouse,
        "database": settings.snowflake_database,
        "schema": settings.snowflake_schema,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise ValueError(f"Missing Snowflake configuration: {', '.join(missing)}")

    role = f"&role={settings.snowflake_role}" if settings.snowflake_role else ""
    url = (
        f"snowflake://{settings.snowflake_user}:{settings.snowflake_password}"
        f"@{settings.snowflake_account}/{settings.snowflake_database}/{settings.snowflake_schema}"
        f"?warehouse={settings.snowflake_warehouse}{role}"
    )
    connect_args = {}
    if settings.snowflake_insecure_mode:
        connect_args["insecure_mode"] = True

    return create_engine(url, connect_args=connect_args)
