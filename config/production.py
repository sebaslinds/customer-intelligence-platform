from config.settings import Settings

REQUIRED_PRODUCTION_ENV_VARS = (
    "snowflake_account",
    "snowflake_user",
    "snowflake_password",
    "snowflake_warehouse",
    "snowflake_database",
    "snowflake_schema",
)


def validate_production_settings(settings: Settings) -> None:
    if settings.app_env.lower() != "production":
        return

    missing = [
        field_name
        for field_name in REQUIRED_PRODUCTION_ENV_VARS
        if not getattr(settings, field_name)
    ]
    if missing:
        formatted = ", ".join(missing)
        raise ValueError(f"Missing required production configuration: {formatted}")
