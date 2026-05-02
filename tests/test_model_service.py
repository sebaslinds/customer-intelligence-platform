import joblib
import pytest

import services.prediction.model_service as model_service
from config.settings import Settings


class DummyModel:
    feature_names_in_ = ["total_orders", "avg_basket_size"]

    def predict_proba(self, features):
        return [[0.2, 0.8]]


@pytest.fixture(autouse=True)
def reset_model_state():
    model_service.model_state["model"] = None
    model_service.model_state["feature_columns"] = model_service.DEFAULT_FEATURE_COLUMNS
    yield
    model_service.model_state["model"] = None
    model_service.model_state["feature_columns"] = model_service.DEFAULT_FEATURE_COLUMNS


def test_load_model_uses_snowflake_uri_when_configured(monkeypatch, tmp_path) -> None:
    model_path = tmp_path / "random_forest_reorder_model.joblib"
    joblib.dump(DummyModel(), model_path)

    monkeypatch.setattr(model_service, "download_model_from_snowflake", lambda model_uri, settings: model_path)

    loaded_model = model_service.load_model(
        "unused-local-model.joblib",
        "@CUSTOMER_INTELLIGENCE.ML_ARTIFACTS.MODEL_STAGE/models/random_forest_reorder_model.joblib",
        Settings(),
    )

    assert isinstance(loaded_model, DummyModel)
    assert model_service.model_state["feature_columns"] == ["total_orders", "avg_basket_size"]


def test_normalize_snowflake_stage_uri_accepts_stage_path() -> None:
    model_uri = "@CUSTOMER_INTELLIGENCE.ML_ARTIFACTS.MODEL_STAGE/models/random_forest_reorder_model.joblib"

    assert model_service.normalize_snowflake_stage_uri(model_uri) == model_uri


def test_normalize_snowflake_stage_uri_accepts_snowflake_uri() -> None:
    model_uri = "snowflake://CUSTOMER_INTELLIGENCE.ML_ARTIFACTS.MODEL_STAGE/models/random_forest_reorder_model.joblib"

    assert (
        model_service.normalize_snowflake_stage_uri(model_uri)
        == "@CUSTOMER_INTELLIGENCE.ML_ARTIFACTS.MODEL_STAGE/models/random_forest_reorder_model.joblib"
    )
