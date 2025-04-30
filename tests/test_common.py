import os
import pytest
from unittest.mock import patch, MagicMock
from canary.common import CanaryBasePrototype, CanaryRunStatus


@pytest.fixture
def canary_instance():
    """Fixture to create a CanaryBasePrototype instance with mocked environment variables."""
    with patch.dict(
        os.environ,
        {
            "CANARY_API_SERVER": "http://example.com",
            "API_KEY": "test_api_key",
            "CANARY_ID": "test_canary_id",
        },
        clear=True,
    ):
        yield CanaryBasePrototype()


def test_initialization(canary_instance):
    """Test that the CanaryBasePrototype initializes correctly."""
    assert canary_instance.status == CanaryRunStatus.NEW
    assert canary_instance._uuid is not None
    assert not canary_instance._error_get
    assert not canary_instance._error_parse


def test_validate_env_vars(canary_instance):
    """Test that required environment variables are validated."""
    env_vars = canary_instance.validate_env_vars(
        ["CANARY_API_SERVER", "API_KEY", "CANARY_ID"], strict=False
    )
    assert env_vars["CANARY_API_SERVER"] == "http://example.com"
    assert env_vars["API_KEY"] == "test_api_key"
    assert env_vars["CANARY_ID"] == "test_canary_id"


def test_missing_env_vars():
    """Test that missing environment variables raise an error."""
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(
            EnvironmentError, match="Missing required environment variables"
        ):
            canary = CanaryBasePrototype()
            canary.validate_env_vars(
                ["CANARY_API_SERVER", "API_KEY", "CANARY_ID"], strict=False
            )


@patch("JSONCanaries.ccmodule.canary.common.logging.getLogger")
def test_logger_setup(mock_get_logger, canary_instance):
    """Test that the logger is set up correctly."""
    mock_logger = MagicMock()
    mock_get_logger.return_value = mock_logger

    canary_instance._set_logger_meta()

    mock_logger.addHandler.assert_called()
    assert mock_logger.hasHandlers()


def test_canary_properties(canary_instance):
    """Test that the CanaryBasePrototype properties return correct values."""
    assert canary_instance._api_key == "test_api_key"
    assert canary_instance._canaryServer == "http://example.com"
    assert canary_instance._canary_id == "test_canary_id"
