import os
import logging
from typing import List, Optional, Dict


class EnvVarValidatorMixin:
    """
    Mixin to validate required environment variables in any class.

    Usage:
        class MyService(EnvVarValidatorMixin):
            def __init__(self):
                self.validate_env_vars(['MY_API_KEY', 'MY_API_SECRET'])
    """

    def validate_env_vars(
        self, required_vars: List[str], strict: bool = False
    ) -> Dict[str, str]:
        """
        Validates that all required environment variables are set.

        Args:
            required_vars (List[str]): List of required environment variable names.
            strict (bool): If True, raises KeyError on missing variables (uses os.environ).
                           If False (default), raises EnvironmentError on missing (uses os.getenv).

        Returns:
            Dict[str, str]: Dictionary of variable names and their values.

        Raises:
            EnvironmentError: If any required variable is missing (in non-strict mode).
            KeyError: If any variable is missing (in strict mode).
        """
        result = {}
        missing = []

        for var in required_vars:
            value = os.environ[var] if strict else os.getenv(var)
            if value is None:
                missing.append(var)
            else:
                result[var] = value

        if missing:
            message = f"Missing required environment variables: {', '.join(missing)}"
            raise KeyError(message) if strict else EnvironmentError(message)

        return result


class LoggerMixin:
    """Mixin to provide consistent logging for classes."""

    @staticmethod
    def get_logger(name: str):
        logger = logging.getLogger(name)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(levelname)s %(asctime)s %(name)s.%(funcName)s:%(lineno)d - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False  # Prevent propagation to the root logger
        return logger
