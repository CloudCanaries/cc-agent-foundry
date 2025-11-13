import os
import logging
from typing import Dict, List, Optional
from datetime import datetime


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


class JSONMixin:
    """Mixin to provide JSON serialization and deserialization methods."""

    @staticmethod
    def to_json(obj: object) -> str:
        """Convert an object to a JSON string."""
        import json

        return json.dumps(obj, default=str)

    @staticmethod
    def from_json(json_str: str) -> object:
        """Convert a JSON string back to an object."""
        import json

        return json.loads(json_str)

    def _json_default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return str(obj)


class ColorPaletteMixin:
    """
    Shared color palette + helpers for all worker agents.

    Agents can override `_palette_overrides` if a custom color is ever required,
    but the common palette stays consistent otherwise.
    """

    DEFAULT_COLOR_PALETTE: Dict[str, str] = {
        "green": "#16a34a",
        "yellow": "#f59e0b",
        "red": "#dc2626",
        "blue": "#3b82f6",
        "gray": "#6b7280",
    }

    def _palette_overrides(self) -> Optional[Dict[str, str]]:
        """Hook for subclasses needing extra named colors."""
        return None

    def _palette(self) -> Dict[str, str]:
        palette = {k.lower(): v for k, v in self.DEFAULT_COLOR_PALETTE.items()}
        overrides = self._palette_overrides()
        if overrides:
            palette.update({k.lower(): v for k, v in overrides.items()})
        return palette

    def _hex(self, color_name: Optional[str]) -> str:
        """
        Resolve a friendly color name (green/yellow/...) into a shared hex value.
        Accepts already-hex values and falls back to gray.
        """
        palette = self._palette()
        if not color_name:
            return palette["gray"]
        normalized = color_name.strip()
        if normalized.startswith("#"):
            return normalized
        return palette.get(normalized.lower(), palette["gray"])
