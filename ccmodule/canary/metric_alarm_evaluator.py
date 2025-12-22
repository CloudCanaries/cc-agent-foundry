import json
from typing import Optional
from .utils import LoggerMixin


class MetricAlarmEvaluatorMixin(LoggerMixin, object):
    """
    Shared alarm evaluation logic that mirrors backend ComparisonOperator behavior.
    Requires subclasses to provide _palette() from ColorPaletteMixin.
    """

    def __init__(self, *args, **kwargs):
        """Setup Logging"""
        super().__init__(*args, **kwargs)
        self._logger = self.get_logger(self.__class__.__name__)
        if hasattr(self, "_set_logger_meta"):
            try:
                self._set_logger_meta()
            except Exception:
                pass
        if not hasattr(self, "metric_alarm_configs") or not getattr(
            self, "metric_alarm_configs", None
        ):
            self.metric_alarm_configs = self._load_metric_alarm_configs()

    def _load_metric_alarm_configs(self):
        raw = getattr(self, "_env_metric_alarm_configs", None) or None
        if raw is None:
            import os

            raw = os.getenv("CANARY_METRIC_ALARM_CONFIGS")
        if not raw:
            return {}
        try:
            data = json.loads(raw)
            if isinstance(data, dict):
                return data
            self._logger.warning(
                "Invalid CANARY_METRIC_ALARM_CONFIGS payload (expected dict)"
            )
        except json.JSONDecodeError as exc:
            self._logger.warning("Failed to parse CANARY_METRIC_ALARM_CONFIGS: %s", exc)
        return {}

    def get_metric_alarm_config(self, metric_name: str):
        if not metric_name:
            return None
        return self.metric_alarm_configs.get(metric_name)

    def is_metric_alarm_enabled(self, metric_name: str) -> bool:
        """
        Return False only when a metric alarm exists and is explicitly disabled.
        Missing configs default to enabled behavior.
        """
        cfg = self.get_metric_alarm_config(metric_name) or {}
        if cfg and cfg.get("enabled") is False:
            return False
        return True

    def _coerce_value_for_comparison(self, value, data_type):
        if value is None:
            return None
        if data_type in ("integer", "decimal"):
            try:
                return float(value)
            except (TypeError, ValueError):
                return None
        if data_type == "boolean":
            if isinstance(value, str):
                return value.strip().lower() in ("true", "1", "yes")
            return bool(value)
        if data_type == "string":
            return str(value)
        return value

    def _is_boolean_operator(self, op: str) -> bool:
        return op in ("EqualTo", "NotEqualTo")

    def _is_string_operator(self, op: str) -> bool:
        return op in ("StringContainsCaseSensitive", "StringContainsNotCaseSensitive")

    def _is_json_operator(self, op: str) -> bool:
        return op == "ParsableJSON"

    def _uses_min_bound(self, op: str) -> bool:
        return "MinThreshold" in op

    def _uses_max_bound(self, op: str) -> bool:
        return "MaxThreshold" in op or self._is_boolean_operator(op)

    def _uses_greater_than(self, op: str) -> bool:
        return "GreaterThan" in op

    def _uses_less_than(self, op: str) -> bool:
        return "LessThan" in op

    def _includes_boundary_value(self, op: str) -> bool:
        return "OrEqualTo" in op

    def _eval_boolean(self, op: str, value, min_thr, max_thr) -> bool:
        _compared = value == min_thr and value == max_thr
        if "Not" in op:
            _compared = not _compared
        return _compared

    def _eval_string(self, op: str, value: str, min_thr: str, max_thr: str) -> bool:
        if not isinstance(value, str):
            return False
        if op == "StringContainsCaseSensitive":
            return (min_thr in value if min_thr else False) and (
                max_thr in value if max_thr else False
            )
        if op == "StringContainsNotCaseSensitive":
            lv = value.lower()
            checks = []
            if min_thr:
                checks.append(str(min_thr).lower() in lv)
            if max_thr:
                checks.append(str(max_thr).lower() in lv)
            return any(checks) if checks else False
        return False

    def _eval_json(self, value) -> bool:
        try:
            json.loads(json.dumps(value))
            return True
        except Exception:
            return False

    def _eval_numeric(self, op: str, value, min_thr, max_thr) -> bool:
        inclusive = self._includes_boundary_value(op)
        boundary = min_thr if self._uses_min_bound(op) else max_thr
        if boundary is None:
            return False

        def _as_number(v):
            if isinstance(v, (int, float)):
                return v
            try:
                return float(v)
            except Exception:
                return 0

        v = _as_number(value)
        b = _as_number(boundary)
        if self._uses_less_than(op):
            return v <= b if inclusive else v < b
        if self._uses_greater_than(op):
            return v >= b if inclusive else v > b
        return False

    def _eval_simple_comparison(
        self, op: str, value, min_thr, max_thr, *, data_type: Optional[str]
    ) -> bool:
        if data_type == "boolean" and self._is_boolean_operator(op):
            return self._eval_boolean(op, value, min_thr, max_thr)
        if data_type == "string" and self._is_string_operator(op):
            return self._eval_string(op, value, min_thr, max_thr)
        if data_type == "json" and self._is_json_operator(op):
            return self._eval_json(value)
        if op == "EqualTo":
            target = min_thr if min_thr is not None else max_thr
            return value == target
        if op == "NotEqualTo":
            target = min_thr if min_thr is not None else max_thr
            return value != target
        return self._eval_numeric(op, value, min_thr, max_thr)

    def _evaluate_comparison_operator(
        self, op: str, value, min_thr, max_thr, *, data_type: Optional[str]
    ) -> bool:
        if not op:
            return False
        if "_AND_" in op:
            left, right = op.split("_AND_", 1)
            return self._eval_simple_comparison(
                left, value, min_thr, max_thr, data_type=data_type
            ) and (
                self._eval_simple_comparison(
                    right, value, min_thr, max_thr, data_type=data_type
                )
            )
        if "_OR_" in op:
            left, right = op.split("_OR_", 1)
            return self._eval_simple_comparison(
                left, value, min_thr, max_thr, data_type=data_type
            ) or (
                self._eval_simple_comparison(
                    right, value, min_thr, max_thr, data_type=data_type
                )
            )
        return self._eval_simple_comparison(
            op, value, min_thr, max_thr, data_type=data_type
        )

    def evaluate_metric_alarm_status(self, metric_name: str, value):
        """
        Mirror backend alarm evaluation so agent-side health matches server alarms.
        Returns one of comparison_true_status/comparison_false_status/no_data_status or None if no config exists.
        """
        cfg = self.get_metric_alarm_config(metric_name) or {}
        if not cfg:
            return None
        if cfg.get("enabled") is False:
            return "NO_DATA"

        data_type = cfg.get("data_type")
        comparison_operator = cfg.get("comparison_operator")
        min_thr = cfg.get("min_threshold")
        max_thr = cfg.get("max_threshold")

        coerced_value = self._coerce_value_for_comparison(value, data_type)
        if coerced_value is None:
            return cfg.get("no_data_status", "NO_DATA")

        def _norm_threshold(thr):
            if thr is None:
                return None
            try:
                return float(thr)
            except (TypeError, ValueError):
                return thr

        min_thr = _norm_threshold(min_thr)
        max_thr = _norm_threshold(max_thr)

        result = self._evaluate_comparison_operator(
            comparison_operator, coerced_value, min_thr, max_thr, data_type=data_type
        )
        return (
            cfg.get("comparison_true_status", "OK")
            if result
            else cfg.get("comparison_false_status", "ALARMING")
        )

    def _color_from_alarm_status(self, status: Optional[str]) -> str:
        palette = self._palette()
        if status == "ALARMING":
            return palette["red"]
        if status == "OK":
            return palette["green"]
        if status == "NO_DATA":
            return palette["gray"]
        return palette["blue"]
