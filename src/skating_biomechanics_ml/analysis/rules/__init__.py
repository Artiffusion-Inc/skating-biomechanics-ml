"""Recommendation rules module."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class RecommendationRule:
    """A single recommendation rule.

    Attributes:
        metric_name: Name of the metric to check.
        condition: Callable that returns True if rule should trigger.
        priority: Lower = more critical (for sorting).
        templates: Mapping from severity to Russian text template.
                  Severity keys: "too_low", "too_high", "default".
                  Template variables: {value}, {unit}, {target_min}, {target_max}
    """

    metric_name: str
    condition: Callable[[float, tuple[float, float]], bool]
    priority: int
    templates: Mapping[str, str]


__all__ = ["RecommendationRule"]
