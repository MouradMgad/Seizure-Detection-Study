"""Shared experiment constants and validated configuration."""

from dataclasses import dataclass
import math


COMMON_BIPOLAR_CHANNELS = (
    "FP1-F7",
    "F7-T7",
    "T7-P7",
    "P7-O1",
    "FP1-F3",
    "F3-C3",
    "C3-P3",
    "P3-O1",
    "FP2-F4",
    "F4-C4",
    "C4-P4",
    "P4-O2",
    "FP2-F8",
    "F8-T8",
    "T8-P8",
    "P8-O2",
    "FZ-CZ",
    "CZ-PZ",
)


@dataclass(frozen=True)
class SignalConfig:
    """Continuous signal processing and windowing choices."""

    low_hz: float = 0.5
    high_hz: float = 45.0
    window_seconds: float = 4.0
    alignment: str = "none"
    alignment_reference_seconds: float = 60.0
    covariance_ridge: float = 1e-6

    def __post_init__(self) -> None:
        values = (
            self.low_hz,
            self.high_hz,
            self.window_seconds,
            self.alignment_reference_seconds,
            self.covariance_ridge,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Signal configuration values must all be finite.")
        if not 0 < self.low_hz < self.high_hz:
            raise ValueError("Require 0 < low_hz < high_hz.")
        if self.window_seconds <= 0:
            raise ValueError("window_seconds must be positive.")
        if self.alignment not in {"none", "euclidean"}:
            raise ValueError("alignment must be 'none' or 'euclidean'.")
        if self.alignment_reference_seconds <= 0:
            raise ValueError("alignment_reference_seconds must be positive.")
        if self.covariance_ridge <= 0:
            raise ValueError("covariance_ridge must be positive.")


@dataclass(frozen=True)
class AlarmConfig:
    """Validation-selected threshold and continuous alarm post-processing."""

    target_false_alarms_per_hour: float = 1.0
    minimum_alarm_seconds: float = 8.0
    merge_gap_seconds: float = 8.0
    threshold_candidates: int = 201

    def __post_init__(self) -> None:
        if not all(
            math.isfinite(value)
            for value in (
                self.target_false_alarms_per_hour,
                self.minimum_alarm_seconds,
                self.merge_gap_seconds,
            )
        ):
            raise ValueError("Alarm configuration values must all be finite.")
        if self.target_false_alarms_per_hour < 0:
            raise ValueError("target_false_alarms_per_hour cannot be negative.")
        if self.minimum_alarm_seconds <= 0 or self.merge_gap_seconds < 0:
            raise ValueError("Alarm duration must be positive and merge gap non-negative.")
        if self.threshold_candidates < 3:
            raise ValueError("threshold_candidates must be at least 3.")
