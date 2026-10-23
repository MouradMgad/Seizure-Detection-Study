"""Continuous alarm generation, event metrics, and threshold calibration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from seizure_detection.config import AlarmConfig
from seizure_detection.features import FeatureTable, RecordMetadata


@dataclass(frozen=True)
class RecordMetrics:
    record_id: str
    subject_id: str
    event_sensitivity: float
    detected_events: int
    total_events: int
    false_alarms: int
    false_alarms_per_hour: float
    time_precision: float
    alarm_count: int
    duration_seconds: float


def alarm_intervals(
    starts: np.ndarray,
    ends: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    config: AlarmConfig,
) -> list[tuple[float, float]]:
    """Threshold windows, merge nearby positive runs, and discard short alarms."""
    order = np.argsort(starts)
    ordered_starts = np.asarray(starts)[order]
    ordered_ends = np.asarray(ends)[order]
    positives = np.asarray(probabilities)[order] >= threshold
    runs: list[list[float]] = []
    for start, end, positive in zip(ordered_starts, ordered_ends, positives):
        if not positive:
            continue
        if runs and start - runs[-1][1] <= config.merge_gap_seconds:
            runs[-1][1] = max(runs[-1][1], float(end))
        else:
            runs.append([float(start), float(end)])
    return [
        (start, end)
        for start, end in runs
        if end - start >= config.minimum_alarm_seconds
    ]


def score_record(
    record_id: str,
    metadata: RecordMetadata,
    starts: np.ndarray,
    ends: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    config: AlarmConfig,
    evaluation_start_seconds: float = 0.0,
) -> tuple[RecordMetrics, list[tuple[float, float]]]:
    if not np.isfinite(evaluation_start_seconds) or evaluation_start_seconds < 0:
        raise ValueError("evaluation_start_seconds must be finite and non-negative.")
    starts = np.asarray(starts)
    ends = np.asarray(ends)
    probabilities = np.asarray(probabilities)
    if starts.ndim != 1 or starts.shape != ends.shape or starts.shape != probabilities.shape:
        raise ValueError("Window starts, ends, and probabilities must be matching 1D arrays.")
    evaluation_start_seconds = min(evaluation_start_seconds, metadata.duration_seconds)
    evaluation_mask = starts >= evaluation_start_seconds
    alarms = alarm_intervals(
        starts[evaluation_mask],
        ends[evaluation_mask],
        probabilities[evaluation_mask],
        threshold,
        config,
    )
    all_evaluation_seizures = tuple(
        (max(start, evaluation_start_seconds), end)
        for start, end in metadata.seizure_intervals
        if end > evaluation_start_seconds
    )
    scored_events = tuple(
        (start, end)
        for start, end in metadata.seizure_intervals
        if start >= evaluation_start_seconds
    )
    detected = sum(
        any(
            alarm_start < seizure_end and alarm_end > seizure_start
            for alarm_start, alarm_end in alarms
        )
        for seizure_start, seizure_end in scored_events
    )
    false_alarms = sum(
        not any(
            alarm_start < seizure_end and alarm_end > seizure_start
            for seizure_start, seizure_end in all_evaluation_seizures
        )
        for alarm_start, alarm_end in alarms
    )
    alarm_seconds = sum(end - start for start, end in alarms)
    ictal_alarm_seconds = sum(
        max(0.0, min(alarm_end, seizure_end) - max(alarm_start, seizure_start))
        for alarm_start, alarm_end in alarms
        for seizure_start, seizure_end in all_evaluation_seizures
    )
    evaluation_duration = max(metadata.duration_seconds - evaluation_start_seconds, 0.0)
    non_seizure_seconds = max(
        evaluation_duration
        - sum(end - start for start, end in all_evaluation_seizures),
        0.0,
    )
    fa_per_hour = false_alarms / (non_seizure_seconds / 3600) if non_seizure_seconds else 0.0
    metrics = RecordMetrics(
        record_id=record_id,
        subject_id=metadata.subject_id,
        event_sensitivity=detected / len(scored_events) if scored_events else float("nan"),
        detected_events=detected,
        total_events=len(scored_events),
        false_alarms=false_alarms,
        false_alarms_per_hour=fa_per_hour,
        time_precision=ictal_alarm_seconds / alarm_seconds if alarm_seconds else 0.0,
        alarm_count=len(alarms),
        duration_seconds=evaluation_duration,
    )
    return metrics, alarms


def score_table(
    table: FeatureTable,
    probabilities: np.ndarray,
    threshold: float,
    config: AlarmConfig,
    evaluation_start_seconds: float = 0.0,
) -> tuple[list[RecordMetrics], dict[str, list[tuple[float, float]]]]:
    if len(probabilities) != len(table.y):
        raise ValueError("Probability count must equal the number of windows.")
    metrics: list[RecordMetrics] = []
    alarms_by_record: dict[str, list[tuple[float, float]]] = {}
    for record_id, metadata in table.records.items():
        mask = table.record_ids == record_id
        result, alarms = score_record(
            record_id,
            metadata,
            table.starts[mask],
            table.ends[mask],
            probabilities[mask],
            threshold,
            config,
            evaluation_start_seconds,
        )
        metrics.append(result)
        alarms_by_record[record_id] = alarms
    return metrics, alarms_by_record


def choose_threshold(
    table: FeatureTable,
    validation_probabilities: np.ndarray,
    config: AlarmConfig,
    evaluation_start_seconds: float = 0.0,
) -> tuple[float, dict[str, float | bool]]:
    """Choose threshold on validation recordings, never on the outer test set."""
    if len(validation_probabilities) != len(table.y):
        raise ValueError("Validation probability count does not match its feature table.")
    if not np.isfinite(validation_probabilities).all():
        raise ValueError("Validation probabilities must all be finite.")
    unique = np.unique(validation_probabilities)
    if len(unique) > config.threshold_candidates:
        candidates_from_scores = np.quantile(
            unique,
            np.linspace(0.0, 1.0, config.threshold_candidates),
        )
    else:
        candidates_from_scores = unique
    candidates = np.unique(
        np.concatenate(([0.0], candidates_from_scores, [1.000001]))
    )
    scored: list[tuple[float, float, float, float, float]] = []
    for threshold in candidates:
        per_record, alarms_by_record = score_table(
            table,
            validation_probabilities,
            float(threshold),
            config,
            evaluation_start_seconds,
        )
        event_records = [row for row in per_record if row.total_events]
        total_events = sum(row.total_events for row in event_records)
        detected = sum(row.detected_events for row in event_records)
        event_sensitivity = detected / total_events if total_events else 0.0
        nonseizure_seconds = sum(
            max(
                row.duration_seconds
                - sum(
                    end - max(start, evaluation_start_seconds)
                    for start, end in table.records[row.record_id].seizure_intervals
                    if end > evaluation_start_seconds
                ),
                0.0,
            )
            for row in per_record
        )
        fa_rate = (
            sum(row.false_alarms for row in per_record) / (nonseizure_seconds / 3600)
            if nonseizure_seconds
            else 0.0
        )
        alarm_seconds = sum(
            sum(end - start for start, end in record_alarms)
            for record_alarms in alarms_by_record.values()
        )
        ictal_seconds = sum(
            sum(
                max(0.0, min(alarm_end, seizure_end) - max(alarm_start, seizure_start))
                for alarm_start, alarm_end in alarms
                for seizure_start, seizure_end in table.records[record_id].seizure_intervals
            )
            for record_id, alarms in alarms_by_record.items()
        )
        time_precision = ictal_seconds / alarm_seconds if alarm_seconds else 0.0
        meets = fa_rate <= config.target_false_alarms_per_hour
        scored.append((float(meets), event_sensitivity, time_precision, float(threshold), fa_rate))
    eligible = [row for row in scored if row[0] == 1.0]
    pool = eligible if eligible else scored
    if eligible:
        best = max(pool, key=lambda row: (row[1], row[2], row[3]))
    else:
        best = max(pool, key=lambda row: (-row[4], row[1], row[2], row[3]))
    threshold = best[3]
    return threshold, {
        "event_sensitivity": best[1],
        "time_precision": best[2],
        "false_alarms_per_hour": best[4],
        "meets_false_alarm_target": bool(eligible),
    }


def window_metrics(
    y_true: np.ndarray, probabilities: np.ndarray, threshold: float = 0.5
) -> dict[str, float]:
    """Compute prevalence-aware secondary window metrics."""
    from sklearn.metrics import f1_score, precision_score, recall_score

    predicted = probabilities >= threshold
    return {
        "window_precision": float(precision_score(y_true, predicted, zero_division=0)),
        "window_recall": float(recall_score(y_true, predicted, zero_division=0)),
        "window_f1": float(f1_score(y_true, predicted, zero_division=0)),
        "window_auroc": float(roc_auc_score(y_true, probabilities))
        if np.unique(y_true).size == 2
        else float("nan"),
        "window_average_precision": float(average_precision_score(y_true, probabilities))
        if np.any(y_true)
        else float("nan"),
    }
