"""Window construction, label-free alignment, and compact EEG features."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable

import numpy as np
from scipy.linalg import eigh
from scipy.integrate import trapezoid
from scipy.signal import welch
from scipy.stats import kurtosis, skew

from seizure_detection.config import SignalConfig
from seizure_detection.data import EEGRecord


@dataclass(frozen=True)
class RecordMetadata:
    subject_id: str
    duration_seconds: float
    seizure_intervals: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class FeatureTable:
    """Tabular windows plus provenance required for grouped evaluation."""

    X: np.ndarray
    y: np.ndarray
    record_ids: np.ndarray
    subject_ids: np.ndarray
    starts: np.ndarray
    ends: np.ndarray
    records: dict[str, RecordMetadata]
    feature_names: tuple[str, ...]


def _trace_normalized_covariance(segment: np.ndarray, ridge: float) -> np.ndarray:
    covariance = segment @ segment.T / max(segment.shape[1], 1)
    trace = float(np.trace(covariance))
    if not np.isfinite(trace) or trace <= np.finfo(np.float64).eps:
        trace = float(covariance.shape[0])
    return covariance * (covariance.shape[0] / trace)


def euclidean_alignment_matrix(
    data_uv: np.ndarray,
    sfreq: float,
    reference_seconds: float = 60.0,
    ridge: float = 1e-6,
    window_seconds: float = 4.0,
) -> np.ndarray:
    """Estimate (R + ridge I)^(-1/2) from unlabeled initial-record covariance."""
    stop = min(data_uv.shape[1], max(1, int(round(reference_seconds * sfreq))))
    reference = data_uv[:, :stop]
    window_samples = max(1, int(round(window_seconds * sfreq)))
    covariances = [
        _trace_normalized_covariance(reference[:, start : start + window_samples], ridge)
        for start in range(0, stop, window_samples)
        if reference[:, start : start + window_samples].shape[1]
    ]
    cov = np.mean(covariances, axis=0) + ridge * np.eye(data_uv.shape[0])
    eigenvalues, eigenvectors = eigh(cov, check_finite=True)
    return (eigenvectors * (1.0 / np.sqrt(np.maximum(eigenvalues, ridge)))) @ eigenvectors.T


def align_record(record: EEGRecord, config: SignalConfig) -> np.ndarray:
    if not np.isfinite(record.data_uv).all():
        raise ValueError(f"{record.record_id} contains non-finite EEG samples.")
    if config.alignment == "none":
        return record.data_uv
    matrix = euclidean_alignment_matrix(
        record.data_uv,
        record.sfreq,
        config.alignment_reference_seconds,
        config.covariance_ridge,
        config.window_seconds,
    )
    return (matrix @ record.data_uv).astype(np.float32, copy=False)


def _feature_vector(segment: np.ndarray, sfreq: float) -> tuple[np.ndarray, tuple[str, ...]]:
    """Extract time, band-power, and log-covariance features from one window."""
    n_channels, n_samples = segment.shape
    centered = segment - segment.mean(axis=1, keepdims=True)
    std = np.sqrt(np.mean(centered**2, axis=1))
    derivative = np.diff(segment, axis=1)
    derivative_std = np.sqrt(np.mean(derivative**2, axis=1)) * np.sqrt(sfreq)
    second_std = np.sqrt(np.mean(np.diff(segment, n=2, axis=1) ** 2, axis=1)) * sfreq
    mobility = derivative_std / np.maximum(std, 1e-8)
    complexity = (second_std / np.maximum(derivative_std, 1e-8)) / np.maximum(mobility, 1e-8)
    per_channel = np.stack(
        [
            np.log(np.maximum(std, 1e-8)),
            np.log(np.maximum(np.sqrt(np.mean(segment**2, axis=1)), 1e-8)),
            np.log(np.maximum(np.mean(np.abs(derivative), axis=1), 1e-8)),
            skew(segment, axis=1, bias=False, nan_policy="omit"),
            kurtosis(segment, axis=1, fisher=True, bias=False, nan_policy="omit"),
            np.log(np.maximum(std**2, 1e-8)),
            np.log(np.maximum(mobility, 1e-8)),
            np.log(np.maximum(complexity, 1e-8)),
        ],
        axis=1,
    )

    frequencies, psd = welch(
        segment,
        fs=sfreq,
        nperseg=min(n_samples, max(64, int(round(sfreq * 2)))),
        axis=1,
        detrend="constant",
        scaling="density",
    )
    bands = ((0.5, 4.0), (4.0, 8.0), (8.0, 13.0), (13.0, 30.0), (30.0, 45.0))
    band_power = np.stack(
        [
            trapezoid(
                psd[:, (frequencies >= low) & (frequencies < high)],
                frequencies[(frequencies >= low) & (frequencies < high)],
                axis=1,
            )
            for low, high in bands
        ],
        axis=1,
    )
    total_power = np.maximum(band_power.sum(axis=1, keepdims=True), 1e-12)
    spectral = np.concatenate(
        [np.log(np.maximum(band_power, 1e-12)), band_power / total_power],
        axis=1,
    )

    covariance = centered @ centered.T / max(n_samples - 1, 1)
    ridge = max(float(np.trace(covariance) / n_channels) * 1e-6, 1e-10)
    covariance += ridge * np.eye(n_channels)
    eigenvalues, eigenvectors = eigh(covariance, check_finite=True)
    log_cov = (eigenvectors * np.log(np.maximum(eigenvalues, ridge))) @ eigenvectors.T
    upper = np.triu_indices(n_channels)
    covariance_features = log_cov[upper].copy()
    covariance_features[upper[0] != upper[1]] *= np.sqrt(2.0)

    feature_names = tuple(
        [f"ch{channel}_time{stat}" for channel in range(n_channels) for stat in
         ("log_std", "log_rms", "log_line_length", "skew", "excess_kurtosis",
          "log_activity", "log_hjorth_mobility", "log_hjorth_complexity")]
        + [f"ch{channel}_{kind}_band_{band}" for channel in range(n_channels)
           for kind in ("log", "relative") for band in
           ("delta", "theta", "alpha", "beta", "gamma")]
        + [f"logcov_{left}_{right}" for left, right in zip(*upper)]
    )
    vector = np.concatenate([per_channel.ravel(), spectral.ravel(), covariance_features])
    vector = np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return vector, feature_names


def extract_record_features(
    record: EEGRecord,
    config: SignalConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
    """Return valid windows, labels, and timestamps for one record."""
    data = align_record(record, config)
    window_samples = int(round(config.window_seconds * record.sfreq))
    features: list[np.ndarray] = []
    labels: list[int] = []
    starts: list[float] = []
    names: tuple[str, ...] = ()
    for index, start, label in iter_labeled_windows(record, config):
        left = index * window_samples
        feature, names = _feature_vector(data[:, left : left + window_samples], record.sfreq)
        features.append(feature)
        labels.append(label)
        starts.append(start)
    if not features:
        raise ValueError(f"No complete non-ambiguous windows found in {record.record_id}.")
    return (
        np.stack(features),
        np.asarray(labels, dtype=np.int8),
        np.asarray(starts, dtype=np.float64),
        names,
    )


def iter_labeled_windows(
    record: EEGRecord,
    config: SignalConfig,
) -> Iterable[tuple[int, float, int]]:
    """Yield complete window indices, start times, and unambiguous labels."""
    window_samples = int(round(config.window_seconds * record.sfreq))
    count = record.data_uv.shape[1] // window_samples
    for index in range(count):
        start = index * config.window_seconds
        end = start + config.window_seconds
        overlaps = [
            (seizure_start, seizure_end)
            for seizure_start, seizure_end in record.seizure_intervals
            if start < seizure_end and end > seizure_start
        ]
        fully_ictal = any(
            start >= seizure_start and end <= seizure_end
            for seizure_start, seizure_end in overlaps
        )
        if overlaps and not fully_ictal:
            continue
        yield index, start, int(fully_ictal)


def build_feature_table(
    records: Iterable[EEGRecord],
    config: SignalConfig,
) -> FeatureTable:
    """Extract windows sequentially to avoid retaining whole EDFs in memory."""
    matrices: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    record_ids: list[np.ndarray] = []
    subject_ids: list[np.ndarray] = []
    starts: list[np.ndarray] = []
    ends: list[np.ndarray] = []
    metadata: dict[str, RecordMetadata] = {}
    all_names: tuple[str, ...] | None = None
    seen_records = 0
    for record in records:
        seen_records += 1
        X, y, window_starts, feature_names = extract_record_features(record, config)
        if all_names is not None and feature_names != all_names:
            raise RuntimeError("Feature schema changed between records.")
        all_names = feature_names
        matrices.append(X)
        labels.append(y)
        record_ids.append(np.full(y.shape, record.record_id, dtype=object))
        subject_ids.append(np.full(y.shape, record.subject_id, dtype=object))
        starts.append(window_starts)
        ends.append(window_starts + config.window_seconds)
        metadata[record.record_id] = RecordMetadata(
            record.subject_id,
            record.duration_seconds,
            record.seizure_intervals,
        )
    if not seen_records:
        raise ValueError("At least one EEG record is required.")
    assert all_names is not None
    return FeatureTable(
        X=np.concatenate(matrices),
        y=np.concatenate(labels),
        record_ids=np.concatenate(record_ids),
        subject_ids=np.concatenate(subject_ids),
        starts=np.concatenate(starts),
        ends=np.concatenate(ends),
        records=metadata,
        feature_names=all_names,
    )


def build_window_index(
    records: Iterable[EEGRecord],
    config: SignalConfig,
) -> FeatureTable:
    """Build only labels/timestamps for deep models, without computing tabular features."""
    labels: list[np.ndarray] = []
    record_ids: list[np.ndarray] = []
    subject_ids: list[np.ndarray] = []
    starts: list[np.ndarray] = []
    ends: list[np.ndarray] = []
    metadata: dict[str, RecordMetadata] = {}
    seen_records = 0
    for record in records:
        seen_records += 1
        windows = list(iter_labeled_windows(record, config))
        if not windows:
            raise ValueError(f"No complete non-ambiguous windows found in {record.record_id}.")
        window_labels = np.asarray([item[2] for item in windows], dtype=np.int8)
        window_starts = np.asarray([item[1] for item in windows], dtype=np.float64)
        labels.append(window_labels)
        record_ids.append(np.full(window_labels.shape, record.record_id, dtype=object))
        subject_ids.append(np.full(window_labels.shape, record.subject_id, dtype=object))
        starts.append(window_starts)
        ends.append(window_starts + config.window_seconds)
        metadata[record.record_id] = RecordMetadata(
            record.subject_id,
            record.duration_seconds,
            record.seizure_intervals,
        )
    if not seen_records:
        raise ValueError("At least one EEG record is required.")
    return FeatureTable(
        X=np.empty((sum(len(item) for item in labels), 0), dtype=np.float32),
        y=np.concatenate(labels),
        record_ids=np.concatenate(record_ids),
        subject_ids=np.concatenate(subject_ids),
        starts=np.concatenate(starts),
        ends=np.concatenate(ends),
        records=metadata,
        feature_names=(),
    )
