"""CHB-MIT discovery, annotation parsing, and strict channel preparation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np

from seizure_detection.config import COMMON_BIPOLAR_CHANNELS, SignalConfig


@dataclass(frozen=True)
class EEGRecord:
    """One continuous recording in microvolts and seconds."""

    record_id: str
    subject_id: str
    path: Path
    data_uv: np.ndarray
    sfreq: float
    seizure_intervals: tuple[tuple[float, float], ...]

    @property
    def duration_seconds(self) -> float:
        return self.data_uv.shape[1] / self.sfreq


def normalize_channel_name(name: str) -> str:
    """Normalize CHB-MIT naming variants while retaining bipolar order."""
    clean = re.sub(r"\s+", "", name.upper().replace("EEG", ""))
    clean = clean.replace("－", "-").replace("_", "-")
    aliases = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}
    return "-".join(aliases.get(part, part) for part in clean.split("-"))


def parse_chbmit_summary(summary_path: Path) -> dict[str, tuple[tuple[float, float], ...]]:
    """Parse seizure start/end times from a PhysioNet CHB-MIT summary."""
    if not summary_path.is_file():
        raise FileNotFoundError(f"CHB-MIT summary file not found: {summary_path}")
    output: dict[str, list[tuple[float, float]]] = {}
    current_file: str | None = None
    starts: dict[int, float] = {}
    ends: dict[int, float] = {}

    def save_current() -> None:
        if current_file is None:
            return
        if starts.keys() != ends.keys():
            raise ValueError(f"Incomplete seizure annotations for {current_file} in {summary_path}")
        intervals = sorted((starts[index], ends[index]) for index in starts)
        for start, end in intervals:
            if not 0 <= start < end:
                raise ValueError(f"Invalid seizure interval [{start}, {end}] in {current_file}")
        output[current_file] = intervals

    for raw_line in summary_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        file_match = re.match(r"File Name:\s*(\S+)", line, flags=re.IGNORECASE)
        if file_match:
            save_current()
            current_file = file_match.group(1)
            starts, ends = {}, {}
            continue
        if current_file is None:
            continue
        match = re.match(r"Seizure\s+(\d+)\s+(Start|End)\s+Time:\s*(\d+)", line, re.I)
        if match:
            index = int(match.group(1))
            target = starts if match.group(2).lower() == "start" else ends
            target[index] = float(match.group(3))
    save_current()
    return {name: tuple(intervals) for name, intervals in output.items()}


def discover_chbmit(data_root: Path) -> list[tuple[Path, str, dict[str, tuple[tuple[float, float], ...]]]]:
    """Find EDFs and associated subject-level seizure annotations."""
    root = data_root.expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"CHB-MIT data root does not exist: {root}")
    discovered = []
    for subject_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        summaries = sorted(subject_dir.glob("*-summary.txt"))
        if len(summaries) != 1:
            raise FileNotFoundError(
                f"Expected one *-summary.txt in {subject_dir}; found {len(summaries)}."
            )
        annotations = parse_chbmit_summary(summaries[0])
        for edf_path in sorted(subject_dir.glob("*.edf")):
            if edf_path.name not in annotations:
                raise ValueError(
                    f"{edf_path.name} is missing from annotation summary {summaries[0]}."
                )
            discovered.append((edf_path, subject_dir.name, annotations))
    if not discovered:
        raise FileNotFoundError(f"No EDF files found below {root}.")
    return discovered


def load_chbmit_record(
    edf_path: Path,
    subject_id: str,
    annotations: dict[str, tuple[tuple[float, float], ...]],
    signal_config: SignalConfig,
    channels: tuple[str, ...] = COMMON_BIPOLAR_CHANNELS,
) -> EEGRecord:
    """Load, filter, and select a single EDF using MNE.

    Any missing requested channel is an error; silent channel substitution
    would alter the model input and invalidate the stated montage.
    """
    try:
        import mne
    except ImportError as exc:
        raise ImportError("EDF ingestion requires MNE; install the project dependencies.") from exc

    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose="ERROR")
    normalized_names = [normalize_channel_name(name) for name in raw.ch_names]
    if len(set(normalized_names)) != len(normalized_names):
        raise ValueError(f"{edf_path.name} has duplicate channels after name normalization.")
    available = dict(zip(normalized_names, raw.ch_names))
    missing = [name for name in channels if normalize_channel_name(name) not in available]
    if missing:
        raise ValueError(
            f"{edf_path.name} lacks required bipolar channels: {', '.join(missing)}. "
            "No channel interpolation is performed by the classical pipeline."
        )
    selected = [available[normalize_channel_name(name)] for name in channels]
    raw.pick(selected)
    if raw.ch_names != selected:
        raw.reorder_channels(selected)
    nyquist = raw.info["sfreq"] / 2
    high_hz = min(signal_config.high_hz, nyquist - 1.0)
    if high_hz <= signal_config.low_hz:
        raise ValueError(f"Sampling rate in {edf_path} is too low for the configured filter.")
    raw.filter(signal_config.low_hz, high_hz, picks="eeg", verbose="ERROR")
    values_uv = raw.get_data().astype(np.float32, copy=False) * 1e6
    if not np.isfinite(values_uv).all():
        raise ValueError(f"{edf_path.name} contains non-finite EEG samples.")
    duration = values_uv.shape[1] / raw.info["sfreq"]
    if edf_path.name not in annotations:
        raise ValueError(f"{edf_path.name} has no entry in its CHB-MIT annotation summary.")
    record_annotations = []
    for start, end in annotations.get(edf_path.name, ()):
        if end > duration + 1.0:
            raise ValueError(
                f"Annotation ending at {end}s exceeds {edf_path.name} duration ({duration:.1f}s)."
            )
        clipped_end = min(end, duration)
        if start >= clipped_end:
            raise ValueError(f"Seizure interval [{start}, {end}] is outside {edf_path.name}.")
        record_annotations.append((start, clipped_end))
    return EEGRecord(
        record_id=f"{subject_id}/{edf_path.stem}",
        subject_id=subject_id,
        path=edf_path,
        data_uv=values_uv,
        sfreq=float(raw.info["sfreq"]),
        seizure_intervals=tuple(record_annotations),
    )
