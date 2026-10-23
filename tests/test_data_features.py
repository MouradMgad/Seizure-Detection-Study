from pathlib import Path

import numpy as np

from seizure_detection.config import COMMON_BIPOLAR_CHANNELS, SignalConfig
from seizure_detection.data import EEGRecord, normalize_channel_name, parse_chbmit_summary
from seizure_detection.features import (
    _feature_vector,
    build_window_index,
    euclidean_alignment_matrix,
    extract_record_features,
)


def test_summary_parser_reads_multiple_files_and_zero_seizure_files(tmp_path: Path) -> None:
    summary = tmp_path / "chb01-summary.txt"
    summary.write_text(
        "File Name: chb01_01.edf\n"
        "Number of Seizures in File: 1\n"
        "Seizure 1 Start Time: 100 seconds\n"
        "Seizure 1 End Time: 120 seconds\n"
        "File Name: chb01_02.edf\n"
        "Number of Seizures in File: 0\n",
        encoding="utf-8",
    )
    parsed = parse_chbmit_summary(summary)
    assert parsed == {"chb01_01.edf": ((100.0, 120.0),), "chb01_02.edf": ()}


def test_channel_aliases_normalize_legacy_temporal_electrodes() -> None:
    assert normalize_channel_name("EEG T3 - T5") == "T7-P7"
    assert normalize_channel_name("Fp1_F7") == "FP1-F7"


def test_window_labels_exclude_onset_and_offset_crossing_windows() -> None:
    rng = np.random.default_rng(5)
    record = EEGRecord(
        record_id="chb01/chb01_01",
        subject_id="chb01",
        path=Path("chb01_01.edf"),
        data_uv=rng.normal(size=(len(COMMON_BIPOLAR_CHANNELS), 2000)).astype(np.float32),
        sfreq=100.0,
        seizure_intervals=((8.0, 16.0),),
    )
    X, y, starts, names = extract_record_features(
        record, SignalConfig(low_hz=0.5, high_hz=45.0, window_seconds=4.0)
    )
    assert starts.tolist() == [0.0, 4.0, 8.0, 12.0, 16.0]
    assert y.tolist() == [0, 0, 1, 1, 0]
    assert X.shape == (5, len(names))
    assert len(names) == 495
    assert np.isfinite(X).all()


def test_euclidean_alignment_is_finite_and_channel_square() -> None:
    rng = np.random.default_rng(3)
    data = rng.normal(size=(18, 10_000))
    matrix = euclidean_alignment_matrix(data, sfreq=100.0, reference_seconds=30)
    assert matrix.shape == (18, 18)
    assert np.isfinite(matrix).all()
    assert np.allclose(matrix, matrix.T, atol=1e-10)


def test_feature_names_align_with_feature_vector() -> None:
    rng = np.random.default_rng(42)
    features, names = _feature_vector(rng.normal(size=(1, 4097)), sfreq=173.61)
    assert features.shape == (len(names),)
    assert np.isfinite(features).all()


def test_window_index_matches_feature_extraction_without_computing_features() -> None:
    rng = np.random.default_rng(17)
    record = EEGRecord(
        record_id="chb03/chb03_01",
        subject_id="chb03",
        path=Path("chb03_01.edf"),
        data_uv=rng.normal(size=(18, 1200)).astype(np.float32),
        sfreq=100.0,
        seizure_intervals=((4.0, 8.0),),
    )
    config = SignalConfig()
    table = build_window_index([record], config)
    _, labels, starts, _ = extract_record_features(record, config)
    assert table.X.shape == (len(labels), 0)
    assert table.y.tolist() == labels.tolist()
    assert table.starts.tolist() == starts.tolist()
