import csv
from pathlib import Path

import numpy as np

import seizure_detection.benchmark as benchmark
from seizure_detection.config import AlarmConfig, COMMON_BIPOLAR_CHANNELS, SignalConfig
from seizure_detection.data import EEGRecord


def test_loso_benchmark_runs_nested_record_grouped_pipeline(tmp_path: Path, monkeypatch) -> None:
    data_root = tmp_path / "data"
    specifications = []
    records_by_path: dict[Path, EEGRecord] = {}
    for patient_index in range(3):
        subject = f"chb{patient_index + 1:02d}"
        for record_index in range(8):
            seizure_record = record_index < 4
            path = data_root / subject / f"{subject}_{record_index:02d}.edf"
            intervals = ((4.0, 12.0),) if seizure_record else ()
            rng = np.random.default_rng(patient_index * 100 + record_index)
            data = rng.normal(
                scale=15 if seizure_record else 5,
                size=(len(COMMON_BIPOLAR_CHANNELS), 1600),
            ).astype(np.float32)
            record = EEGRecord(
                record_id=f"{subject}/{path.stem}",
                subject_id=subject,
                path=path,
                data_uv=data,
                sfreq=100.0,
                seizure_intervals=intervals,
            )
            records_by_path[path] = record
            annotations = {path.name: intervals}
            specifications.append((path, subject, annotations))

    monkeypatch.setattr(benchmark, "discover_chbmit", lambda _root: specifications)
    monkeypatch.setattr(
        benchmark,
        "load_chbmit_record",
        lambda path, _subject, _annotations, _config: records_by_path[path],
    )
    result = benchmark.run_loso_benchmark(
        data_root,
        tmp_path / "results",
        SignalConfig(alignment_reference_seconds=4.0),
        AlarmConfig(
            target_false_alarms_per_hour=10.0,
            minimum_alarm_seconds=4.0,
            merge_gap_seconds=0.0,
        ),
        model_names=("logistic",),
        seed=7,
        bootstrap_samples=20,
    )
    assert result["n_patients"] == 3
    assert result["n_records"] == 24
    assert len(result["folds"]) == 3
    assert len(result["paired_patient_bootstrap_comparisons"]) == 0
    assert result["folds"][0]["unlabeled_calibration_shift"]["metric"] == (
        "standardized_feature_mean_distance"
    )
    assert result["folds"][0]["unlabeled_calibration_shift"]["n_calibration_windows"] > 0
    assert result["folds"][0]["n_calibration_windows_excluded"] == 8
    assert result["folds"][0]["n_test_windows"] == 24
    assert result["folds"][0]["evaluation_start_seconds"] == 4.0
    assert result["evaluation_protocol"]["warmup_seconds_excluded_per_record"] == 4.0
    with (tmp_path / "results" / "window_predictions.csv").open(
        newline="", encoding="utf-8"
    ) as predictions_file:
        prediction_rows = list(csv.DictReader(predictions_file))
    assert all(float(row["start_seconds"]) >= 4.0 for row in prediction_rows)
    assert (tmp_path / "results" / "run.json").is_file()
    assert (tmp_path / "results" / "window_predictions.csv").is_file()
    assert "NaN" not in (tmp_path / "results" / "run.json").read_text(encoding="utf-8")
