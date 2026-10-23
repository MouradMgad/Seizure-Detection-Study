import csv
from pathlib import Path

import numpy as np
import pytest

from seizure_detection.alarms import alarm_intervals, choose_threshold, score_record
from seizure_detection.bonn import load_bonn_manifest, run_bonn_benchmark
from seizure_detection.config import AlarmConfig
from seizure_detection.features import FeatureTable, RecordMetadata


def test_alarm_runs_merge_gaps_and_drop_short_runs() -> None:
    alarms = alarm_intervals(
        starts=np.asarray([0.0, 4.0, 8.0, 20.0]),
        ends=np.asarray([4.0, 8.0, 12.0, 24.0]),
        probabilities=np.asarray([0.9, 0.8, 0.7, 0.99]),
        threshold=0.5,
        config=AlarmConfig(minimum_alarm_seconds=8, merge_gap_seconds=4),
    )
    assert alarms == [(0.0, 12.0)]


def test_event_and_alarm_metrics_separate_ictal_and_false_alarms() -> None:
    metrics, alarms = score_record(
        "patient/record",
        RecordMetadata("patient", 24.0, ((8.0, 16.0),)),
        starts=np.asarray([0.0, 4.0, 8.0, 16.0, 20.0]),
        ends=np.asarray([4.0, 8.0, 12.0, 20.0, 24.0]),
        probabilities=np.asarray([0.1, 0.1, 0.9, 0.1, 0.1]),
        threshold=0.5,
        config=AlarmConfig(minimum_alarm_seconds=4, merge_gap_seconds=0),
    )
    assert alarms == [(8.0, 12.0)]
    assert metrics.detected_events == 1
    assert metrics.false_alarms == 0
    assert metrics.event_sensitivity == 1.0
    assert metrics.time_precision == 1.0


def test_score_record_excludes_calibration_windows_and_events_from_metrics() -> None:
    metrics, alarms = score_record(
        "patient/record",
        RecordMetadata(
            "patient",
            120.0,
            ((20.0, 24.0), (55.0, 65.0), (70.0, 74.0)),
        ),
        starts=np.asarray([0.0, 56.0, 60.0, 64.0, 68.0, 72.0, 76.0, 84.0]),
        ends=np.asarray([4.0, 60.0, 64.0, 68.0, 72.0, 76.0, 80.0, 88.0]),
        probabilities=np.asarray([0.9, 0.9, 0.9, 0.1, 0.1, 0.9, 0.1, 0.9]),
        threshold=0.5,
        config=AlarmConfig(minimum_alarm_seconds=4, merge_gap_seconds=0),
        evaluation_start_seconds=60.0,
    )

    assert alarms == [(60.0, 64.0), (72.0, 76.0), (84.0, 88.0)]
    assert metrics.total_events == 1
    assert metrics.detected_events == 1
    assert metrics.false_alarms == 1
    assert metrics.duration_seconds == 60.0
    assert metrics.false_alarms_per_hour == pytest.approx(3600 / 51)


def test_threshold_is_selected_from_validation_records() -> None:
    table = FeatureTable(
        X=np.zeros((6, 1), dtype=np.float32),
        y=np.asarray([0, 1, 0, 0, 0, 0], dtype=np.int8),
        record_ids=np.asarray(["seizure", "seizure", "seizure", "control", "control", "control"]),
        subject_ids=np.asarray(["p1", "p1", "p1", "p2", "p2", "p2"]),
        starts=np.asarray([0, 4, 8, 0, 4, 8], dtype=float),
        ends=np.asarray([4, 8, 12, 4, 8, 12], dtype=float),
        records={
            "seizure": RecordMetadata("p1", 12, ((4.0, 8.0),)),
            "control": RecordMetadata("p2", 12, ()),
        },
        feature_names=("feature",),
    )
    threshold, summary = choose_threshold(
        table,
        np.asarray([0.1, 0.9, 0.1, 0.1, 0.9, 0.1]),
        AlarmConfig(
            target_false_alarms_per_hour=500,
            minimum_alarm_seconds=4,
            merge_gap_seconds=0,
        ),
    )
    assert threshold == 0.9
    assert summary["event_sensitivity"] == 1.0
    assert summary["meets_false_alarm_target"] is True


def test_bonn_manifest_requires_group_and_loads_segments(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "label", "group_id"])
        writer.writeheader()
        for index in range(5):
            signal_path = tmp_path / f"segment_{index}.txt"
            np.savetxt(signal_path, np.sin(np.linspace(0, 100, 4097) + index))
            writer.writerow(
                {
                    "path": signal_path.name,
                    "label": "E" if index < 2 else "A",
                    "group_id": f"source_{index}",
                }
            )
    X, y, groups, paths = load_bonn_manifest(manifest_path)
    assert X.shape[0] == 5
    assert y.tolist() == [1, 1, 0, 0, 0]
    assert len(np.unique(groups)) == 5
    assert len(paths) == 5


def test_bonn_nested_grouped_benchmark_writes_machine_readable_results(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "label", "group_id"])
        writer.writeheader()
        for group_index in range(10):
            is_ictal = group_index < 5
            for segment_index in range(2):
                path = tmp_path / f"g{group_index}_{segment_index}.txt"
                time = np.arange(4097) / 173.61
                frequency = 12.0 if is_ictal else 3.0
                signal = np.sin(2 * np.pi * frequency * time)
                signal += np.random.default_rng(group_index * 10 + segment_index).normal(
                    scale=0.05, size=len(time)
                )
                np.savetxt(path, signal)
                writer.writerow(
                    {
                        "path": path.name,
                        "label": "E" if is_ictal else "A",
                        "group_id": f"source_{group_index}",
                    }
                )
    output_dir = tmp_path / "results"
    result = run_bonn_benchmark(
        manifest_path,
        output_dir,
        model_names=("logistic",),
        seed=22,
        random_window_ablation=True,
    )
    assert len(result["folds"]) == 5
    assert result["n_manifest_groups"] == 10
    assert len(result["random_window_leakage_ablation"]["logistic"]) == 5
    assert "Not a valid estimate" in result["random_window_ablation_warning"]
    assert (output_dir / "bonn-run.json").is_file()
    assert (output_dir / "segment_predictions.csv").is_file()
    assert (output_dir / "figures" / "bonn-grouped-fold-metrics.png").is_file()
    assert (output_dir / "figures" / "bonn-confusion-matrices.png").is_file()


def test_bonn_original_set_labels_map_to_ictal_and_interictal() -> None:
    from seizure_detection.bonn import _label

    assert _label("S") == 1
    assert _label("F") == 0
