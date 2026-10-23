import json
from pathlib import Path

import pytest

from seizure_detection.compare import compare_alignment_runs


def _write_run(path: Path, alignment: str, event_sensitivity: list[float]) -> None:
    subjects = ["p1", "p2", "p3"]
    folds = []
    for index, subject in enumerate(subjects):
        folds.append(
            {
                "model": "logistic",
                "subject_id": subject,
                "event_sensitivity": event_sensitivity[index],
                "false_alarms_per_hour": [4.0, 3.0, 2.0][index],
                "time_precision": [0.1, 0.2, 0.3][index],
                "unlabeled_calibration_shift": {
                    "metric": "standardized_feature_mean_distance",
                    "mean_distance": ([1.0, 2.0, 3.0] if alignment == "none" else [0.5, 1.0, 1.5])[index],
                },
            }
        )
    path.write_text(
        json.dumps(
            {
                "data_root": "data/chbmit",
                "subjects": subjects,
                "record_ids": [f"{subject}/r1" for subject in subjects],
                "discovered_records": [f"{subject}/r1.edf" for subject in subjects],
                "seed": 2026,
                "max_negative_ratio": 5.0,
                "n_records": 3,
                "n_windows": 300,
                "n_features": 100,
                "models": ["logistic"],
                "signal_config": {
                    "alignment": alignment,
                    "alignment_reference_seconds": 60.0,
                    "low_hz": 0.5,
                },
                "alarm_config": {"target_false_alarms_per_hour": 1.0},
                "evaluation_protocol": {
                    "warmup_seconds_excluded_per_record": 60.0,
                    "euclidean_alignment_reference_seconds": (
                        60.0 if alignment == "euclidean" else None
                    ),
                    "evaluation_windows": "post-warm-up only",
                },
                "target_data_policy": {
                    "target_labels_used_for_adaptation": False,
                    "unlabeled_target_signal_used_for_adaptation": alignment == "euclidean",
                    "alignment_signal_interval_seconds": (
                        [0.0, 60.0] if alignment == "euclidean" else None
                    ),
                    "calibration_interval_may_contain_seizures_or_artifacts": (
                        alignment == "euclidean"
                    ),
                },
                "folds": folds,
            }
        ),
        encoding="utf-8",
    )


def test_compare_alignment_runs_writes_paired_table_intervals_and_figures(
    tmp_path: Path,
) -> None:
    no_alignment_path = tmp_path / "none.json"
    euclidean_path = tmp_path / "euclidean.json"
    _write_run(no_alignment_path, "none", [0.2, 0.5, 1.0])
    _write_run(euclidean_path, "euclidean", [0.3, 0.4, 0.8])

    output_path = tmp_path / "comparison.json"
    result = compare_alignment_runs(
        no_alignment_path,
        euclidean_path,
        output_path,
        bootstrap_samples=100,
        seed=4,
    )

    summary = result["paired_comparisons"]["logistic"]["event_sensitivity"]
    assert summary["mean_difference"] == pytest.approx(-1 / 15)
    assert summary["n_paired_patients"] == 3
    assert result["unlabeled_shift_diagnostics_available"] is True
    assert result["shift_association"]["logistic"]["event_sensitivity"][
        "spearman_rho_with_metric_change"
    ] == pytest.approx(-1.0)
    assert Path(result["patient_differences_csv"]).is_file()
    assert len(result["figures"]) == 2
    assert all(Path(path).is_file() for path in result["figures"])
    assert output_path.is_file()


def test_compare_alignment_runs_rejects_mismatched_alarm_settings(
    tmp_path: Path,
) -> None:
    no_alignment_path = tmp_path / "none.json"
    euclidean_path = tmp_path / "euclidean.json"
    _write_run(no_alignment_path, "none", [0.2, 0.5, 1.0])
    _write_run(euclidean_path, "euclidean", [0.3, 0.4, 0.8])
    euclidean = json.loads(euclidean_path.read_text(encoding="utf-8"))
    euclidean["alarm_config"]["target_false_alarms_per_hour"] = 2.0
    euclidean_path.write_text(json.dumps(euclidean), encoding="utf-8")

    with pytest.raises(ValueError, match="different alarm"):
        compare_alignment_runs(
            no_alignment_path,
            euclidean_path,
            tmp_path / "comparison.json",
            bootstrap_samples=10,
        )


def test_compare_alignment_runs_rejects_different_alignment_roles(
    tmp_path: Path,
) -> None:
    no_alignment_path = tmp_path / "none.json"
    euclidean_path = tmp_path / "euclidean.json"
    _write_run(no_alignment_path, "euclidean", [0.2, 0.5, 1.0])
    _write_run(euclidean_path, "euclidean", [0.3, 0.4, 0.8])

    with pytest.raises(ValueError, match="one no-alignment run"):
        compare_alignment_runs(
            no_alignment_path,
            euclidean_path,
            tmp_path / "comparison.json",
            bootstrap_samples=10,
        )
