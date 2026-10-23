import json
from pathlib import Path

import pytest

from seizure_detection.compare import compare_chbmit_runs


def _write_run(path: Path, subjects: list[str], record_ids: list[str], folds: list[dict]) -> None:
    path.write_text(
        json.dumps(
            {
                "subjects": subjects,
                "record_ids": record_ids,
                "signal_config": {"alignment": "none"},
                "alarm_config": {"target_false_alarms_per_hour": 1.0},
                "evaluation_protocol": {
                    "warmup_seconds_excluded_per_record": 60.0,
                    "euclidean_alignment_reference_seconds": None,
                },
                "target_data_policy": {
                    "target_labels_used_for_adaptation": False,
                    "unlabeled_target_signal_used_for_adaptation": False,
                    "alignment_signal_interval_seconds": None,
                    "calibration_interval_may_contain_seizures_or_artifacts": False,
                },
                "folds": folds,
            }
        ),
        encoding="utf-8",
    )


def test_compare_chbmit_runs_requires_matching_cohorts_and_bootstraps_paired_differences(
    tmp_path: Path,
) -> None:
    classical_path = tmp_path / "classical.json"
    foundation_path = tmp_path / "foundation.json"
    _write_run(
        classical_path,
        ["p1", "p2"],
        ["p1/r1", "p2/r1"],
        [
            {
                "model": "logistic",
                "subject_id": "p1",
                "event_sensitivity": 0.5,
                "false_alarms_per_hour": 2.0,
                "time_precision": 0.2,
            },
            {
                "model": "logistic",
                "subject_id": "p2",
                "event_sensitivity": 1.0,
                "false_alarms_per_hour": 1.0,
                "time_precision": 0.4,
            },
        ],
    )
    _write_run(
        foundation_path,
        ["p1", "p2"],
        ["p1/r1", "p2/r1"],
        [
            {
                "model": "eegpt_finetuned",
                "subject_id": "p1",
                "event_sensitivity": 0.0,
                "false_alarms_per_hour": 1.0,
                "time_precision": 0.1,
            },
            {
                "model": "eegpt_finetuned",
                "subject_id": "p2",
                "event_sensitivity": 1.0,
                "false_alarms_per_hour": 2.0,
                "time_precision": 0.3,
            },
        ],
    )
    output_path = tmp_path / "comparison.json"
    result = compare_chbmit_runs(
        classical_path,
        foundation_path,
        output_path,
        bootstrap_samples=100,
        seed=1,
    )
    assert result["paired_comparisons"]["logistic"]["event_sensitivity"][
        "classical_minus_eegpt_mean"
    ] == pytest.approx(0.25)
    assert output_path.is_file()


def test_compare_chbmit_runs_rejects_different_subjects(tmp_path: Path) -> None:
    classical_path = tmp_path / "classical.json"
    foundation_path = tmp_path / "foundation.json"
    fold = [
        {
            "model": "x",
            "subject_id": "p1",
            "event_sensitivity": 1.0,
            "false_alarms_per_hour": 1.0,
            "time_precision": 1.0,
        }
    ]
    _write_run(classical_path, ["p1"], ["p1/r1"], fold)
    _write_run(foundation_path, ["p2"], ["p2/r1"], fold)
    with pytest.raises(ValueError, match="same held-out patients"):
        compare_chbmit_runs(
            classical_path, foundation_path, tmp_path / "comparison.json", bootstrap_samples=10
        )
