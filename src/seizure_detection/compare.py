"""Paired comparison of saved classical and EEGPT LOSO result files."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from seizure_detection.serialization import write_json


def _paired_difference(
    baseline_folds: list[dict[str, Any]],
    foundation_folds: list[dict[str, Any]],
    metric: str,
    samples: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    baseline = {item["subject_id"]: item[metric] for item in baseline_folds}
    foundation = {item["subject_id"]: item[metric] for item in foundation_folds}
    shared = sorted(set(baseline) & set(foundation))
    differences = np.asarray(
        [baseline[key] - foundation[key] for key in shared],
        dtype=float,
    )
    differences = differences[np.isfinite(differences)]
    if not differences.size:
        return {
            "classical_minus_eegpt_mean": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "n_paired_patients": 0,
        }
    draws = rng.choice(
        differences,
        size=(samples, len(differences)),
        replace=True,
    ).mean(axis=1)
    return {
        "classical_minus_eegpt_mean": float(differences.mean()),
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
        "n_paired_patients": int(len(differences)),
    }


def compare_chbmit_runs(
    classical_run_path: Path,
    foundation_run_path: Path,
    output_path: Path,
    bootstrap_samples: int = 2000,
    seed: int = 2026,
) -> dict[str, Any]:
    """Compare models only when both LOSO runs cover identical patients and protocol."""
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive.")
    classical = json.loads(classical_run_path.read_text(encoding="utf-8"))
    foundation = json.loads(foundation_run_path.read_text(encoding="utf-8"))
    if set(classical["subjects"]) != set(foundation["subjects"]):
        raise ValueError("The classical and EEGPT runs must use exactly the same held-out patients.")
    if set(classical["record_ids"]) != set(foundation["record_ids"]):
        raise ValueError("The classical and EEGPT runs must include exactly the same recordings.")
    if classical["signal_config"] != foundation["signal_config"]:
        raise ValueError("The runs use different signal preprocessing/alignment configurations.")
    if classical["alarm_config"] != foundation["alarm_config"]:
        raise ValueError("The runs use different alarm post-processing or threshold targets.")
    classical_protocol = classical.get("evaluation_protocol")
    foundation_protocol = foundation.get("evaluation_protocol")
    if not isinstance(classical_protocol, dict) or not isinstance(foundation_protocol, dict):
        raise ValueError(
            "Both runs must record the post-warm-up evaluation protocol; rerun older benchmarks."
        )
    if classical_protocol != foundation_protocol:
        raise ValueError("The runs use different warm-up/evaluation protocols.")
    if classical.get("target_data_policy") != foundation.get("target_data_policy"):
        raise ValueError("The runs use different target-patient data access policies.")
    rng = np.random.default_rng(seed)
    foundation_folds = foundation["folds"]
    models = sorted({row["model"] for row in classical["folds"]})
    comparisons: dict[str, Any] = {}
    for model in models:
        baseline_folds = [row for row in classical["folds"] if row["model"] == model]
        comparisons[model] = {
            metric: _paired_difference(
                baseline_folds,
                foundation_folds,
                metric,
                bootstrap_samples,
                rng,
            )
            for metric in ("event_sensitivity", "false_alarms_per_hour", "time_precision")
        }
    result = {
        "classical_run": str(classical_run_path.resolve()),
        "foundation_run": str(foundation_run_path.resolve()),
        "matched_subjects": sorted(classical["subjects"]),
        "matched_record_count": len(classical["record_ids"]),
        "difference_direction": "classical model minus EEGPT; lower false alarms/hour is better",
        "paired_patient_bootstrap_samples": bootstrap_samples,
        "paired_comparisons": comparisons,
        "claim_guardrail": (
            "Paired retrospective comparison only. EEGPT uses a midpoint bipolar montage "
            "mapping and is not a clinical validation."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_path, result)
    from seizure_detection.plotting import plot_comparison_run

    plot_comparison_run(output_path, output_path.parent / "figures")
    return result


def _metric_value(fold: dict[str, Any], metric: str) -> float | None:
    value = fold.get(metric)
    if value is None:
        return None
    numeric = float(value)
    return numeric if np.isfinite(numeric) else None


def _alignment_shift(fold: dict[str, Any]) -> float | None:
    diagnostic = fold.get("unlabeled_calibration_shift")
    if not isinstance(diagnostic, dict):
        return None
    value = diagnostic.get("mean_distance")
    if value is None:
        return None
    numeric = float(value)
    return numeric if np.isfinite(numeric) else None


def _validate_alignment_runs(
    no_alignment: dict[str, Any],
    euclidean: dict[str, Any],
) -> tuple[
    list[str],
    list[str],
    dict[str, dict[str, dict[str, Any]]],
    dict[str, dict[str, dict[str, Any]]],
]:
    """Validate matched LOSO runs and index each fold by model and patient."""
    for key in ("subjects", "record_ids", "discovered_records"):
        if sorted(no_alignment.get(key, [])) != sorted(euclidean.get(key, [])):
            raise ValueError(f"The runs must use identical {key.replace('_', ' ')}.")
    if not no_alignment.get("subjects"):
        raise ValueError("The CHB-MIT runs contain no held-out patients.")
    for key in ("data_root", "seed", "max_negative_ratio", "n_records", "n_windows", "n_features"):
        if no_alignment.get(key) != euclidean.get(key):
            raise ValueError(f"The runs differ in {key.replace('_', ' ')}.")
    if no_alignment.get("alarm_config") != euclidean.get("alarm_config"):
        raise ValueError("The runs use different alarm or threshold-selection settings.")
    no_evaluation = no_alignment.get("evaluation_protocol")
    ea_evaluation = euclidean.get("evaluation_protocol")
    if not isinstance(no_evaluation, dict) or not isinstance(ea_evaluation, dict):
        raise ValueError(
            "Both runs must record the post-warm-up evaluation protocol; "
            "rerun benchmarks created before warm-up exclusion was implemented."
        )
    no_common_protocol = {
        key: value
        for key, value in no_evaluation.items()
        if key != "euclidean_alignment_reference_seconds"
    }
    ea_common_protocol = {
        key: value
        for key, value in ea_evaluation.items()
        if key != "euclidean_alignment_reference_seconds"
    }
    if no_common_protocol != ea_common_protocol:
        raise ValueError("The runs use different calibration/evaluation window protocols.")

    no_signal = no_alignment.get("signal_config")
    ea_signal = euclidean.get("signal_config")
    if not isinstance(no_signal, dict) or not isinstance(ea_signal, dict):
        raise ValueError("Both runs must include signal_config metadata.")
    if no_signal.get("alignment") != "none" or ea_signal.get("alignment") != "euclidean":
        raise ValueError("Provide one no-alignment run and one Euclidean-alignment run.")
    no_other_signal = {key: value for key, value in no_signal.items() if key != "alignment"}
    ea_other_signal = {key: value for key, value in ea_signal.items() if key != "alignment"}
    if no_other_signal != ea_other_signal:
        raise ValueError("The runs differ in signal preprocessing or alignment parameters.")
    if (
        no_evaluation.get("warmup_seconds_excluded_per_record")
        != no_signal.get("alignment_reference_seconds")
    ):
        raise ValueError("The evaluation warm-up does not match the configured warm-up period.")
    if (
        no_evaluation.get("euclidean_alignment_reference_seconds") is not None
        or ea_evaluation.get("euclidean_alignment_reference_seconds")
        != ea_signal.get("alignment_reference_seconds")
    ):
        raise ValueError("The Euclidean alignment reference differs from its recorded protocol.")
    no_policy = no_alignment.get("target_data_policy")
    ea_policy = euclidean.get("target_data_policy")
    if not isinstance(no_policy, dict) or not isinstance(ea_policy, dict):
        raise ValueError("Both runs must record what target-patient data adaptation may use.")
    if (
        no_policy.get("target_labels_used_for_adaptation") is not False
        or ea_policy.get("target_labels_used_for_adaptation") is not False
    ):
        raise ValueError("Target-patient labels are not permitted for adaptation.")
    if no_policy.get("unlabeled_target_signal_used_for_adaptation") is not False:
        raise ValueError("The no-alignment baseline must not adapt using target-patient EEG.")
    if ea_policy.get("unlabeled_target_signal_used_for_adaptation") is not True:
        raise ValueError("The Euclidean-alignment run must record unlabeled target EEG use.")
    if ea_policy.get("alignment_signal_interval_seconds") != [
        0.0,
        ea_signal.get("alignment_reference_seconds"),
    ]:
        raise ValueError("The alignment run's target-data interval does not match its configuration.")

    no_models = list(no_alignment.get("models", []))
    ea_models = list(euclidean.get("models", []))
    if not no_models or set(no_models) != set(ea_models):
        raise ValueError("The runs must include the same non-empty set of models.")
    subjects = sorted(no_alignment["subjects"])
    models = sorted(no_models)

    def index_folds(run: dict[str, Any], label: str) -> dict[str, dict[str, dict[str, Any]]]:
        indexed: dict[str, dict[str, dict[str, Any]]] = {model: {} for model in models}
        for fold in run.get("folds", []):
            model = fold.get("model")
            subject = fold.get("subject_id")
            if model not in indexed or subject not in subjects:
                raise ValueError(f"{label} run contains an unexpected model or patient fold.")
            if subject in indexed[model]:
                raise ValueError(f"{label} run contains a duplicate fold for {model}/{subject}.")
            for metric in ("event_sensitivity", "false_alarms_per_hour", "time_precision"):
                if metric not in fold:
                    raise ValueError(f"{label} fold {model}/{subject} is missing {metric}.")
            indexed[model][subject] = fold
        for model in models:
            if set(indexed[model]) != set(subjects):
                raise ValueError(f"{label} run does not have exactly one {model} fold per patient.")
        return indexed

    return (
        subjects,
        models,
        index_folds(no_alignment, "No-alignment"),
        index_folds(euclidean, "Euclidean-alignment"),
    )


def _bootstrap_mean_difference(
    rows: list[dict[str, Any]],
    metric: str,
    samples: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    values = np.asarray(
        [row[f"delta_euclidean_minus_none_{metric}"] for row in rows],
        dtype=float,
    )
    values = values[np.isfinite(values)]
    if not len(values):
        return {
            "mean_difference": None,
            "ci_low": None,
            "ci_high": None,
            "n_paired_patients": 0,
        }
    draws = rng.choice(values, size=(samples, len(values)), replace=True).mean(axis=1)
    return {
        "mean_difference": float(values.mean()),
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
        "n_paired_patients": int(len(values)),
    }


def compare_alignment_runs(
    no_alignment_run_path: Path,
    euclidean_run_path: Path,
    output_path: Path,
    bootstrap_samples: int = 2000,
    seed: int = 2026,
) -> dict[str, Any]:
    """Create paired patient-level metrics and diagnostics for matched CHB-MIT runs."""
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive.")
    no_alignment = json.loads(no_alignment_run_path.read_text(encoding="utf-8"))
    euclidean = json.loads(euclidean_run_path.read_text(encoding="utf-8"))
    subjects, models, no_folds, ea_folds = _validate_alignment_runs(
        no_alignment, euclidean
    )

    metrics = ("event_sensitivity", "false_alarms_per_hour", "time_precision")
    patient_rows: list[dict[str, Any]] = []
    for model in models:
        for subject in subjects:
            no_fold = no_folds[model][subject]
            ea_fold = ea_folds[model][subject]
            row: dict[str, Any] = {
                "model": model,
                "subject_id": subject,
                "no_alignment_shift_distance": _alignment_shift(no_fold),
                "euclidean_shift_distance": _alignment_shift(ea_fold),
            }
            for metric in metrics:
                no_value = _metric_value(no_fold, metric)
                ea_value = _metric_value(ea_fold, metric)
                row[f"no_alignment_{metric}"] = no_value
                row[f"euclidean_{metric}"] = ea_value
                row[f"delta_euclidean_minus_none_{metric}"] = (
                    ea_value - no_value
                    if ea_value is not None and no_value is not None
                    else None
                )
            patient_rows.append(row)

    rng = np.random.default_rng(seed)
    paired_comparisons = {
        model: {
            metric: _bootstrap_mean_difference(
                [row for row in patient_rows if row["model"] == model],
                metric,
                bootstrap_samples,
                rng,
            )
            for metric in metrics
        }
        for model in models
    }

    shift_summary: dict[str, Any] = {}
    for model in models:
        model_rows = [row for row in patient_rows if row["model"] == model]
        shift_summary[model] = {}
        for metric in metrics:
            paired = [
                row
                for row in model_rows
                if row["no_alignment_shift_distance"] is not None
                and row[f"delta_euclidean_minus_none_{metric}"] is not None
            ]
            x = np.asarray([row["no_alignment_shift_distance"] for row in paired], dtype=float)
            y = np.asarray(
                [row[f"delta_euclidean_minus_none_{metric}"] for row in paired],
                dtype=float,
            )
            if len(paired) >= 3 and np.unique(x).size > 1 and np.unique(y).size > 1:
                from scipy.stats import spearmanr

                rho = float(spearmanr(x, y).statistic)
            else:
                rho = None
            shift_summary[model][metric] = {
                "shift_metric": "unlabeled_calibration_shift.mean_distance",
                "spearman_rho_with_metric_change": rho,
                "n_paired_patients": len(paired),
            }

    shift_available = all(
        _alignment_shift(no_folds[model][subject]) is not None
        and _alignment_shift(ea_folds[model][subject]) is not None
        for model in models
        for subject in subjects
    )
    result = {
        "task": "Paired CHB-MIT Euclidean-alignment comparison",
        "no_alignment_run": str(no_alignment_run_path.resolve()),
        "euclidean_alignment_run": str(euclidean_run_path.resolve()),
        "matched_subjects": subjects,
        "matched_record_count": len(no_alignment["record_ids"]),
        "models": models,
        "difference_direction": (
            "Euclidean alignment minus no alignment; positive event-sensitivity "
            "and time-precision changes are favorable, negative false-alarms/hour "
            "changes are favorable."
        ),
        "paired_patient_bootstrap_samples": bootstrap_samples,
        "paired_comparisons": paired_comparisons,
        "unlabeled_shift_diagnostics_available": shift_available,
        "shift_association": shift_summary,
        "interpretation_note": (
            "Shift associations are exploratory and use unlabeled calibration-window "
            "feature means standardized by source-training feature standard deviations. "
            "They are not causal evidence or a validated adaptation policy."
        ),
        "claim_guardrail": (
            "Retrospective patient-held-out comparison only. Euclidean alignment uses "
            "unlabeled initial data from each held-out recording; it is not a causal "
            "streaming evaluation."
        ),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    table_path = output_path.parent / "patient_differences.csv"
    fieldnames = [
        "model",
        "subject_id",
        "no_alignment_shift_distance",
        "euclidean_shift_distance",
        *[
            column
            for metric in metrics
            for column in (
                f"no_alignment_{metric}",
                f"euclidean_{metric}",
                f"delta_euclidean_minus_none_{metric}",
            )
        ],
    ]
    with table_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(patient_rows)
    result["patient_differences_csv"] = str(table_path.resolve())
    write_json(output_path, result)
    from seizure_detection.plotting import plot_alignment_comparison

    figures = plot_alignment_comparison(output_path, output_path.parent / "figures")
    result["figures"] = [str(path.resolve()) for path in figures]
    write_json(output_path, result)
    return result
