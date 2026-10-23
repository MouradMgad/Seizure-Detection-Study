"""Nested, recording-grouped patient-level benchmark orchestration."""

from __future__ import annotations

import csv
from dataclasses import asdict
from pathlib import Path
import platform
import sys
from itertools import combinations
import math
from typing import Any

import numpy as np
import imblearn
import scipy
import sklearn
from sklearn.model_selection import GridSearchCV, GroupShuffleSplit, StratifiedGroupKFold

from seizure_detection.alarms import choose_threshold, score_table, window_metrics
from seizure_detection.config import AlarmConfig, SignalConfig
from seizure_detection.data import discover_chbmit, load_chbmit_record
from seizure_detection.features import FeatureTable, build_feature_table
from seizure_detection.models import MODEL_NAMES, make_model
from seizure_detection.serialization import write_json


def subset_table(table: FeatureTable, indices: np.ndarray) -> FeatureTable:
    """Slice windows while retaining metadata only for represented recordings."""
    mask = np.zeros(len(table.y), dtype=bool)
    mask[np.asarray(indices, dtype=int)] = True
    included = set(table.record_ids[mask])
    return FeatureTable(
        X=table.X[mask],
        y=table.y[mask],
        record_ids=table.record_ids[mask],
        subject_ids=table.subject_ids[mask],
        starts=table.starts[mask],
        ends=table.ends[mask],
        records={key: table.records[key] for key in included},
        feature_names=table.feature_names,
    )


def _validation_split(
    table: FeatureTable,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    y = table.y
    record_ids = table.record_ids
    splitter = GroupShuffleSplit(n_splits=100, test_size=0.2, random_state=seed)
    fallback: tuple[np.ndarray, np.ndarray] | None = None
    for train, validation in splitter.split(np.zeros(len(y)), y, groups=record_ids):
        if (
            np.unique(y[train]).size == 2
            and np.unique(y[validation]).size == 2
            and len(set(record_ids[train])) >= 2
        ):
            split = (train, validation)
            train_records = set(record_ids[train])
            validation_records = set(record_ids[validation])
            train_has_seizure_record = any(
                table.records[key].seizure_intervals for key in train_records
            )
            train_has_control_record = any(
                not table.records[key].seizure_intervals for key in train_records
            )
            validation_has_seizure_record = any(
                table.records[key].seizure_intervals for key in validation_records
            )
            validation_has_control_record = any(
                not table.records[key].seizure_intervals for key in validation_records
            )
            if (
                train_has_seizure_record
                and train_has_control_record
                and validation_has_seizure_record
                and validation_has_control_record
            ):
                return split
            if fallback is None:
                fallback = split
    if fallback is not None:
        return fallback
    raise ValueError(
        "Could not make a recording-disjoint validation split containing both classes. "
        "More seizure-bearing recordings or a better class-balanced cohort are required."
    )


def _grouped_cv(y: np.ndarray, record_ids: np.ndarray, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    unique_groups = np.unique(record_ids)
    if len(unique_groups) < 3:
        raise ValueError("At least three training recordings are required for grouped model selection.")
    n_splits = min(4, len(unique_groups))
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = []
    for train, validation in splitter.split(np.zeros(len(y)), y, groups=record_ids):
        if np.unique(y[train]).size == 2 and np.unique(y[validation]).size == 2:
            splits.append((train, validation))
    if len(splits) != n_splits:
        raise ValueError(
            "Every grouped hyperparameter fold must contain both classes; "
            "collect more class-balanced, independent recordings."
        )
    return splits


def _fit_model(
    name: str,
    X: np.ndarray,
    y: np.ndarray,
    record_ids: np.ndarray,
    seed: int,
    max_negative_ratio: float,
) -> tuple[Any, dict[str, Any]]:
    estimator, parameter_grid = make_model(name, seed, max_negative_ratio)
    cv = _grouped_cv(y, record_ids, seed)
    search = GridSearchCV(
        estimator,
        parameter_grid,
        scoring="average_precision",
        cv=cv,
        n_jobs=1,
        refit=True,
        error_score="raise",
    )
    search.fit(X, y)
    return search.best_estimator_, {
        "best_params": search.best_params_,
        "grouped_cv_average_precision": float(search.best_score_),
        "grouped_cv_splits": len(cv),
    }


def _predict_probability(estimator: Any, X: np.ndarray) -> np.ndarray:
    classes = list(estimator.named_steps["classifier"].classes_)
    if 1 not in classes:
        raise ValueError("Fitted estimator has no positive class.")
    return estimator.predict_proba(X)[:, classes.index(1)]


def _standardized_feature_mean_distance(
    source_features: np.ndarray,
    target_calibration_features: np.ndarray,
) -> float:
    """Measure unlabeled feature-mean shift in source-standard-deviation units."""
    if (
        source_features.ndim != 2
        or target_calibration_features.ndim != 2
        or not len(source_features)
        or not len(target_calibration_features)
        or source_features.shape[1] != target_calibration_features.shape[1]
    ):
        raise ValueError("Source and target calibration features must be non-empty and aligned.")
    source_scale = np.std(source_features, axis=0)
    source_scale = np.maximum(source_scale, 1e-8)
    standardized_delta = (
        np.mean(target_calibration_features, axis=0) - np.mean(source_features, axis=0)
    ) / source_scale
    return float(np.sqrt(np.mean(standardized_delta**2)))


def _aggregate_patient_bootstrap(
    fold_results: list[dict[str, Any]],
    n_bootstrap: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    if not fold_results:
        return {}
    rng = np.random.default_rng(seed)
    subjects = np.asarray([row["subject_id"] for row in fold_results], dtype=object)
    output: dict[str, dict[str, float]] = {}
    for metric in ("event_sensitivity", "false_alarms_per_hour", "time_precision"):
        values = np.asarray([row[metric] for row in fold_results], dtype=float)
        valid = np.isfinite(values)
        if not valid.any():
            output[metric] = {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
            continue
        valid_subjects = subjects[valid]
        valid_values = values[valid]
        unique_subjects = np.unique(valid_subjects)
        draws = np.empty(n_bootstrap, dtype=float)
        for index in range(n_bootstrap):
            sampled = rng.choice(unique_subjects, size=len(unique_subjects), replace=True)
            sampled_values = np.concatenate(
                [valid_values[valid_subjects == subject] for subject in sampled]
            )
            draws[index] = float(np.mean(sampled_values))
        output[metric] = {
            "mean": float(np.mean(valid_values)),
            "ci_low": float(np.quantile(draws, 0.025)),
            "ci_high": float(np.quantile(draws, 0.975)),
        }
    return output


def _paired_patient_comparisons(
    fold_results: list[dict[str, Any]],
    model_names: tuple[str, ...],
    n_bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    """Bootstrap paired model differences over matching held-out patients."""
    rng = np.random.default_rng(seed)
    output: dict[str, Any] = {}
    by_model = {
        model: {row["subject_id"]: row for row in fold_results if row["model"] == model}
        for model in model_names
    }
    for left, right in combinations(model_names, 2):
        paired_subjects = sorted(set(by_model[left]) & set(by_model[right]))
        comparison: dict[str, Any] = {"n_paired_patients": len(paired_subjects)}
        for metric in ("event_sensitivity", "false_alarms_per_hour", "time_precision"):
            differences = np.asarray(
                [
                    by_model[left][subject][metric] - by_model[right][subject][metric]
                    for subject in paired_subjects
                ],
                dtype=float,
            )
            differences = differences[np.isfinite(differences)]
            if not differences.size:
                comparison[metric] = {
                    "mean_difference": float("nan"),
                    "ci_low": float("nan"),
                    "ci_high": float("nan"),
                    "n_patients": 0,
                }
                continue
            draws = rng.choice(
                differences,
                size=(n_bootstrap, len(differences)),
                replace=True,
            ).mean(axis=1)
            comparison[metric] = {
                "mean_difference": float(differences.mean()),
                "ci_low": float(np.quantile(draws, 0.025)),
                "ci_high": float(np.quantile(draws, 0.975)),
                "n_patients": int(len(differences)),
            }
        output[f"{left}_minus_{right}"] = comparison
    return output


def run_loso_benchmark(
    data_root: Path,
    output_dir: Path,
    signal_config: SignalConfig,
    alarm_config: AlarmConfig,
    model_names: tuple[str, ...] = MODEL_NAMES,
    seed: int = 2026,
    bootstrap_samples: int = 2000,
    selected_subjects: tuple[str, ...] | None = None,
    max_negative_ratio: float = 5.0,
) -> dict[str, Any]:
    """Run outer LOSO tests with recording-grouped model and threshold selection."""
    if len(model_names) == 0 or set(model_names) - set(MODEL_NAMES):
        raise ValueError(f"Select model names from: {', '.join(MODEL_NAMES)}.")
    if (
        bootstrap_samples < 1
        or not math.isfinite(max_negative_ratio)
        or max_negative_ratio < 1
    ):
        raise ValueError("bootstrap_samples must be positive and max_negative_ratio at least 1.")
    discovered = discover_chbmit(data_root)
    if selected_subjects is not None:
        requested = set(selected_subjects)
        available = {subject for _, subject, _ in discovered}
        missing = sorted(requested - available)
        if missing:
            raise ValueError(f"Requested patients not found in data root: {', '.join(missing)}.")
        discovered = [row for row in discovered if row[1] in requested]
    record_stream = (
        load_chbmit_record(path, subject, annotations, signal_config)
        for path, subject, annotations in discovered
    )
    table = build_feature_table(record_stream, signal_config)
    subjects = np.unique(table.subject_ids)
    if len(subjects) < 3:
        raise ValueError("LOSO requires at least three patients for train/validation/test separation.")
    output_dir.mkdir(parents=True, exist_ok=True)
    fold_results: list[dict[str, Any]] = []
    prediction_path = output_dir / "window_predictions.csv"
    with prediction_path.open("w", newline="", encoding="utf-8") as prediction_file:
        writer = csv.DictWriter(
            prediction_file,
            fieldnames=[
                "model",
                "held_out_subject",
                "record_id",
                "start_seconds",
                "end_seconds",
                "label",
                "probability",
                "predicted_at_selected_threshold",
            ],
        )
        writer.writeheader()
        for model_name in model_names:
            for fold_number, subject in enumerate(subjects):
                test_indices = np.flatnonzero(table.subject_ids == subject)
                train_pool_indices = np.flatnonzero(table.subject_ids != subject)
                if np.unique(table.y[test_indices]).size < 1:
                    raise RuntimeError(f"Empty test fold for {subject}.")
                pool = subset_table(table, train_pool_indices)
                train_local, val_local = _validation_split(pool, seed + fold_number)
                train_table = subset_table(pool, train_local)
                validation_table = subset_table(pool, val_local)
                if len(set(train_table.record_ids)) < 2:
                    raise ValueError(f"Insufficient training recordings in fold {subject}.")

                tuned_estimator, tuning = _fit_model(
                    model_name,
                    train_table.X,
                    train_table.y,
                    train_table.record_ids,
                    seed + fold_number,
                    max_negative_ratio,
                )
                validation_evaluation_indices = np.flatnonzero(
                    validation_table.starts >= signal_config.alignment_reference_seconds
                )
                if not len(validation_evaluation_indices):
                    raise ValueError(
                        f"Validation recordings in fold {subject} have no windows after the "
                        f"{signal_config.alignment_reference_seconds:g}-second calibration "
                        "period."
                    )
                validation_evaluation_table = subset_table(
                    validation_table, validation_evaluation_indices
                )
                if np.unique(validation_evaluation_table.y).size != 2:
                    raise ValueError(
                        f"Validation recordings in fold {subject} do not contain both "
                        "window classes after warm-up exclusion; threshold selection "
                        "requires post-warm-up ictal and non-ictal data."
                    )
                validation_probabilities = _predict_probability(
                    tuned_estimator, validation_evaluation_table.X
                )
                threshold, validation_summary = choose_threshold(
                    validation_evaluation_table,
                    validation_probabilities,
                    alarm_config,
                    evaluation_start_seconds=signal_config.alignment_reference_seconds,
                )
                # GridSearchCV refits its best estimator on train_table; validation
                # and test therefore use the same fitted model.
                final_estimator = tuned_estimator
                test_table_all = subset_table(table, test_indices)
                calibration_indices = np.flatnonzero(
                    test_table_all.ends <= signal_config.alignment_reference_seconds
                )
                if len(calibration_indices):
                    calibration_shift = {
                        "metric": "standardized_feature_mean_distance",
                        "mean_distance": _standardized_feature_mean_distance(
                            train_table.X,
                            test_table_all.X[calibration_indices],
                        ),
                        "reference_seconds_per_record": (
                            signal_config.alignment_reference_seconds
                        ),
                        "n_calibration_windows": int(len(calibration_indices)),
                        "description": (
                            "RMS source-standard-deviation distance between source training "
                            "feature means and unlabeled target windows starting within the "
                            "per-record calibration interval. Diagnostic only; no labels used."
                        ),
                    }
                else:
                    calibration_shift = None
                evaluation_indices = np.flatnonzero(
                    test_table_all.starts >= signal_config.alignment_reference_seconds
                )
                if not len(evaluation_indices):
                    raise ValueError(
                        f"Held-out patient {subject} has no windows after the "
                        f"{signal_config.alignment_reference_seconds:g}-second calibration "
                        "period."
                    )
                test_table = subset_table(test_table_all, evaluation_indices)
                test_probabilities = _predict_probability(final_estimator, test_table.X)
                record_metrics, alarms_by_record = score_table(
                    test_table,
                    test_probabilities,
                    threshold,
                    alarm_config,
                    evaluation_start_seconds=signal_config.alignment_reference_seconds,
                )
                event_rows = [asdict(item) for item in record_metrics]
                seizure_count = sum(row["total_events"] for row in event_rows)
                detected_count = sum(row["detected_events"] for row in event_rows)
                nonseizure_seconds = sum(
                    max(
                        row["duration_seconds"]
                        - sum(
                            end - max(start, signal_config.alignment_reference_seconds)
                            for start, end in test_table.records[row["record_id"]].seizure_intervals
                            if end > signal_config.alignment_reference_seconds
                        ),
                        0.0,
                    )
                    for row in event_rows
                )
                summary = {
                    "model": model_name,
                    "subject_id": str(subject),
                    "n_test_records": len(test_table.records),
                    "n_test_windows": int(len(test_table.y)),
                    "n_calibration_windows_excluded": int(len(calibration_indices)),
                    "evaluation_start_seconds": signal_config.alignment_reference_seconds,
                    "n_test_seizures": int(seizure_count),
                    "event_sensitivity": detected_count / seizure_count if seizure_count else float("nan"),
                    "false_alarms": sum(row["false_alarms"] for row in event_rows),
                    "false_alarms_per_hour": (
                        sum(row["false_alarms"] for row in event_rows)
                        / (nonseizure_seconds / 3600)
                        if nonseizure_seconds
                        else 0.0
                    ),
                    "time_precision": (
                        sum(
                            row["time_precision"] * sum(end - start for start, end in alarms_by_record[row["record_id"]])
                            for row in event_rows
                        )
                        / sum(
                            sum(end - start for start, end in alarms)
                            for alarms in alarms_by_record.values()
                        )
                        if any(alarms_by_record.values())
                        else 0.0
                    ),
                    "selected_threshold": float(threshold),
                    "validation_threshold_summary": validation_summary,
                    "unlabeled_calibration_shift": calibration_shift,
                    "window_metrics": window_metrics(
                        test_table.y, test_probabilities, threshold=threshold
                    ),
                    "tuning": tuning,
                    "records": event_rows,
                    "alarm_intervals": alarms_by_record,
                }
                fold_results.append(summary)
                for record_id, start, end, label, probability in zip(
                    test_table.record_ids,
                    test_table.starts,
                    test_table.ends,
                    test_table.y,
                    test_probabilities,
                ):
                    writer.writerow(
                        {
                            "model": model_name,
                            "held_out_subject": subject,
                            "record_id": record_id,
                            "start_seconds": f"{start:.3f}",
                            "end_seconds": f"{end:.3f}",
                            "label": int(label),
                            "probability": f"{probability:.8f}",
                            "predicted_at_selected_threshold": int(probability >= threshold),
                        }
                    )
                print(
                    f"{model_name} | held out {subject}: "
                    f"event sensitivity={summary['event_sensitivity']:.3f}, "
                    f"false alarms/hour={summary['false_alarms_per_hour']:.3f}"
                )

    grouped_results: dict[str, Any] = {}
    for model_name in model_names:
        rows = [item for item in fold_results if item["model"] == model_name]
        grouped_results[model_name] = _aggregate_patient_bootstrap(
            rows, bootstrap_samples, seed
        )
    try:
        import mne

        mne_version = mne.__version__
    except ImportError:
        mne_version = "not-installed"
    run = {
        "project_version": "0.1.0",
        "python_version": sys.version,
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "scikit_learn_version": sklearn.__version__,
        "imbalanced_learn_version": imblearn.__version__,
        "mne_version": mne_version,
        "data_root": str(data_root.expanduser().resolve()),
        "discovered_records": [str(path.relative_to(data_root.resolve())) for path, _, _ in discovered],
        "subjects": [str(subject) for subject in subjects],
        "record_ids": sorted(table.records),
        "requested_subjects": list(selected_subjects) if selected_subjects else None,
        "n_patients": len(subjects),
        "n_records": len(discovered),
        "n_windows": len(table.y),
        "n_positive_windows": int(table.y.sum()),
        "n_features": int(table.X.shape[1]),
        "seed": seed,
        "max_negative_ratio": max_negative_ratio,
        "signal_config": asdict(signal_config),
        "alarm_config": asdict(alarm_config),
        "models": list(model_names),
        "grouping": "outer patient LOSO; validation and inner CV grouped by EDF recording",
        "target_data_policy": {
            "target_labels_used_for_adaptation": False,
            "unlabeled_target_signal_used_for_adaptation": (
                signal_config.alignment == "euclidean"
            ),
            "alignment_signal_interval_seconds": (
                [0.0, signal_config.alignment_reference_seconds]
                if signal_config.alignment == "euclidean"
                else None
            ),
            "calibration_interval_may_contain_seizures_or_artifacts": (
                signal_config.alignment == "euclidean"
            ),
        },
        "evaluation_protocol": {
            "warmup_seconds_excluded_per_record": signal_config.alignment_reference_seconds,
            "euclidean_alignment_reference_seconds": (
                signal_config.alignment_reference_seconds
                if signal_config.alignment == "euclidean"
                else None
            ),
            "evaluation_windows": (
                "Only windows whose start time is at or after the per-record "
                "warm-up interval are evaluated."
            ),
            "event_handling": (
                "Events beginning before the evaluation interval are excluded from "
                "event sensitivity; their post-warm-up overlap is excluded from "
                "non-seizure exposure."
            ),
            "filtering": "Retrospective zero-phase filtering; not causal streaming.",
        },
        "aggregate_patient_bootstrap": grouped_results,
        "paired_patient_bootstrap_comparisons": _paired_patient_comparisons(
            fold_results, model_names, bootstrap_samples, seed + 1
        ),
        "folds": fold_results,
    }
    write_json(output_dir / "run.json", run)
    from seizure_detection.plotting import plot_chbmit_run

    plot_chbmit_run(output_dir / "run.json", output_dir / "figures")
    return run
