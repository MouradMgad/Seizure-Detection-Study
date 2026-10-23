"""Manifest-driven Bonn segment benchmark with nested grouped CV."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

import numpy as np
import imblearn
import sklearn
import scipy
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, StratifiedKFold

from seizure_detection.alarms import window_metrics
from seizure_detection.features import _feature_vector
from seizure_detection.models import MODEL_NAMES, make_model
from seizure_detection.serialization import write_json


def _label(value: str) -> int:
    normalized = value.strip().lower()
    if normalized in {"ictal", "seizure", "e", "s", "1"}:
        return 1
    if normalized in {
        "non-ictal",
        "interictal",
        "normal",
        "a",
        "b",
        "c",
        "d",
        "f",
        "n",
        "o",
        "z",
        "0",
    }:
        return 0
    raise ValueError(
        f"Unknown Bonn label {value!r}; use A-E or ictal/non-ictal (E is ictal)."
    )


def _read_signal(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        values = np.load(path, allow_pickle=False)
    elif path.suffix.lower() in {".txt", ".csv"}:
        values = np.loadtxt(path, delimiter="," if path.suffix.lower() == ".csv" else None)
    else:
        raise ValueError(f"Unsupported Bonn segment format {path.suffix!r} for {path}.")
    values = np.asarray(values, dtype=np.float64).squeeze()
    if values.ndim != 1 or not values.size or not np.isfinite(values).all():
        raise ValueError(f"Bonn signal must be a non-empty finite one-dimensional array: {path}.")
    return values[None, :]


def load_bonn_manifest(
    manifest_path: Path,
    sampling_rate: float = 173.61,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Load segments, binary labels, and manifest-supplied groups."""
    if sampling_rate <= 90:
        raise ValueError("sampling_rate must exceed 90 Hz for the configured 45 Hz feature band.")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Bonn manifest not found: {manifest_path}")
    X: list[np.ndarray] = []
    y: list[int] = []
    groups: list[str] = []
    paths: list[str] = []
    with manifest_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"path", "label", "group_id"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError("Bonn manifest must contain path,label,group_id columns.")
        for line_number, row in enumerate(reader, start=2):
            if not row.get("group_id", "").strip():
                raise ValueError(f"Missing source group_id at {manifest_path}:{line_number}.")
            signal_path = (manifest_path.parent / row["path"]).resolve()
            if not signal_path.is_file():
                raise FileNotFoundError(f"Manifest signal not found: {signal_path}")
            feature, _ = _feature_vector(_read_signal(signal_path), sampling_rate)
            X.append(feature)
            y.append(_label(row["label"]))
            groups.append(row["group_id"].strip())
            paths.append(str(signal_path))
    if not X:
        raise ValueError("Bonn manifest contains no segments.")
    if len(np.unique(groups)) < 5:
        raise ValueError(
            "Five outer grouped folds require at least five distinct manifest groups. "
            "For source-independent evaluation, group related segments by their true "
            "source recording."
        )
    return np.stack(X), np.asarray(y, dtype=np.int8), np.asarray(groups), paths


def run_bonn_benchmark(
    manifest_path: Path,
    output_dir: Path,
    sampling_rate: float = 173.61,
    model_names: tuple[str, ...] = MODEL_NAMES,
    seed: int = 2026,
    random_window_ablation: bool = False,
    max_negative_ratio: float = 5.0,
    data_provenance: str = "external_dataset",
    grouping_note: str = (
        "Grouped by manifest group_id; source-level validity depends on accurate "
        "source recording identifiers."
    ),
) -> dict[str, Any]:
    """Run five outer and four inner folds grouped by manifest group_id."""
    if set(model_names) - set(MODEL_NAMES) or not model_names:
        raise ValueError(f"Select model names from: {', '.join(MODEL_NAMES)}.")
    if not math.isfinite(max_negative_ratio) or max_negative_ratio < 1:
        raise ValueError("max_negative_ratio must be at least 1.")
    X, y, groups, paths = load_bonn_manifest(manifest_path, sampling_rate)
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    outer_splits = list(outer.split(X, y, groups))
    fold_results: list[dict[str, Any]] = []
    random_ablation: dict[str, list[dict[str, Any]]] = {}
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "segment_predictions.csv"
    with predictions_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["model", "fold", "path", "group_id", "label", "probability"],
        )
        writer.writeheader()
        for model_name in model_names:
            for fold_index, (train, test) in enumerate(outer_splits, start=1):
                if np.unique(y[train]).size < 2 or np.unique(y[test]).size < 2:
                    raise ValueError(
                        f"Outer Bonn fold {fold_index} does not contain both classes; "
                        "verify manifest group labels and cohort composition."
                    )
                inner_splitter = StratifiedGroupKFold(
                    n_splits=4, shuffle=True, random_state=seed + fold_index
                )
                inner = [
                    (inner_train, inner_valid)
                    for inner_train, inner_valid in inner_splitter.split(
                        X[train], y[train], groups[train]
                    )
                    if np.unique(y[train][inner_train]).size == 2
                    and np.unique(y[train][inner_valid]).size == 2
                ]
                if len(inner) != 4:
                    raise ValueError(
                        f"Outer Bonn fold {fold_index} cannot support four valid grouped "
                        "inner folds. More independent manifest groups per class are needed."
                    )
                estimator, grid = make_model(
                    model_name, seed + fold_index, max_negative_ratio
                )
                search = GridSearchCV(
                    estimator,
                    grid,
                    scoring="average_precision",
                    cv=inner,
                    n_jobs=1,
                    refit=True,
                    error_score="raise",
                )
                search.fit(X[train], y[train])
                classes = list(search.best_estimator_.named_steps["classifier"].classes_)
                probabilities = search.best_estimator_.predict_proba(X[test])[:, classes.index(1)]
                fold_results.append(
                    {
                        "model": model_name,
                        "fold": fold_index,
                        "n_train_segments": int(len(train)),
                        "n_test_segments": int(len(test)),
                        "n_test_groups": int(len(np.unique(groups[test]))),
                        "metrics": window_metrics(y[test], probabilities),
                        "confusion_matrix": confusion_matrix(
                            y[test], probabilities >= 0.5, labels=[0, 1]
                        ).tolist(),
                        "best_params": search.best_params_,
                        "inner_grouped_average_precision": float(search.best_score_),
                    }
                )
                for row_index, probability in zip(test, probabilities):
                    writer.writerow(
                        {
                            "model": model_name,
                            "fold": fold_index,
                            "path": paths[row_index],
                            "group_id": groups[row_index],
                            "label": int(y[row_index]),
                            "probability": f"{probability:.8f}",
                        }
                    )
            if random_window_ablation:
                random_outer = StratifiedKFold(
                    n_splits=5, shuffle=True, random_state=seed
                )
                random_rows = []
                for fold_index, (train, test) in enumerate(
                    random_outer.split(X, y), start=1
                ):
                    random_inner = StratifiedKFold(
                        n_splits=4, shuffle=True, random_state=seed + fold_index
                    )
                    estimator, grid = make_model(
                        model_name, seed + fold_index, max_negative_ratio
                    )
                    search = GridSearchCV(
                        estimator,
                        grid,
                        scoring="average_precision",
                        cv=random_inner,
                        n_jobs=1,
                        refit=True,
                        error_score="raise",
                    )
                    search.fit(X[train], y[train])
                    classes = list(search.best_estimator_.named_steps["classifier"].classes_)
                    probabilities = search.best_estimator_.predict_proba(X[test])[
                        :, classes.index(1)
                    ]
                    random_rows.append(
                        {
                            "fold": fold_index,
                            "metrics": window_metrics(y[test], probabilities),
                        }
                    )
                random_ablation[model_name] = random_rows
    result = {
        "task": "Bonn ictal versus non-ictal segment classification",
        "data_provenance": data_provenance,
        "manifest": str(manifest_path.resolve()),
        "sampling_rate_hz": sampling_rate,
        "n_segments": int(len(y)),
        "n_manifest_groups": int(len(np.unique(groups))),
        "positive_segments": int(y.sum()),
        "negative_segments": int((y == 0).sum()),
        "grouping": (
            "five outer folds grouped by manifest group_id; four inner grouped folds. "
            "The validity of source-recording generalization depends on the manifest "
            "containing true source recording identifiers."
        ),
        "grouping_note": grouping_note,
        "seed": seed,
        "max_negative_ratio": max_negative_ratio,
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "scikit_learn_version": sklearn.__version__,
        "imbalanced_learn_version": imblearn.__version__,
        "folds": fold_results,
        "random_window_leakage_ablation": random_ablation,
        "random_window_ablation_warning": (
            "Not a valid estimate of unseen-recording generalization; segments from one "
            "source group may occur in both train and test."
            if random_window_ablation
            else None
        ),
        "claim_guardrail": (
            "Bonn segment discrimination does not measure seizure onset timing or continuous "
            "alarm burden. Source-recording generalization is only supported when group_id "
            "contains true source recording identifiers."
        ),
    }
    write_json(output_dir / "bonn-run.json", result)
    from seizure_detection.plotting import plot_bonn_run

    plot_bonn_run(output_dir / "bonn-run.json", output_dir / "figures")
    return result
