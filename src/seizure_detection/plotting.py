"""Publication-oriented figures for saved seizure-detection benchmark outputs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve

COLORS = ("#2369bd", "#e07a24", "#318a59", "#b04759", "#7655a5", "#1595a3")
DISPLAY_NAMES = {
    "logistic": "Logistic regression",
    "rbf_svm": "RBF-SVM",
    "random_forest": "Random forest",
    "extra_trees": "Extra Trees",
    "hist_gradient_boosting": "Hist. gradient boosting",
    "eegpt_finetuned": "EEGPT fine-tuned",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 180,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.alpha": 0.22,
        }
    )


def _label(name: str) -> str:
    return DISPLAY_NAMES.get(name, name.replace("_", " ").title())


def _fold_interval(values: list[Any]) -> tuple[float, float, float]:
    numeric = np.asarray(
        [np.nan if value is None else value for value in values],
        dtype=float,
    )
    numeric = numeric[np.isfinite(numeric)]
    if not numeric.size:
        return float("nan"), float("nan"), float("nan")
    return (
        float(numeric.mean()),
        float(np.quantile(numeric, 0.025)),
        float(np.quantile(numeric, 0.975)),
    )


def plot_chbmit_run(run_path: Path, output_dir: Path) -> list[Path]:
    """Plot patient-level operating points, bootstrap intervals, and window PR."""
    run = json.loads(run_path.read_text(encoding="utf-8"))
    folds = run.get("folds", [])
    if not folds or not any(
        "event_sensitivity" in fold or "event_sensitivity" in fold.get("metrics", {})
        for fold in folds
    ):
        raise ValueError(f"{run_path} does not contain continuous CHB-MIT folds.")
    _style()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    models = sorted({fold["model"] for fold in folds})

    figure, axis = plt.subplots(figsize=(8.2, 5.3), constrained_layout=True)
    for index, model in enumerate(models):
        rows = [fold for fold in folds if fold["model"] == model]
        x = np.asarray([row["false_alarms_per_hour"] for row in rows], dtype=float)
        y = np.asarray(
            [
                row.get("event_sensitivity", row.get("metrics", {}).get("event_sensitivity"))
                for row in rows
            ],
            dtype=float,
        )
        valid = np.isfinite(x) & np.isfinite(y)
        axis.scatter(
            x[valid],
            y[valid],
            s=42,
            alpha=0.64,
            color=COLORS[index % len(COLORS)],
            label=_label(model),
        )
        if valid.any():
            axis.scatter(
                [x[valid].mean()],
                [y[valid].mean()],
                marker="D",
                s=76,
                edgecolor="black",
                linewidth=0.7,
                color=COLORS[index % len(COLORS)],
                zorder=5,
            )
    axis.set_xlabel("False alarms per non-seizure hour (held-out patient)")
    axis.set_ylabel("Seizure event sensitivity")
    axis.set_ylim(-0.04, 1.04)
    axis.set_title("Patient-level continuous detection trade-off")
    axis.grid(True)
    axis.legend(frameon=False, loc="best", fontsize=8)
    axis.text(
        0.99,
        0.02,
        "Circles = held-out patients; diamonds = model means",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="#4f5d6b",
    )
    figure_path = output_dir / "patient-operating-points.png"
    figure.savefig(figure_path, bbox_inches="tight")
    plt.close(figure)
    paths.append(figure_path)

    summary = run.get("aggregate_patient_bootstrap", {})
    if not summary:
        summary = {}
        for model in models:
            summary[model] = {}
            for metric in ("event_sensitivity", "false_alarms_per_hour"):
                values = [
                    row.get(metric, row.get("metrics", {}).get(metric))
                    for row in folds
                    if row["model"] == model
                ]
                mean, ci_low, ci_high = _fold_interval(values)
                summary[model][metric] = {
                    "mean": mean,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                }
    if summary:
        figure, axes = plt.subplots(1, 2, figsize=(10.2, 4.8), constrained_layout=True)
        metrics = (
            ("event_sensitivity", "Event sensitivity", (0.0, 1.0)),
            ("false_alarms_per_hour", "False alarms per hour", None),
        )
        positions = np.arange(len(models))
        for axis, (metric, title, limits) in zip(axes, metrics):
            means, lower, upper = [], [], []
            for model in models:
                item = summary.get(model, {}).get(metric, {})
                mean = item.get("mean")
                low, high = item.get("ci_low"), item.get("ci_high")
                if mean is None or low is None or high is None:
                    mean = low = high = np.nan
                means.append(mean)
                lower.append(max(0.0, mean - low) if np.isfinite(mean) else np.nan)
                upper.append(max(0.0, high - mean) if np.isfinite(mean) else np.nan)
            means_array = np.asarray(means, dtype=float)
            errors = np.vstack([lower, upper])
            axis.errorbar(
                means_array,
                positions,
                xerr=errors,
                fmt="o",
                capsize=3,
                color="#2369bd",
                ecolor="#718096",
                markersize=6,
            )
            axis.set_yticks(positions, [_label(model) for model in models])
            axis.invert_yaxis()
            axis.set_title(title)
            axis.grid(axis="x")
            if limits:
                axis.set_xlim(*limits)
            axis.set_xlabel("Mean and 95% patient-bootstrap interval")
        figure.suptitle("Held-out patient summary (patients are the resampling unit)")
        figure_path = output_dir / "patient-bootstrap-summary.png"
        figure.savefig(figure_path, bbox_inches="tight")
        plt.close(figure)
        paths.append(figure_path)

    prediction_path = run_path.parent / "window_predictions.csv"
    if prediction_path.is_file():
        grouped: dict[str, dict[str, list[float]]] = {}
        with prediction_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                item = grouped.setdefault(row["model"], {"labels": [], "scores": []})
                item["labels"].append(int(row["label"]))
                item["scores"].append(float(row["probability"]))
        figure, axis = plt.subplots(figsize=(7.2, 5.3), constrained_layout=True)
        for index, model in enumerate(models):
            if model not in grouped:
                continue
            labels = np.asarray(grouped[model]["labels"])
            scores = np.asarray(grouped[model]["scores"])
            if np.unique(labels).size < 2:
                continue
            precision, recall, _ = precision_recall_curve(labels, scores)
            axis.plot(
                recall,
                precision,
                linewidth=2,
                color=COLORS[index % len(COLORS)],
                label=f"{_label(model)} (AP={average_precision_score(labels, scores):.3f})",
            )
        axis.set_xlabel("Window recall")
        axis.set_ylabel("Window precision")
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1.02)
        axis.set_title("Outer-test window precision–recall")
        axis.grid(True)
        axis.legend(frameon=False, fontsize=8, loc="best")
        axis.text(
            0.99,
            0.02,
            "Secondary window-level view; primary metrics are event sensitivity and alarms/hour",
            transform=axis.transAxes,
            ha="right",
            va="bottom",
            fontsize=7,
            color="#4f5d6b",
        )
        figure_path = output_dir / "window-precision-recall.png"
        figure.savefig(figure_path, bbox_inches="tight")
        plt.close(figure)
        paths.append(figure_path)
    return paths


def plot_bonn_run(run_path: Path, output_dir: Path) -> list[Path]:
    """Plot nested grouped-fold discrimination and pooled confusion matrices."""
    run = json.loads(run_path.read_text(encoding="utf-8"))
    folds = run.get("folds", [])
    if not folds or "metrics" not in folds[0]:
        raise ValueError(f"{run_path} does not contain a Bonn segment benchmark.")
    _style()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    models = sorted({fold["model"] for fold in folds})
    metric_specs = (
        ("window_auroc", "AUROC"),
        ("window_average_precision", "Average precision"),
        ("window_f1", "Window F1"),
    )
    figure, axes = plt.subplots(1, 3, figsize=(11, 4.8), constrained_layout=True)
    for axis, (metric, title) in zip(axes, metric_specs):
        for index, model in enumerate(models):
            values = [
                fold["metrics"].get(metric)
                for fold in folds
                if fold["model"] == model and fold["metrics"].get(metric) is not None
            ]
            values = np.asarray(values, dtype=float)
            if not values.size:
                continue
            mean = values.mean()
            sem = values.std(ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0.0
            axis.errorbar(
                index,
                mean,
                yerr=sem,
                fmt="o",
                capsize=3,
                color=COLORS[index % len(COLORS)],
            )
            axis.scatter(
                np.full(len(values), index) + np.linspace(-0.08, 0.08, len(values)),
                values,
                s=18,
                alpha=0.55,
                color=COLORS[index % len(COLORS)],
            )
        axis.set_xticks(range(len(models)), [_label(model) for model in models], rotation=35, ha="right")
        axis.set_ylim(0, 1.04)
        axis.set_title(title)
        axis.grid(axis="y")
    grouping_note = run.get(
        "grouping_note",
        "Grouping validity depends on the manifest-provided group_id.",
    )
    figure.suptitle(
        "Bonn nested grouped-segment test folds\n"
        f"{grouping_note}\n"
        "(points; mean ± fold SEM)"
    )
    figure_path = output_dir / "bonn-grouped-fold-metrics.png"
    figure.savefig(figure_path, bbox_inches="tight")
    plt.close(figure)
    paths.append(figure_path)

    figure, axes = plt.subplots(1, len(models), figsize=(3.4 * len(models), 3.5), constrained_layout=True)
    if len(models) == 1:
        axes = [axes]
    for axis, model in zip(axes, models):
        matrix = sum(
            (np.asarray(fold["confusion_matrix"], dtype=float) for fold in folds if fold["model"] == model),
            start=np.zeros((2, 2), dtype=float),
        )
        axis.imshow(matrix, cmap="Blues")
        for row in range(2):
            for column in range(2):
                axis.text(
                    column,
                    row,
                    f"{int(matrix[row, column])}",
                    ha="center",
                    va="center",
                    color="white" if matrix[row, column] > matrix.max() * 0.55 else "#17202a",
                    fontsize=11,
                )
        axis.set_xticks([0, 1], ["Non-ictal", "Ictal"], rotation=25, ha="right")
        axis.set_yticks([0, 1], ["Non-ictal", "Ictal"])
        axis.set_xlabel("Predicted")
        axis.set_ylabel("Actual")
        axis.set_title(_label(model))
    figure.suptitle(
        "Pooled outer-fold confusion matrices\n"
        f"{grouping_note}"
    )
    figure_path = output_dir / "bonn-confusion-matrices.png"
    figure.savefig(figure_path, bbox_inches="tight")
    plt.close(figure)
    paths.append(figure_path)
    return paths


def plot_comparison_run(comparison_path: Path, output_dir: Path) -> list[Path]:
    """Plot paired patient-bootstrap model differences from compare-chbmit output."""
    result = json.loads(comparison_path.read_text(encoding="utf-8"))
    comparisons = result.get("paired_comparisons", {})
    if not comparisons:
        raise ValueError(f"{comparison_path} contains no paired model comparisons.")
    _style()
    output_dir.mkdir(parents=True, exist_ok=True)
    models = sorted(comparisons)
    metrics = (
        ("event_sensitivity", "Event sensitivity"),
        ("false_alarms_per_hour", "False alarms/hour"),
        ("time_precision", "Time precision"),
    )
    figure, axes = plt.subplots(1, 3, figsize=(11, max(4.5, len(models) * 0.55)), constrained_layout=True)
    positions = np.arange(len(models))
    for axis, (metric, title) in zip(axes, metrics):
        means, lows, highs = [], [], []
        for model in models:
            item = comparisons[model][metric]
            mean = item["classical_minus_eegpt_mean"]
            low, high = item["ci_low"], item["ci_high"]
            if mean is None or low is None or high is None:
                mean = low = high = np.nan
            means.append(mean)
            lows.append(mean - low if np.isfinite(mean) else np.nan)
            highs.append(high - mean if np.isfinite(mean) else np.nan)
        means = np.asarray(means, dtype=float)
        axis.errorbar(
            means,
            positions,
            xerr=np.vstack([lows, highs]),
            fmt="o",
            capsize=3,
            color="#2369bd",
        )
        axis.axvline(0, color="#4f5d6b", linestyle="--", linewidth=1)
        axis.set_yticks(positions, [_label(model) for model in models])
        axis.invert_yaxis()
        axis.set_title(title)
        axis.set_xlabel("Classical − EEGPT (95% paired patient-bootstrap CI)")
        axis.grid(axis="x")
    figure.suptitle("Paired held-out patient comparisons")
    figure_path = output_dir / "paired-classical-vs-eegpt.png"
    figure.savefig(figure_path, bbox_inches="tight")
    plt.close(figure)
    return [figure_path]


def plot_alignment_comparison(comparison_path: Path, output_dir: Path) -> list[Path]:
    """Plot per-patient adaptation effects and their unlabeled shift diagnostics."""
    result = json.loads(comparison_path.read_text(encoding="utf-8"))
    table_path = Path(result["patient_differences_csv"])
    with table_path.open(newline="", encoding="utf-8") as input_file:
        rows = list(csv.DictReader(input_file))
    if not rows:
        raise ValueError(f"{table_path} contains no paired patient differences.")
    _style()
    output_dir.mkdir(parents=True, exist_ok=True)
    models = list(result["models"])
    subjects = list(result["matched_subjects"])
    metrics = (
        ("event_sensitivity", "Event sensitivity change", True),
        ("false_alarms_per_hour", "False alarms/hour change", False),
        ("time_precision", "Time precision change", True),
    )
    paths: list[Path] = []

    figure, axes = plt.subplots(
        len(models),
        len(metrics),
        figsize=(13.5, max(3.2, 0.48 * len(subjects) + 1.4) * len(models)),
        squeeze=False,
        constrained_layout=True,
    )
    positions = np.arange(len(subjects))
    for model_index, model in enumerate(models):
        model_rows = {
            row["subject_id"]: row for row in rows if row["model"] == model
        }
        for metric_index, (metric, title, higher_is_better) in enumerate(metrics):
            axis = axes[model_index, metric_index]
            values = np.asarray(
                [
                    float(model_rows[subject][f"delta_euclidean_minus_none_{metric}"])
                    if model_rows[subject][f"delta_euclidean_minus_none_{metric}"]
                    else np.nan
                    for subject in subjects
                ],
                dtype=float,
            )
            favorable = values >= 0 if higher_is_better else values <= 0
            colors = np.where(favorable, "#318a59", "#b04759")
            valid = np.isfinite(values)
            axis.scatter(values[valid], positions[valid], c=colors[valid], s=32, zorder=3)
            axis.axvline(0, color="#4f5d6b", linestyle="--", linewidth=1)
            axis.set_title(f"{_label(model)}\n{title}")
            axis.set_yticks(positions)
            if metric_index == 0:
                axis.set_yticklabels(subjects)
            else:
                axis.tick_params(axis="y", labelleft=False)
            axis.invert_yaxis()
            axis.grid(axis="x")
            axis.set_xlabel("Euclidean alignment − no alignment")
    figure.suptitle(
        "Patient-paired alignment effects (green = favorable direction; red = unfavorable)"
    )
    figure_path = output_dir / "paired-alignment-patient-differences.png"
    figure.savefig(figure_path, bbox_inches="tight")
    plt.close(figure)
    paths.append(figure_path)

    if result.get("unlabeled_shift_diagnostics_available"):
        figure, axes = plt.subplots(1, len(metrics), figsize=(13.5, 4.4), constrained_layout=True)
        for axis, (metric, title, _higher_is_better) in zip(axes, metrics):
            for model_index, model in enumerate(models):
                model_rows = [row for row in rows if row["model"] == model]
                for row in model_rows:
                    no_shift = row.get("no_alignment_shift_distance")
                    ea_shift = row.get("euclidean_shift_distance")
                    delta = row.get(f"delta_euclidean_minus_none_{metric}")
                    if no_shift in ("", None) or ea_shift in ("", None) or delta in ("", None):
                        continue
                    x_no = float(no_shift)
                    x_ea = float(ea_shift)
                    y_delta = float(delta)
                    axis.plot(
                        [x_no, x_ea],
                        [y_delta, y_delta],
                        color="#aab2b9",
                        linewidth=0.6,
                        alpha=0.45,
                        zorder=1,
                    )
                    axis.scatter(
                        x_no,
                        y_delta,
                        marker="o",
                        s=23,
                        color=COLORS[model_index % len(COLORS)],
                        alpha=0.75,
                        label=_label(model),
                        zorder=2,
                    )
                    axis.scatter(
                        x_ea,
                        y_delta,
                        marker="^",
                        s=24,
                        color=COLORS[model_index % len(COLORS)],
                        alpha=0.75,
                        zorder=2,
                    )
            axis.axhline(0, color="#4f5d6b", linestyle="--", linewidth=1)
            axis.set_title(title)
            axis.set_xlabel("Unlabeled feature-mean shift (source-standardized)")
            axis.set_ylabel("Euclidean alignment − no alignment")
            axis.grid()
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            unique = dict(zip(labels, handles))
            figure.legend(unique.values(), unique.keys(), loc="upper center", ncol=min(5, len(unique)))
        figure.suptitle("Alignment effect versus unlabeled calibration shift")
        shift_figure_path = output_dir / "alignment-effect-vs-unlabeled-shift.png"
        figure.savefig(shift_figure_path, bbox_inches="tight")
        plt.close(figure)
        paths.append(shift_figure_path)
    return paths


def plot_saved_run(run_path: Path, output_dir: Path) -> list[Path]:
    """Infer supported result type from saved JSON and generate its figures."""
    run = json.loads(run_path.read_text(encoding="utf-8"))
    if run.get("task", "").startswith("Bonn"):
        return plot_bonn_run(run_path, output_dir)
    if run.get("task", "").startswith("Paired CHB-MIT Euclidean-alignment"):
        return plot_alignment_comparison(run_path, output_dir)
    if run.get("paired_comparisons"):
        return plot_comparison_run(run_path, output_dir)
    if "folds" in run:
        return plot_chbmit_run(run_path, output_dir)
    raise ValueError(f"Unsupported results file: {run_path}")
