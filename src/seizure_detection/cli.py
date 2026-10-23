"""Command-line interface for data inspection and experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from seizure_detection.alarms import AlarmConfig
from seizure_detection.bonn import run_bonn_benchmark
from seizure_detection.benchmark import run_loso_benchmark
from seizure_detection.compare import compare_alignment_runs, compare_chbmit_runs
from seizure_detection.config import SignalConfig
from seizure_detection.data import discover_chbmit, load_chbmit_record
from seizure_detection.plotting import plot_saved_run


def _signal_config(args: argparse.Namespace) -> SignalConfig:
    return SignalConfig(
        alignment=args.alignment,
        alignment_reference_seconds=args.alignment_reference_seconds,
    )


def _alarm_config(args: argparse.Namespace) -> AlarmConfig:
    return AlarmConfig(
        target_false_alarms_per_hour=args.target_false_alarms_per_hour,
        minimum_alarm_seconds=args.minimum_alarm_seconds,
        merge_gap_seconds=args.merge_gap_seconds,
        threshold_candidates=args.threshold_candidates,
    )


def _inspect(data_root: Path) -> None:
    records = discover_chbmit(data_root)
    subjects: dict[str, dict[str, int]] = {}
    for path, subject, annotations in records:
        subject_entry = subjects.setdefault(subject, {"records": 0, "seizure_records": 0, "seizures": 0})
        subject_entry["records"] += 1
        seizure_count = len(annotations.get(path.name, ()))
        subject_entry["seizure_records"] += int(seizure_count > 0)
        subject_entry["seizures"] += seizure_count
    result = {
        "data_root": str(data_root.expanduser().resolve()),
        "n_patients": len(subjects),
        "n_records": len(records),
        "patients": subjects,
        "note": "Counts are discovered from EDF files and original summary annotations; no signal data loaded.",
    }
    print(json.dumps(result, indent=2))


def _add_experiment_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--subjects",
        nargs="+",
        help="Optional explicit CHB-MIT subject IDs; otherwise every discovered patient is used.",
    )
    parser.add_argument("--alignment", choices=("none", "euclidean"), default="none")
    parser.add_argument("--alignment-reference-seconds", type=float, default=60.0)
    parser.add_argument("--target-false-alarms-per-hour", type=float, default=1.0)
    parser.add_argument("--minimum-alarm-seconds", type=float, default=8.0)
    parser.add_argument("--merge-gap-seconds", type=float, default=8.0)
    parser.add_argument("--threshold-candidates", type=int, default=201)
    parser.add_argument("--max-negative-ratio", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=2026)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="seizure-detection",
        description="Leakage-resistant seizure detection research benchmark.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    inspect_parser = subparsers.add_parser("inspect-chbmit", help="Count patients and annotated records.")
    inspect_parser.add_argument("--data-root", type=Path, required=True)

    benchmark_parser = subparsers.add_parser(
        "benchmark-chbmit", help="Run the grouped classical LOSO comparison."
    )
    _add_experiment_arguments(benchmark_parser)
    benchmark_parser.add_argument(
        "--models",
        nargs="+",
        choices=("logistic", "rbf_svm", "random_forest", "extra_trees", "hist_gradient_boosting"),
        default=("logistic", "rbf_svm", "random_forest", "extra_trees", "hist_gradient_boosting"),
    )
    benchmark_parser.add_argument("--bootstrap-samples", type=int, default=2000)

    bonn_parser = subparsers.add_parser(
        "benchmark-bonn", help="Run nested manifest-grouped Bonn segment evaluation."
    )
    bonn_parser.add_argument("--manifest", type=Path, required=True)
    bonn_parser.add_argument("--output-dir", type=Path, required=True)
    bonn_parser.add_argument("--sampling-rate", type=float, default=173.61)
    bonn_parser.add_argument("--seed", type=int, default=2026)
    bonn_parser.add_argument("--max-negative-ratio", type=float, default=5.0)
    bonn_parser.add_argument(
        "--data-provenance",
        default="external_dataset",
        help="Free-text dataset name/source stored in the run metadata.",
    )
    bonn_parser.add_argument(
        "--grouping-note",
        default=(
            "Grouped by manifest group_id; source-level validity depends on accurate "
            "source recording identifiers."
        ),
        help="Describe what each manifest group represents; stored in run metadata and figures.",
    )
    bonn_parser.add_argument(
        "--random-window-ablation",
        action="store_true",
        help="Also run the intentionally leaky random-segment split for comparison only.",
    )
    bonn_parser.add_argument(
        "--models",
        nargs="+",
        choices=("logistic", "rbf_svm", "random_forest", "extra_trees", "hist_gradient_boosting"),
        default=("logistic", "rbf_svm", "random_forest", "extra_trees", "hist_gradient_boosting"),
    )

    eegpt_parser = subparsers.add_parser(
        "finetune-eegpt", help="Fine-tune pretrained EEGPT on grouped CHB-MIT folds."
    )
    _add_experiment_arguments(eegpt_parser)
    eegpt_parser.add_argument("--model-id", default="braindecode/eegpt-pretrained")
    eegpt_parser.add_argument("--input-scale", type=float, default=1.0)
    eegpt_parser.add_argument("--epochs", type=int, default=10)
    eegpt_parser.add_argument("--batch-size", type=int, default=32)
    eegpt_parser.add_argument("--learning-rate", type=float, default=1e-5)

    compare_parser = subparsers.add_parser(
        "compare-chbmit", help="Paired patient-bootstrap comparison of saved LOSO runs."
    )
    compare_parser.add_argument("--classical-run", type=Path, required=True)
    compare_parser.add_argument("--foundation-run", type=Path, required=True)
    compare_parser.add_argument("--output", type=Path, required=True)
    compare_parser.add_argument("--bootstrap-samples", type=int, default=2000)
    compare_parser.add_argument("--seed", type=int, default=2026)

    compare_alignment_parser = subparsers.add_parser(
        "compare-alignment",
        help="Paired patient-level comparison of matched no-alignment and Euclidean runs.",
    )
    compare_alignment_parser.add_argument("--no-alignment-run", type=Path, required=True)
    compare_alignment_parser.add_argument("--euclidean-run", type=Path, required=True)
    compare_alignment_parser.add_argument("--output", type=Path, required=True)
    compare_alignment_parser.add_argument("--bootstrap-samples", type=int, default=2000)
    compare_alignment_parser.add_argument("--seed", type=int, default=2026)

    plot_parser = subparsers.add_parser(
        "plot-results", help="Regenerate figures from a saved benchmark or comparison JSON."
    )
    plot_parser.add_argument("--run", type=Path, required=True)
    plot_parser.add_argument("--output-dir", type=Path)

    args = parser.parse_args()
    if args.command == "inspect-chbmit":
        _inspect(args.data_root)
        return
    if args.command == "benchmark-bonn":
        run_bonn_benchmark(
            args.manifest,
            args.output_dir,
            sampling_rate=args.sampling_rate,
            model_names=tuple(args.models),
            seed=args.seed,
            random_window_ablation=args.random_window_ablation,
            max_negative_ratio=args.max_negative_ratio,
            data_provenance=args.data_provenance,
            grouping_note=args.grouping_note,
        )
        return
    if args.command == "compare-chbmit":
        compare_chbmit_runs(
            args.classical_run,
            args.foundation_run,
            args.output,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
        )
        return
    if args.command == "compare-alignment":
        compare_alignment_runs(
            args.no_alignment_run,
            args.euclidean_run,
            args.output,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
        )
        return
    if args.command == "plot-results":
        destination = args.output_dir or args.run.parent / "figures"
        for figure_path in plot_saved_run(args.run, destination):
            print(figure_path)
        return
    signal_config = _signal_config(args)
    alarm_config = _alarm_config(args)
    if args.command == "benchmark-chbmit":
        run_loso_benchmark(
            args.data_root,
            args.output_dir,
            signal_config,
            alarm_config,
            tuple(args.models),
            args.seed,
            args.bootstrap_samples,
            tuple(args.subjects) if args.subjects else None,
            args.max_negative_ratio,
        )
        return
    if args.command == "finetune-eegpt":
        if signal_config.alignment != "none":
            raise ValueError(
                "The EEGPT waveform path currently evaluates the unaligned signal protocol only."
            )
        from seizure_detection.foundation import require_eegpt_dependencies, run_eegpt_loso

        require_eegpt_dependencies()
        discovered = discover_chbmit(args.data_root)
        if args.subjects:
            requested = set(args.subjects)
            available = {subject for _, subject, _ in discovered}
            missing = sorted(requested - available)
            if missing:
                raise ValueError(f"Requested patients not found in data root: {', '.join(missing)}.")
            discovered = [row for row in discovered if row[1] in requested]
        record_stream = (
            load_chbmit_record(path, subject, annotations, signal_config)
            for path, subject, annotations in discovered
        )
        from seizure_detection.features import build_window_index

        table = build_window_index(record_stream, signal_config)
        run_eegpt_loso(
            discovered,
            table,
            args.output_dir,
            signal_config,
            alarm_config,
            model_id=args.model_id,
            input_scale=args.input_scale,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_negative_ratio=args.max_negative_ratio,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
