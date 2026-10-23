import csv
import json
from pathlib import Path

from seizure_detection.plotting import plot_chbmit_run, plot_saved_run


def test_chbmit_plotter_generates_patient_and_window_figures(tmp_path: Path) -> None:
    run_path = tmp_path / "run.json"
    run_path.write_text(
        json.dumps(
            {
                "folds": [
                    {
                        "model": "logistic",
                        "subject_id": "p1",
                        "event_sensitivity": 0.5,
                        "false_alarms_per_hour": 1.0,
                        "time_precision": 0.3,
                    },
                    {
                        "model": "logistic",
                        "subject_id": "p2",
                        "event_sensitivity": 1.0,
                        "false_alarms_per_hour": 0.5,
                        "time_precision": 0.6,
                    },
                ],
                "aggregate_patient_bootstrap": {
                    "logistic": {
                        "event_sensitivity": {"mean": 0.75, "ci_low": 0.5, "ci_high": 1.0},
                        "false_alarms_per_hour": {"mean": 0.75, "ci_low": 0.5, "ci_high": 1.0},
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    with (tmp_path / "window_predictions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["model", "label", "probability"],
        )
        writer.writeheader()
        for label, probability in ((0, 0.1), (0, 0.2), (1, 0.8), (1, 0.9)):
            writer.writerow(
                {"model": "logistic", "label": label, "probability": probability}
            )

    figures = plot_chbmit_run(run_path, tmp_path / "figures")
    assert {path.name for path in figures} == {
        "patient-operating-points.png",
        "patient-bootstrap-summary.png",
        "window-precision-recall.png",
    }
    assert all(path.stat().st_size > 1000 for path in figures)


def test_plot_saved_run_dispatches_to_bonn_plotter(tmp_path: Path) -> None:
    run_path = tmp_path / "bonn-run.json"
    run_path.write_text(
        json.dumps(
            {
                "task": "Bonn ictal versus non-ictal segment classification",
                "folds": [
                    {
                        "model": "logistic",
                        "metrics": {
                            "window_auroc": 0.8,
                            "window_average_precision": 0.7,
                            "window_f1": 0.6,
                        },
                        "confusion_matrix": [[8, 2], [3, 7]],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    figures = plot_saved_run(run_path, tmp_path / "figures")
    assert len(figures) == 2
    assert all(path.is_file() for path in figures)
