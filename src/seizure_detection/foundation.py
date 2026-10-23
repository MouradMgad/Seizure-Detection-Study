"""Optional EEGPT pretrained-transformer fine-tuning experiment."""

from __future__ import annotations

from copy import deepcopy
from fractions import Fraction
import math
from pathlib import Path
from typing import Any

import numpy as np
import scipy
from scipy.signal import resample_poly

from seizure_detection.alarms import choose_threshold, score_table, window_metrics
from seizure_detection.config import AlarmConfig, COMMON_BIPOLAR_CHANNELS, SignalConfig
from seizure_detection.data import EEGRecord, load_chbmit_record
from seizure_detection.features import FeatureTable
from seizure_detection.serialization import write_json
from seizure_detection.benchmark import (
    _aggregate_patient_bootstrap,
    _validation_split,
    subset_table,
)


def _require_deep_dependencies() -> tuple[Any, Any]:
    try:
        import torch
        import mne
        from braindecode.models import EEGPT
    except ImportError as exc:
        raise ImportError(
            "EEGPT fine-tuning requires the optional extra: "
            "`python -m pip install -e '.[foundation]'`."
        ) from exc
    return torch, (mne, EEGPT)


def require_eegpt_dependencies() -> None:
    """Fail before loading EEG data when optional foundation dependencies are absent."""
    _require_deep_dependencies()


def _bipolar_channel_info(mne: Any, target_sfreq: float) -> list[dict[str, Any]]:
    """Build approximate bipolar midpoint locations for pretrained montage mapping."""
    montage = mne.channels.make_standard_montage("colin27_1020")
    positions = {
        name.upper(): position
        for name, position in montage.get_positions()["ch_pos"].items()
    }
    info = mne.create_info(
        list(COMMON_BIPOLAR_CHANNELS),
        sfreq=target_sfreq,
        ch_types=["eeg"] * len(COMMON_BIPOLAR_CHANNELS),
    )
    for channel, channel_info in zip(COMMON_BIPOLAR_CHANNELS, info["chs"]):
        left, right = channel.split("-")
        if left not in positions or right not in positions:
            raise ValueError(f"EEGPT montage cannot locate electrodes for bipolar channel {channel}.")
        midpoint = (positions[left] + positions[right]) / 2
        norm = np.linalg.norm(midpoint)
        if norm <= np.finfo(float).eps:
            raise ValueError(f"Invalid midpoint location for bipolar channel {channel}.")
        channel_info["loc"][:3] = midpoint / norm * 0.09
    return info["chs"]


def _make_pretrained_eegpt(
    chs_info: list[dict[str, Any]],
    n_times: int,
    model_id: str,
) -> Any:
    torch, (mne, EEGPT) = _require_deep_dependencies()
    del torch, mne
    model = EEGPT.from_pretrained(
        model_id,
        chs_info=chs_info,
        channel_strategy="spline",
    )
    if model.n_chans != len(COMMON_BIPOLAR_CHANNELS) or model.n_times != n_times:
        raise RuntimeError(
            "Pretrained EEGPT configuration must resolve to 18 channels and the requested "
            f"{n_times} samples; received {model.n_chans} channels and {model.n_times} samples."
        )
    if not hasattr(model, "reset_head"):
        raise RuntimeError("Installed Braindecode EEGPT has no supported reset_head method.")
    model.reset_head(2)
    return model


def _resampled_record(record: EEGRecord, target_sfreq: float) -> np.ndarray:
    ratio = Fraction(target_sfreq / record.sfreq).limit_denominator(1000)
    return resample_poly(record.data_uv, ratio.numerator, ratio.denominator, axis=1).astype(
        np.float32, copy=False
    )


def _materialize_windows(
    table: FeatureTable,
    record_lookup: dict[str, tuple[Path, str, dict[str, tuple[tuple[float, float], ...]]]],
    selected_indices: np.ndarray,
    signal_config: SignalConfig,
    target_sfreq: float,
    window_seconds: float,
    input_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Materialize only selected raw windows, allowing training-only subsampling."""
    output = np.empty(
        (len(selected_indices), len(COMMON_BIPOLAR_CHANNELS), int(round(window_seconds * target_sfreq))),
        dtype=np.float32,
    )
    labels = table.y[selected_indices].astype(np.int64)
    grouped_output_positions: dict[str, list[int]] = {}
    for output_position, table_index in enumerate(selected_indices):
        record_id = str(table.record_ids[table_index])
        grouped_output_positions.setdefault(record_id, []).append(output_position)
    for record_id, output_positions in grouped_output_positions.items():
        path, subject_id, annotations = record_lookup[record_id]
        record = load_chbmit_record(path, subject_id, annotations, signal_config)
        data = _resampled_record(record, target_sfreq)
        for output_position in output_positions:
            table_index = selected_indices[output_position]
            start_sample = int(round(float(table.starts[table_index]) * target_sfreq))
            stop_sample = start_sample + output.shape[2]
            segment = data[:, start_sample:stop_sample]
            if segment.shape[1] != output.shape[2]:
                raise ValueError(
                    f"Window at {record_id}:{table.starts[table_index]}s is incomplete."
                )
            output[output_position] = segment * input_scale
    return output, labels


def _training_indices(
    table: FeatureTable,
    indices: np.ndarray,
    seed: int,
    max_negative_ratio: float,
) -> np.ndarray:
    labels = table.y[indices]
    positive = indices[labels == 1]
    negative = indices[labels == 0]
    if not positive.size or not negative.size:
        raise ValueError("Fine-tuning requires both seizure and non-seizure training windows.")
    rng = np.random.default_rng(seed)
    maximum_negatives = int(np.ceil(max_negative_ratio * len(positive)))
    if len(negative) > maximum_negatives:
        negative = np.sort(rng.choice(negative, size=maximum_negatives, replace=False))
    selected = np.concatenate([positive, negative])
    return selected[rng.permutation(len(selected))]


def _predict_torch(
    torch: Any,
    model: Any,
    values: np.ndarray,
    device: Any,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    probabilities: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(values), batch_size):
            batch = torch.from_numpy(values[start : start + batch_size]).to(device)
            logits = model(batch)
            if isinstance(logits, dict):
                logits = logits.get("logits")
            if logits is None or logits.ndim != 2 or logits.shape[1] != 2:
                raise RuntimeError(
                    "EEGPT must return two-class logits; check the installed Braindecode API."
                )
            probabilities.extend(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
    return np.asarray(probabilities, dtype=np.float64)


def run_eegpt_loso(
    record_specs: list[
        tuple[Path, str, dict[str, tuple[tuple[float, float], ...]]]
    ],
    table: FeatureTable,
    output_dir: Path,
    signal_config: SignalConfig,
    alarm_config: AlarmConfig,
    model_id: str = "braindecode/eegpt-pretrained",
    target_sfreq: float = 250.0,
    input_scale: float = 1.0,
    epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 1e-5,
    max_negative_ratio: float = 5.0,
    seed: int = 2026,
) -> dict[str, Any]:
    """Fine-tune pretrained EEGPT separately per patient-level LOSO fold."""
    if signal_config.alignment != "none":
        raise ValueError(
            "The EEGPT waveform path currently evaluates the unaligned signal protocol only."
        )
    if (
        epochs < 1
        or batch_size < 1
        or not math.isfinite(learning_rate)
        or learning_rate <= 0
        or not math.isfinite(max_negative_ratio)
        or max_negative_ratio < 1
        or not math.isfinite(target_sfreq)
        or target_sfreq <= 0
        or not math.isfinite(input_scale)
        or input_scale <= 0
    ):
        raise ValueError("epochs, batch size, learning rate, and negative ratio must be positive.")
    torch, (mne, _) = _require_deep_dependencies()
    output_dir.mkdir(parents=True, exist_ok=True)
    record_lookup = {
        f"{subject_id}/{path.stem}": (path, subject_id, annotations)
        for path, subject_id, annotations in record_specs
    }
    subjects = np.unique(table.subject_ids)
    if len(subjects) < 3:
        raise ValueError("EEGPT LOSO requires at least three patients.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_window_samples = int(round(signal_config.window_seconds * target_sfreq))
    chs_info = _bipolar_channel_info(mne, target_sfreq)
    fold_results: list[dict[str, Any]] = []

    for fold_number, subject in enumerate(subjects):
        pool_indices = np.flatnonzero(table.subject_ids != subject)
        test_indices = np.flatnonzero(table.subject_ids == subject)
        pool = subset_table(table, pool_indices)
        train_local, validation_local = _validation_split(pool, seed + fold_number)
        train_indices = pool_indices[train_local]
        validation_indices = pool_indices[validation_local]
        train_indices = _training_indices(
            table, train_indices, seed + fold_number, max_negative_ratio
        )
        validation_evaluation_indices = validation_indices[
            table.starts[validation_indices] >= signal_config.alignment_reference_seconds
        ]
        test_evaluation_indices = test_indices[
            table.starts[test_indices] >= signal_config.alignment_reference_seconds
        ]
        if not len(validation_evaluation_indices):
            raise ValueError(
                f"Validation recordings in fold {subject} have no windows after the "
                f"{signal_config.alignment_reference_seconds:g}-second calibration period."
            )
        if not len(test_evaluation_indices):
            raise ValueError(
                f"Held-out patient {subject} has no windows after the "
                f"{signal_config.alignment_reference_seconds:g}-second calibration period."
            )

        train_X, train_y = _materialize_windows(
            table,
            record_lookup,
            train_indices,
            signal_config,
            target_sfreq,
            signal_config.window_seconds,
            input_scale,
        )
        validation_X, validation_y = _materialize_windows(
            table,
            record_lookup,
            validation_evaluation_indices,
            signal_config,
            target_sfreq,
            signal_config.window_seconds,
            input_scale,
        )
        if np.unique(validation_y).size != 2:
            raise ValueError(
                f"Validation recordings in fold {subject} do not contain both window "
                "classes after warm-up exclusion; threshold/epoch selection requires "
                "post-warm-up ictal and non-ictal data."
            )
        validation_table = subset_table(table, validation_evaluation_indices)
        test_table = subset_table(table, test_evaluation_indices)
        test_X, _ = _materialize_windows(
            table,
            record_lookup,
            test_evaluation_indices,
            signal_config,
            target_sfreq,
            signal_config.window_seconds,
            input_scale,
        )
        if train_X.shape[2] != target_window_samples:
            raise RuntimeError("Unexpected resampled EEGPT input shape.")

        torch.manual_seed(seed + fold_number)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed + fold_number)
        model = _make_pretrained_eegpt(chs_info, target_window_samples, model_id).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
        class_counts = np.bincount(train_y, minlength=2)
        class_weights = torch.tensor(
            [1.0, class_counts[0] / max(class_counts[1], 1)],
            dtype=torch.float32,
            device=device,
        )
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
        generator = torch.Generator().manual_seed(seed + fold_number)
        dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(train_X), torch.from_numpy(train_y)
        )
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
            num_workers=0,
        )
        best_state = None
        best_validation_ap = -np.inf
        best_epoch = 0
        for epoch in range(epochs):
            model.train()
            for batch_X, batch_y in loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(batch_X)
                if isinstance(logits, dict):
                    logits = logits.get("logits")
                if logits is None or logits.shape != (len(batch_y), 2):
                    raise RuntimeError(
                        "EEGPT must return [batch, 2] logits; check the Braindecode version."
                    )
                loss = criterion(logits, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            validation_probabilities = _predict_torch(
                torch, model, validation_X, device, batch_size
            )
            from sklearn.metrics import average_precision_score

            validation_ap = (
                float(average_precision_score(validation_y, validation_probabilities))
                if np.any(validation_y)
                else float("nan")
            )
            if np.isfinite(validation_ap) and validation_ap > best_validation_ap:
                best_validation_ap = validation_ap
                best_epoch = epoch + 1
                best_state = deepcopy(model.state_dict())
        if best_state is None:
            raise ValueError(f"No valid validation average precision for held-out patient {subject}.")
        model.load_state_dict(best_state)
        validation_probabilities = _predict_torch(
            torch, model, validation_X, device, batch_size
        )
        threshold, validation_summary = choose_threshold(
            validation_table,
            validation_probabilities,
            alarm_config,
            evaluation_start_seconds=signal_config.alignment_reference_seconds,
        )
        test_probabilities = _predict_torch(torch, model, test_X, device, batch_size)
        records_metrics, alarms_by_record = score_table(
            test_table,
            test_probabilities,
            threshold,
            alarm_config,
            evaluation_start_seconds=signal_config.alignment_reference_seconds,
        )
        events = sum(item.total_events for item in records_metrics)
        detected = sum(item.detected_events for item in records_metrics)
        false_alarms = sum(item.false_alarms for item in records_metrics)
        nonseizure_seconds = sum(
            max(
                item.duration_seconds
                - sum(
                    end - max(start, signal_config.alignment_reference_seconds)
                    for start, end in test_table.records[item.record_id].seizure_intervals
                    if end > signal_config.alignment_reference_seconds
                ),
                0.0,
            )
            for item in records_metrics
        )
        alarm_seconds = sum(
            end - start for intervals in alarms_by_record.values() for start, end in intervals
        )
        ictal_alarm_seconds = sum(
            max(0.0, min(alarm_end, seizure_end) - max(alarm_start, seizure_start))
            for record_id, intervals in alarms_by_record.items()
            for alarm_start, alarm_end in intervals
            for seizure_start, seizure_end in test_table.records[record_id].seizure_intervals
        )
        fold = {
            "model": "eegpt_finetuned",
            "subject_id": str(subject),
            "n_test_records": len(test_table.records),
            "n_test_windows": int(len(test_table.y)),
            "n_calibration_windows_excluded": int(len(test_indices) - len(test_evaluation_indices)),
            "evaluation_start_seconds": signal_config.alignment_reference_seconds,
            "n_test_seizures": events,
            "event_sensitivity": detected / events if events else float("nan"),
            "false_alarms": false_alarms,
            "false_alarms_per_hour": (
                false_alarms / (nonseizure_seconds / 3600) if nonseizure_seconds else 0.0
            ),
            "time_precision": ictal_alarm_seconds / alarm_seconds if alarm_seconds else 0.0,
            "selected_threshold": float(threshold),
            "validation_threshold_summary": validation_summary,
            "validation_average_precision": float(best_validation_ap),
            "selected_epoch": best_epoch,
            "window_metrics": window_metrics(test_table.y, test_probabilities, threshold),
            "records": [item.__dict__ for item in records_metrics],
        }
        fold_results.append(fold)
        torch.save(best_state, output_dir / f"eegpt-{subject}.pt")
        write_json(output_dir / f"eegpt-{subject}-alarms.json", alarms_by_record)
        print(
            f"eegpt_finetuned | held out {subject}: "
            f"event sensitivity={fold['event_sensitivity']:.3f}, "
            f"false alarms/hour={fold['false_alarms_per_hour']:.3f}"
        )

    results = {
        "model": "eegpt_finetuned",
        "pretrained_model_id": model_id,
        "target_sampling_rate_hz": target_sfreq,
        "input_scale": input_scale,
        "channel_mapping": "18 bipolar derivations mapped to standard-electrode midpoint locations",
        "device": str(device),
        "braindecode_version": __import__("importlib.metadata", fromlist=["version"]).version(
            "braindecode"
        ),
        "torch_version": torch.__version__,
        "mne_version": mne.__version__,
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "signal_config": signal_config.__dict__,
        "alarm_config": alarm_config.__dict__,
        "target_data_policy": {
            "target_labels_used_for_adaptation": False,
            "unlabeled_target_signal_used_for_adaptation": False,
            "alignment_signal_interval_seconds": None,
            "calibration_interval_may_contain_seizures_or_artifacts": False,
        },
        "evaluation_protocol": {
            "warmup_seconds_excluded_per_record": signal_config.alignment_reference_seconds,
            "euclidean_alignment_reference_seconds": None,
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
        "subjects": [str(subject) for subject in subjects],
        "record_ids": sorted(record_lookup),
        "seed": seed,
        "aggregate_patient_bootstrap": _aggregate_patient_bootstrap(
            fold_results, 2000, seed + 1
        ),
        "folds": fold_results,
    }
    write_json(output_dir / "eegpt-run.json", results)
    from seizure_detection.plotting import plot_chbmit_run

    plot_chbmit_run(output_dir / "eegpt-run.json", output_dir / "figures")
    return results
