from dataclasses import replace

import numpy as np

from seizure_detection.benchmark import _grouped_cv, _validation_split, subset_table
from seizure_detection.features import FeatureTable, RecordMetadata


def _table() -> FeatureTable:
    records = {}
    record_ids = []
    subject_ids = []
    labels = []
    for index in range(6):
        record_id = f"p{index}/record"
        subject = f"p{index}"
        seizure = index < 3
        records[record_id] = RecordMetadata(
            subject, 8.0, ((2.0, 6.0),) if seizure else ()
        )
        record_ids.extend([record_id, record_id])
        subject_ids.extend([subject, subject])
        labels.extend([0, int(seizure)])
    count = len(labels)
    return FeatureTable(
        X=np.arange(count, dtype=np.float32).reshape(-1, 1),
        y=np.asarray(labels, dtype=np.int8),
        record_ids=np.asarray(record_ids, dtype=object),
        subject_ids=np.asarray(subject_ids, dtype=object),
        starts=np.asarray([0, 4] * 6, dtype=float),
        ends=np.asarray([4, 8] * 6, dtype=float),
        records=records,
        feature_names=("x",),
    )


def test_validation_split_keeps_recordings_disjoint_and_both_record_types() -> None:
    table = _table()
    train, validation = _validation_split(table, seed=12)
    train_records = set(table.record_ids[train])
    validation_records = set(table.record_ids[validation])
    assert train_records.isdisjoint(validation_records)
    assert any(table.records[key].seizure_intervals for key in train_records)
    assert any(not table.records[key].seizure_intervals for key in train_records)
    assert any(table.records[key].seizure_intervals for key in validation_records)
    assert any(not table.records[key].seizure_intervals for key in validation_records)


def test_subset_table_drops_unrepresented_record_metadata() -> None:
    table = _table()
    subset = subset_table(table, np.asarray([0, 1, 2, 3]))
    assert set(subset.records) == {"p0/record", "p1/record"}
    assert len(subset.y) == 4


def test_small_seizure_record_cohort_uses_record_disjoint_validation() -> None:
    table = _table()
    records = {
        key: replace(record, seizure_intervals=((2.0, 6.0),))
        for key, record in table.records.items()
    }
    table = replace(table, records=records)

    train, validation = _validation_split(table, seed=12)
    train_records = set(table.record_ids[train])
    validation_records = set(table.record_ids[validation])
    assert train_records.isdisjoint(validation_records)
    assert np.unique(table.y[train]).size == 2
    assert np.unique(table.y[validation]).size == 2


def test_grouped_cv_supports_three_independent_recordings() -> None:
    record_ids = np.repeat(["p1/r1", "p2/r1", "p3/r1"], 4)
    labels = np.tile([0, 0, 1, 1], 3)
    splits = _grouped_cv(labels, record_ids, seed=8)

    assert len(splits) == 3
    for train, validation in splits:
        assert set(record_ids[train]).isdisjoint(record_ids[validation])
        assert np.unique(labels[train]).size == 2
        assert np.unique(labels[validation]).size == 2
