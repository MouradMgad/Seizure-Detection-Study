import numpy as np

from seizure_detection.models import make_model


def test_negative_subsampling_is_fit_only_and_caps_training_ratio() -> None:
    rng = np.random.default_rng(10)
    X = rng.normal(size=(100, 3))
    y = np.asarray([0] * 90 + [1] * 10, dtype=np.int8)
    pipeline, _ = make_model("logistic", seed=11, max_negative_ratio=5)
    pipeline.fit(X, y)
    sampled = pipeline.named_steps["undersample"].sample_indices_
    sampled_labels = y[sampled]
    assert int(np.sum(sampled_labels == 0)) == 50
    assert int(np.sum(sampled_labels == 1)) == 10
    assert pipeline.predict_proba(X).shape == (100, 2)


def test_negative_subsampling_does_not_reduce_already_balanced_training_data() -> None:
    rng = np.random.default_rng(14)
    X = rng.normal(size=(20, 3))
    y = np.asarray([0] * 10 + [1] * 10, dtype=np.int8)
    pipeline, _ = make_model("logistic", seed=15, max_negative_ratio=5)
    pipeline.fit(X, y)
    assert len(pipeline.named_steps["undersample"].sample_indices_) == len(y)
