import numpy as np
import pytest

from seizure_detection.foundation import _bipolar_channel_info
from seizure_detection.models import MODEL_NAMES, make_model


def test_pretrained_montage_uses_all_bipolar_channel_midpoints() -> None:
    mne = pytest.importorskip("mne")
    channel_info = _bipolar_channel_info(mne, target_sfreq=250.0)
    assert len(channel_info) == 18
    for item in channel_info:
        position = np.asarray(item["loc"][:3])
        assert np.isfinite(position).all()
        assert np.linalg.norm(position) == pytest.approx(0.09, abs=1e-8)


def test_all_documented_classical_models_have_valid_pipelines() -> None:
    for name in MODEL_NAMES:
        estimator, parameter_grid = make_model(name, seed=2)
        assert "undersample" in estimator.named_steps
        assert "classifier" in estimator.named_steps
        assert parameter_grid
