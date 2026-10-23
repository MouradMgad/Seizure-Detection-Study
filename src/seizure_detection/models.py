"""Classical estimator definitions used by the grouped benchmark."""

from __future__ import annotations

from functools import partial
import math
from typing import Any

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler


def _negative_ratio_strategy(y: np.ndarray, max_negative_ratio: float) -> dict[int, int]:
    counts = np.bincount(np.asarray(y, dtype=np.int64), minlength=2)
    if counts[1] == 0:
        raise ValueError("Training data must contain seizure-positive windows.")
    target_negative_count = min(
        int(counts[0]),
        int(np.ceil(max_negative_ratio * counts[1])),
    )
    return {0: target_negative_count}


def make_model(
    name: str,
    seed: int,
    max_negative_ratio: float = 5.0,
) -> tuple[Pipeline, dict[str, list[Any]]]:
    """Return a train-only sampling pipeline and a compact hyperparameter grid."""
    if not math.isfinite(max_negative_ratio) or max_negative_ratio < 1:
        raise ValueError("max_negative_ratio must be at least 1.")
    scale = False
    if name == "logistic":
        classifier = LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            solver="lbfgs",
            random_state=seed,
        )
        scale = True
        grid = {"classifier__C": [0.1, 1.0, 10.0]}
    elif name == "rbf_svm":
        classifier = SVC(
            class_weight="balanced",
            kernel="rbf",
            probability=True,
            random_state=seed,
        )
        scale = True
        grid = {"classifier__C": [0.1, 1.0, 10.0], "classifier__gamma": ["scale", 0.01]}
    elif name == "random_forest":
        classifier = RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=seed,
        )
        grid = {"classifier__max_features": ["sqrt", 0.5]}
    elif name == "extra_trees":
        classifier = ExtraTreesClassifier(
            n_estimators=300,
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed,
        )
        grid = {"classifier__max_features": ["sqrt", 0.5]}
    elif name == "hist_gradient_boosting":
        classifier = HistGradientBoostingClassifier(
            class_weight="balanced",
            max_iter=150,
            random_state=seed,
        )
        grid = {
            "classifier__learning_rate": [0.05, 0.1],
            "classifier__max_leaf_nodes": [15, 31],
        }
    else:
        raise ValueError(f"Unknown model {name!r}.")
    common = [
        (
            "undersample",
            RandomUnderSampler(
                sampling_strategy=partial(
                    _negative_ratio_strategy,
                    max_negative_ratio=max_negative_ratio,
                ),
                random_state=seed,
            ),
        )
    ]
    common.append(("imputer", SimpleImputer(strategy="median")))
    if scale:
        common.append(("scale", StandardScaler()))
    common.append(("classifier", classifier))
    return Pipeline(common), grid


MODEL_NAMES = (
    "logistic",
    "rbf_svm",
    "random_forest",
    "extra_trees",
    "hist_gradient_boosting",
)
