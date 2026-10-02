"""Small model utilities shared by training and inference."""

from __future__ import annotations

import numpy as np


class FittedCycleLifeModel:
    """Wrap an estimator so ``predict`` always returns cycle-life units."""

    def __init__(self, estimator, target_scale="raw"):
        self.estimator = estimator
        self.target_scale = target_scale

    def predict(self, X):
        values = np.asarray(self.estimator.predict(X), dtype=float)
        return np.power(10.0, values) if self.target_scale == "log10" else values
