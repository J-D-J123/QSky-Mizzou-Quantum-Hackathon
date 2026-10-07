"""Stand-in estimators with fixed numbers, so orchestration tests never fit anything.

The class names match scikit-learn's because the portable export dispatches on them.
"""

import numpy as np


class StandardScaler:
    def __init__(self, mean, scale):
        self.mean_, self.scale_ = np.asarray(mean, dtype=float), np.asarray(scale, dtype=float)

    def transform(self, x):
        return (np.asarray(x, dtype=float) - self.mean_) / self.scale_


class PCA:
    whiten = False

    def __init__(self, components):
        self.components_ = np.asarray(components, dtype=float)
        self.mean_ = np.zeros(self.components_.shape[1])

    def transform(self, x):
        return (np.asarray(x, dtype=float) - self.mean_) @ self.components_.T


class Pipeline:
    def __init__(self, *steps):
        self.steps = [(type(step).__name__.lower(), step) for step in steps]

    def transform(self, x):
        for _, step in self.steps:
            x = step.transform(x)
        return x


class MLPClassifier:
    activation, out_activation_, n_outputs_ = "relu", "logistic", 1

    def __init__(self, inputs, seed):
        rng = np.random.default_rng(seed)
        self.coefs_ = [rng.normal(size=(inputs, 3)), rng.normal(size=(3, 1))]
        self.intercepts_ = [rng.normal(size=3), rng.normal(size=1)]

    def predict_proba(self, x):
        hidden = np.maximum(np.asarray(x, dtype=float) @ self.coefs_[0] + self.intercepts_[0], 0)
        p = 1 / (1 + np.exp(-(hidden @ self.coefs_[1] + self.intercepts_[1])[:, 0]))
        return np.column_stack([1 - p, p])


def preprocessor(train_raw, k):
    """Training-row mean/std and the first k columns: plain arithmetic, no estimator."""
    train_raw = np.asarray(train_raw, dtype=float)
    scale = train_raw.std(axis=0)
    scale[scale == 0] = 1.0
    transform = Pipeline(StandardScaler(train_raw.mean(axis=0), scale), PCA(np.eye(k, train_raw.shape[1])))
    return transform, transform.transform(train_raw)


def tabular(name, train_x, train_y, val_x, val_y, val_ids, *, seed, small_width, max_iter,
            selection="balanced_accuracy"):
    model = MLPClassifier(train_x.shape[1], seed + ord(name))
    return model, {
        "params": {"stand_in": True}, "val_clip_balanced_accuracy": 0.5, "selection_metric": selection,
        "val_selection_key": [0.5], "threshold": 0.5, "fit_seconds": 0.0, "parameter_count": 0,
        "support_vectors": None, "trials": [],
    }


def forbidden(*args, **kwargs):
    raise AssertionError("This test must not fit a model")
