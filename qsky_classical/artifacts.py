"""Version-stamped artifacts, a guarded loader, and pickle-free portable exports."""

import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import warnings

import joblib
import numpy as np

PACKAGES = ("numpy", "scipy", "librosa", "soundfile", "scikit-learn", "joblib", "torch")
PORTABLE_SCHEMA = "qsky-portable-v1"


class IncompatibleArtifactError(RuntimeError):
    pass


class PortableExportUnsupported(TypeError):
    pass


def versions():
    found = {"python": platform.python_version()}
    for package in PACKAGES:
        try:
            found[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            found[package] = None
    return found


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ids_sha256(ids):
    """Fingerprint of an ordered ID list; the same definition as Dhanya's result tables."""
    return hashlib.sha256("\n".join(str(i) for i in ids).encode("utf-8")).hexdigest()


def pickled_sklearn_versions(path):
    """Read the scikit-learn version(s) recorded in a pickle without unpickling it."""
    data = Path(path).read_bytes()
    found = re.findall(rb"_sklearn_version[^0-9]{0,6}(\d+\.\d+(?:\.\d+)?[0-9A-Za-z.]*)", data)
    return sorted({v.decode("ascii") for v in found})


def minor(version):
    return tuple(str(version).split(".")[:2])


def check_sklearn_artifact(path, *, allow_version_mismatch=False):
    """Refuse a scikit-learn pickle from another minor release unless explicitly allowed."""
    path = Path(path)
    runtime = importlib.metadata.version("scikit-learn")
    sidecar = Path(f"{path}.meta.json")
    if sidecar.is_file():
        meta = json.loads(sidecar.read_text())
        if meta.get("sha256") != sha256_file(path):
            raise IncompatibleArtifactError(f"{path} does not match the checksum in {sidecar.name}.")
        saved = [meta.get("versions", {}).get("scikit-learn")]
    else:
        saved = pickled_sklearn_versions(path)
    if saved and all(v and minor(v) == minor(runtime) for v in saved):
        return saved
    described = ", ".join(str(v) for v in saved) if saved else "unknown (none recorded)"
    message = (f"{path} was created with scikit-learn {described}; this environment has {runtime}. "
               "scikit-learn does not support loading pickles across versions: results may be "
               "wrong without any error. Use a matching environment or a portable JSON export.")
    if not allow_version_mismatch:
        raise IncompatibleArtifactError(message)
    warnings.warn("Loading despite version mismatch: " + message, RuntimeWarning, stacklevel=3)
    return saved


def save_bundle(path, bundle, **metadata):
    path = Path(path)
    joblib.dump(bundle, path)
    meta = {"sha256": sha256_file(path), "versions": versions(), **metadata}
    Path(f"{path}.meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def load_bundle(path, *, allow_version_mismatch=False):
    check_sklearn_artifact(path, allow_version_mismatch=allow_version_mismatch)
    return joblib.load(path)


class Standardizer:
    """Per-feature (x - mean) / scale with named, ordered features; no pickle needed.

    Matches StandardScaler: population standard deviation, and a zero-variance
    feature keeps scale 1.
    """

    def __init__(self, mean, scale, features):
        self.mean = np.asarray(mean, dtype=np.float64)
        self.scale = np.asarray(scale, dtype=np.float64)
        self.features = list(features)
        if not (len(self.mean) == len(self.scale) == len(self.features)):
            raise ValueError("mean, scale, and features must have the same length.")

    @classmethod
    def from_rows(cls, x, features):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2 or not len(x) or not np.isfinite(x).all():
            raise ValueError("Standardization needs a non-empty finite 2-D matrix.")
        scale = x.std(axis=0)
        scale[scale < 10 * np.finfo(np.float64).eps] = 1.0
        return cls(x.mean(axis=0), scale, features)

    def transform(self, x):
        return (np.asarray(x, dtype=np.float64) - self.mean) / self.scale

    def to_dict(self):
        return {"type": "standardize", "mean": self.mean.tolist(), "scale": self.scale.tolist(),
                "features": self.features}


def portable_steps(preprocessor):
    """Describe a fitted StandardScaler/PCA/MinMaxScaler pipeline as plain numbers."""
    if isinstance(preprocessor, Standardizer):
        return [preprocessor.to_dict()]
    steps = []
    for step in getattr(preprocessor, "steps", [("", preprocessor)]):
        step = step[1]
        kind = type(step).__name__
        if kind == "StandardScaler":
            steps.append({"type": "standardize", "mean": step.mean_.tolist(), "scale": step.scale_.tolist()})
        elif kind == "PCA":
            if step.whiten:
                raise PortableExportUnsupported("Portable export does not support whitened PCA.")
            steps.append({"type": "pca", "mean": step.mean_.tolist(), "components": step.components_.tolist()})
        elif kind == "MinMaxScaler":
            steps.append({"type": "minmax", "scale": step.scale_.tolist(), "min": step.min_.tolist(),
                          "clip": bool(step.clip), "range": [float(v) for v in step.feature_range]})
        else:
            raise PortableExportUnsupported(f"Portable export does not support preprocessing step {kind}.")
    return steps


def portable_model(model):
    """Describe a fitted binary RBF SVC or MLPClassifier as plain numbers."""
    kind = type(model).__name__
    if kind == "SVC":
        if model.kernel != "rbf" or len(model.classes_) != 2:
            raise PortableExportUnsupported("Portable export supports binary RBF SVC only.")
        return {"type": "rbf_svm", "gamma": float(model._gamma), "intercept": float(model.intercept_[0]),
                "support_vectors": model.support_vectors_.tolist(),
                "dual_coef": model.dual_coef_[0].tolist(), "score": "decision_function"}
    if kind == "MLPClassifier":
        if model.out_activation_ != "logistic" or model.n_outputs_ != 1:
            raise PortableExportUnsupported("Portable export supports binary MLPClassifier only.")
        return {"type": "mlp", "activation": model.activation, "score": "positive_probability",
                "weights": [w.tolist() for w in model.coefs_],
                "biases": [b.tolist() for b in model.intercepts_]}
    raise PortableExportUnsupported(f"Portable export does not support model type {kind}.")


def apply_steps(steps, x):
    x = np.asarray(x, dtype=np.float64)
    for step in steps:
        if step["type"] == "standardize":
            x = (x - np.asarray(step["mean"])) / np.asarray(step["scale"])
        elif step["type"] == "pca":
            x = (x - np.asarray(step["mean"])) @ np.asarray(step["components"]).T
        elif step["type"] == "minmax":
            x = x * np.asarray(step["scale"]) + np.asarray(step["min"])
            if step["clip"]:
                x = np.clip(x, *step["range"])
        else:
            raise ValueError(f"Unknown preprocessing step: {step['type']}")
    return x


ACTIVATIONS = {
    "relu": lambda z: np.maximum(z, 0), "tanh": np.tanh, "identity": lambda z: z,
    "logistic": lambda z: 1 / (1 + np.exp(-z)),
}


def model_scores(model, x):
    x = np.asarray(x, dtype=np.float64)
    if model["type"] == "rbf_svm":
        vectors = np.asarray(model["support_vectors"])
        distances = ((x[:, None, :] - vectors[None, :, :]) ** 2).sum(axis=2)
        return np.exp(-model["gamma"] * distances) @ np.asarray(model["dual_coef"]) + model["intercept"]
    if model["type"] == "mlp":
        layers = list(zip(model["weights"], model["biases"]))
        for weights, biases in layers[:-1]:
            x = ACTIVATIONS[model["activation"]](x @ np.asarray(weights) + np.asarray(biases))
        weights, biases = layers[-1]
        return ACTIVATIONS["logistic"](x @ np.asarray(weights) + np.asarray(biases))[:, 0]
    raise ValueError(f"Unknown model type: {model['type']}")


class PortablePredictor:
    """NumPy-only inference from a portable JSON export; independent of scikit-learn."""

    def __init__(self, spec):
        if spec.get("schema") != PORTABLE_SCHEMA:
            raise IncompatibleArtifactError(f"Unsupported portable schema: {spec.get('schema')!r}")
        self.spec = spec

    def scores(self, raw_features):
        return model_scores(self.spec["model"], apply_steps(self.spec["preprocessing"], raw_features))

    def predict(self, raw_features):
        return (self.scores(raw_features) >= self.spec["threshold"]).astype(int)


def export_portable(path, *, preprocessing, model, threshold, feature_order, **metadata):
    spec = {"schema": PORTABLE_SCHEMA, "feature_order": list(feature_order),
            "preprocessing": preprocessing, "model": model, "threshold": float(threshold),
            "versions": versions(), **metadata}
    Path(path).write_text(json.dumps(spec) + "\n")
    return spec


def load_portable(path):
    return PortablePredictor(json.loads(Path(path).read_text()))
