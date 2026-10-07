"""Artifact versioning, the guarded loader, and NumPy-only portable inference."""

import importlib.metadata
import json

import joblib
import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

from qsky_classical.artifacts import (
    IncompatibleArtifactError, PortableExportUnsupported, PortablePredictor, Standardizer, apply_steps,
    check_sklearn_artifact, export_portable, ids_sha256, load_bundle, load_portable, model_scores,
    pickled_sklearn_versions, portable_model, portable_steps, save_bundle, versions,
)

RUNTIME = importlib.metadata.version("scikit-learn")


def test_versions_record_the_environment():
    found = versions()
    assert found["scikit-learn"] == RUNTIME and found["numpy"] and found["python"]
    assert "torch" in found  # None when PyTorch is not installed


def test_pickle_version_is_read_without_unpickling(tmp_path):
    # The byte layout found in Dhanya's saved scaler and SVM.
    foreign = tmp_path / "foreign.pkl"
    foreign.write_bytes(b"\x80\x05junk_sklearn_version\x94\x8c\x051.9.1\x94ub.")
    assert pickled_sklearn_versions(foreign) == ["1.9.1"]
    local = tmp_path / "local.joblib"
    joblib.dump(StandardScaler(), local)  # unfitted: only its version stamp matters
    assert pickled_sklearn_versions(local) == [RUNTIME]


def test_incompatible_or_unversioned_pickles_are_refused(tmp_path):
    other = "1.9.1" if not RUNTIME.startswith("1.9.") else "1.4.0"
    foreign = tmp_path / "foreign.pkl"
    foreign.write_bytes(b"_sklearn_version\x94\x8c\x05" + other.encode() + b"\x94ub.")
    with pytest.raises(IncompatibleArtifactError, match=f"created with scikit-learn {other}"):
        check_sklearn_artifact(foreign)
    with pytest.raises(IncompatibleArtifactError, match=other):
        load_bundle(foreign)
    with pytest.warns(RuntimeWarning, match="Loading despite version mismatch"):
        assert check_sklearn_artifact(foreign, allow_version_mismatch=True) == [other]
    unknown = tmp_path / "unknown.pkl"
    unknown.write_bytes(b"no version recorded")
    with pytest.raises(IncompatibleArtifactError, match="unknown"):
        check_sklearn_artifact(unknown)


def test_saved_bundles_carry_versions_and_a_checksum(tmp_path):
    path = tmp_path / "bundle.joblib"
    meta = save_bundle(path, {"preprocessor": StandardScaler(), "threshold": 0.0}, feature_order=["a", "b"])
    assert meta["versions"]["scikit-learn"] == RUNTIME and meta["feature_order"] == ["a", "b"]
    assert load_bundle(path)["threshold"] == 0.0
    sidecar = tmp_path / "bundle.joblib.meta.json"
    recorded = json.loads(sidecar.read_text())
    recorded["versions"]["scikit-learn"] = "0.24.2"
    sidecar.write_text(json.dumps(recorded))
    with pytest.raises(IncompatibleArtifactError, match="0.24.2"):
        load_bundle(path)
    recorded["versions"]["scikit-learn"] = RUNTIME
    sidecar.write_text(json.dumps(recorded))
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(IncompatibleArtifactError, match="checksum"):
        load_bundle(path)


def test_id_fingerprint_matches_her_definition():
    import hashlib
    ids = ["7a2c8a3f64abeba8e910", "7b5c623eaac7bdf9ce5f"]
    assert ids_sha256(ids) == hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()
    assert ids_sha256(ids) != ids_sha256(ids[::-1])


def test_standardizer_uses_population_statistics():
    rows = np.array([[1.0, 5.0, 2.0], [3.0, 5.0, 4.0], [5.0, 5.0, 9.0]])
    scaler = Standardizer.from_rows(rows, ["a", "b", "c"])
    np.testing.assert_allclose(scaler.mean, [3.0, 5.0, 5.0])
    np.testing.assert_allclose(scaler.scale, [np.sqrt(8 / 3), 1.0, np.sqrt(26 / 3)])
    np.testing.assert_allclose(scaler.transform(rows).mean(axis=0), 0, atol=1e-12)
    np.testing.assert_allclose(apply_steps([scaler.to_dict()], rows), scaler.transform(rows))
    with pytest.raises(ValueError, match="non-empty finite"):
        Standardizer.from_rows(np.array([[np.nan, 1.0]]), ["a", "b"])


def test_portable_steps_and_models_follow_their_formulas():
    x = np.array([[1.0, 2.0], [3.0, -1.0]])
    pca = {"type": "pca", "mean": [1.0, 1.0], "components": [[0.0, 1.0]]}
    np.testing.assert_allclose(apply_steps([pca], x), [[1.0], [-2.0]])
    minmax = {"type": "minmax", "scale": [2.0], "min": [0.5], "clip": True, "range": [0.0, np.pi]}
    np.testing.assert_allclose(apply_steps([pca, minmax], x), [[2.5], [0.0]])
    svm = {"type": "rbf_svm", "gamma": 1.0, "intercept": -1.0, "support_vectors": [[0.0, 0.0]], "dual_coef": [2.0]}
    np.testing.assert_allclose(model_scores(svm, [[0.0, 0.0], [1.0, 0.0]]), [1.0, 2 * np.exp(-1) - 1])
    mlp = {"type": "mlp", "activation": "relu", "weights": [[[1.0], [-1.0]], [[2.0]]], "biases": [[0.0], [-1.0]]}
    # hidden = relu(x1 - x2); output = sigmoid(2 * hidden - 1)
    np.testing.assert_allclose(model_scores(mlp, x), [1 / (1 + np.exp(1.0)), 1 / (1 + np.exp(-7.0))])


def test_portable_export_round_trip_and_refusals(tmp_path):
    spec = export_portable(
        tmp_path / "model.portable.json", feature_order=["a", "b"], threshold=0.0,
        preprocessing=[{"type": "standardize", "mean": [1.0, 0.0], "scale": [2.0, 1.0]}],
        model={"type": "rbf_svm", "gamma": 0.5, "intercept": -0.5, "support_vectors": [[0.0, 0.0]], "dual_coef": [1.0]},
        aggregation="average the window features of a recording, then classify it once")
    assert spec["versions"]["scikit-learn"] == RUNTIME
    predictor = load_portable(tmp_path / "model.portable.json")
    np.testing.assert_allclose(predictor.scores([[1.0, 0.0], [5.0, 0.0]]), [0.5, np.exp(-2.0) - 0.5])
    np.testing.assert_array_equal(predictor.predict([[1.0, 0.0], [5.0, 0.0]]), [1, 0])
    with pytest.raises(IncompatibleArtifactError, match="schema"):
        PortablePredictor({"schema": "something-else"})
    with pytest.raises(PortableExportUnsupported, match="model type"):
        portable_model(object())
    with pytest.raises(PortableExportUnsupported, match="preprocessing step"):
        portable_steps(object())


@pytest.mark.fits
def test_portable_export_matches_scikit_learn():
    """Fits tiny estimators: run on the training machine before relying on portable exports."""
    from sklearn.decomposition import PCA
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.svm import SVC

    rng = np.random.default_rng(0)
    x = rng.normal(size=(60, 6))
    y = (x[:, 0] + x[:, 1] ** 2 > 0.5).astype(int)
    pipeline = make_pipeline(StandardScaler(), PCA(n_components=4, svd_solver="full"),
                             MinMaxScaler(feature_range=(0, np.pi), clip=True))
    z = pipeline.fit_transform(x)
    steps = portable_steps(pipeline)
    np.testing.assert_allclose(apply_steps(steps, x), z, atol=1e-9)
    for model in (SVC(C=1.0, gamma="scale", class_weight="balanced").fit(z, y),
                  SVC(C=10.0, gamma=0.1).fit(z, y),
                  MLPClassifier(hidden_layer_sizes=(4,), solver="lbfgs", max_iter=50, random_state=0).fit(z, y),
                  MLPClassifier(hidden_layer_sizes=(64, 32), solver="lbfgs", max_iter=50, random_state=0).fit(z, y)):
        expected = model.decision_function(z) if isinstance(model, SVC) else model.predict_proba(z)[:, 1]
        np.testing.assert_allclose(model_scores(portable_model(model), z), expected, atol=1e-8)
    scaler = StandardScaler().fit(x)
    np.testing.assert_allclose(Standardizer.from_rows(x, list("abcdef")).transform(x), scaler.transform(x), atol=1e-12)
