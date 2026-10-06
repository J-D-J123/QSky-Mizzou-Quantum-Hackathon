import csv
from dataclasses import replace
import importlib.util

import joblib
import numpy as np
import pytest
import soundfile as sf
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from qsky_classical.cli import parser, run
from qsky_classical.data import Clip, features, load_windows, mix_at_snr, read_manifest, split_clips
from qsky_classical.models import aggregate_clips


def test_recording_groups_stay_together():
    clips = [Clip(f"{label}-{group}-{part}.wav", label, f"{label}-{group}")
             for label in (0, 1) for group in range(15) for part in range(3)]
    result = split_clips(clips, seed=42)
    assert result == split_clips(clips, seed=42)
    group_splits = {}
    for clip in result:
        group_splits.setdefault(clip.group_id, set()).add(clip.split)
    assert all(len(splits) == 1 for splits in group_splits.values())
    assert {c.split for c in result} == {"train", "val", "test"}
    broken = [replace(c, split="test") if i == 0 else c for i, c in enumerate(result)]
    # Force an explicit recording-group conflict, regardless of the random assignment.
    broken[1] = replace(broken[1], split="train")
    with pytest.raises(ValueError, match="crosses splits"):
        split_clips(broken)


@pytest.mark.parametrize("snr", [20, 10, 0, -10])
def test_measured_snr(snr):
    rng = np.random.default_rng(2)
    signal = rng.normal(size=16000).astype(np.float32)
    noise = rng.normal(size=16000).astype(np.float32)
    mixed = mix_at_snr(signal, noise, snr)
    measured = 10 * np.log10(np.mean(signal ** 2) / np.mean((mixed - signal) ** 2))
    assert measured == pytest.approx(snr, abs=1e-4)


def test_silent_snr_is_not_fabricated():
    with pytest.raises(ValueError, match="undefined"):
        mix_at_snr(np.zeros(16000), np.ones(16000), 10)


def test_mono_resampling_padding_and_features(tmp_path):
    time = np.arange(22050 + 5512) / 22050
    audio = 0.2 * np.sin(2 * np.pi * 440 * time)
    path = tmp_path / "stereo.wav"
    sf.write(path, np.column_stack([audio, audio * 0.5]), 22050)
    windows = load_windows(path)
    assert windows.shape == (2, 16000)
    np.testing.assert_allclose(np.max(np.abs(windows), axis=1), 1)
    assert np.all(windows[1, 5000:] == 0)
    assert features(windows).shape == (2, 26)
    assert features(windows, mel=True).shape == (2, 64, 101)


def test_training_statistics_and_angle_bounds():
    rng = np.random.default_rng(4)
    train = rng.normal(size=(30, 26))
    test = rng.normal(size=(10, 26)) + 1000
    transform = make_pipeline(StandardScaler(), PCA(n_components=4),
                              MinMaxScaler(feature_range=(0, np.pi), clip=True))
    transform.fit(train)
    original_mean = transform[0].mean_.copy()
    angles = transform.transform(test)
    np.testing.assert_array_equal(transform[0].mean_, original_mean)
    np.testing.assert_allclose(original_mean, train.mean(axis=0))
    assert np.all((angles >= 0) & (angles <= np.pi))


def test_clip_scores_are_averaged():
    labels, scores, ids = aggregate_clips(np.array([1, 0, 1]), np.array([0.2, 0.3, 0.8]),
                                           np.array(["a", "b", "a"]))
    np.testing.assert_array_equal(labels, [1, 0])
    np.testing.assert_allclose(scores, [0.5, 0.3])
    np.testing.assert_array_equal(ids, ["a", "b"])


@pytest.fixture
def manifests(tmp_path):
    rng = np.random.default_rng(17)
    t = np.arange(16000) / 16000
    paths = []
    for is_noise in (False, True):
        manifest = tmp_path / ("noise.csv" if is_noise else "clips.csv")
        with manifest.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["path", "label", "group_id", "split"])
            for split in ("train", "val", "test"):
                for label in (0, 1):
                    for index in range(2):
                        name = f"{is_noise}-{split}-{label}-{index}.wav"
                        audio = 0.1 * rng.normal(size=len(t))
                        if not is_noise:
                            audio += 0.4 * np.sin(2 * np.pi * (200 + label * 1000 + index * 10) * t)
                        sf.write(tmp_path / name, audio, 16000)
                        writer.writerow([name, label, name, split])
        paths.append(manifest)
    return paths


def test_manifest_rejects_duplicate_paths(manifests):
    path = manifests[0]
    with path.open() as handle:
        lines = handle.readlines()
    with path.open("a") as handle:
        handle.write(lines[1])
    with pytest.raises(ValueError, match="Duplicate"):
        read_manifest(path)


def test_missing_noise_fails_before_output(manifests, tmp_path):
    output = tmp_path / "bad"
    args = parser().parse_args(["--manifest", str(manifests[0]), "--output", str(output)])
    with pytest.raises(ValueError, match="noise-manifest"):
        run(args)
    assert not output.exists()


def test_complete_comparison_and_saved_preprocessor(manifests, tmp_path):
    # Three PCA widths require at least six train windows; duplicate the training
    # clip duration rather than introduce windows from validation/test.
    for clip in read_manifest(manifests[0]):
        if clip.split == "train":
            audio, sr = sf.read(clip.path)
            sf.write(clip.path, np.tile(audio, 2), sr)
    has_torch = importlib.util.find_spec("torch") is not None
    selected = ["all"] if has_torch else ["A", "B", "C"]
    output = tmp_path / "run"
    args = parser().parse_args([
        "--manifest", str(manifests[0]), "--noise-manifest", str(manifests[1]),
        "--models", *selected, "--pca", "2", "4", "6", "--train-snrs", "10",
        "--epochs", "1", "--max-iter", "50", "--output", str(output),
    ])
    rows = run(args)
    # Each model/PCA pair has five SNRs and both window and clip metrics.
    assert len(rows) == (10 if has_torch else 9) * 5 * 2
    assert {r["model"] for r in rows} == set("ABCD" if has_torch else "ABC")
    assert {r["snr_db"] for r in rows} == {"clean", "20", "10", "0", "-10"}
    assert all(np.isfinite(r["accuracy"]) for r in rows)
    arrays = np.load(output / "artifacts/features_k2.npz")
    assert arrays["train_x"].shape == (16, 2)
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        assert not set(arrays[f"{a}_clip_ids"]) & set(arrays[f"{b}_clip_ids"])
    bundle = joblib.load(output / "artifacts/model_A_k2.joblib")
    raw_train = np.concatenate([features(load_windows(c.path)) for c in read_manifest(manifests[0]) if c.split == "train"])
    np.testing.assert_allclose(bundle["preprocessor"].transform(raw_train), arrays["train_x"][:8], atol=1e-5)
    assert (output / "results.csv").is_file()
    assert (output / "selection.json").is_file()
    assert (output / "predictions.csv").is_file()
    with pytest.raises(ValueError, match="new or empty"):
        run(args)


def test_cnn_saved_predictions_match(manifests, tmp_path):
    torch = pytest.importorskip("torch")
    from qsky_classical.cnn import CNNModel, LogMelCNN

    output = tmp_path / "cnn"
    args = parser().parse_args(["--manifest", str(manifests[0]), "--models", "D", "--snrs", "clean",
                               "--epochs", "1", "--output", str(output)])
    run(args)
    saved = torch.load(output / "artifacts/model_D.pt", weights_only=True)
    network = LogMelCNN()
    network.load_state_dict(saved["state_dict"])
    model = CNNModel(network, saved["mean"], saved["std"], saved["batch_size"])
    test = [c for c in read_manifest(manifests[0]) if c.split == "test"]
    x = np.concatenate([features(load_windows(c.path), mel=True) for c in test])
    with (output / "predictions.csv").open() as handle:
        scores = [float(r["score"]) for r in csv.DictReader(handle) if r["level"] == "window"]
    np.testing.assert_allclose(model.predict_proba(x)[:, 1], scores)
