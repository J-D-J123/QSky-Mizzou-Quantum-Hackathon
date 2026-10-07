"""Shared-protocol checks. Nothing here fits an estimator: orchestration uses fixed stand-ins."""

import csv
import json

import numpy as np
import pytest
import soundfile as sf

from qsky_classical import cli
from qsky_classical.artifacts import load_bundle, load_portable
from qsky_classical.cli import SNR_PRESETS, parser, run
from qsky_classical.data import (
    drop_silent, features, load_windows, make_windows, mix_at_snr, read_manifest, split_clips, window_samples,
)
from qsky_classical.models import aggregate_clips, aggregate_features, mlp_parameter_count, selection_key
from qsky_classical.subsets import balanced_subsets, group_labels

import standins

SR = 16000


def tone(seconds, frequency=440.0, level=0.3):
    t = np.arange(int(SR * seconds)) / SR
    return level * np.sin(2 * np.pi * frequency * t)


@pytest.mark.parametrize("seconds, count", [(1, 4), (3, 2)])
def test_window_duration_is_configurable(tmp_path, seconds, count):
    path = tmp_path / "clip.wav"
    sf.write(path, tone(3.5), SR)
    windows = load_windows(path, seconds)
    assert windows.shape == (count, SR * seconds)
    np.testing.assert_allclose(np.max(np.abs(windows), axis=1), 1)
    assert features(windows).shape == (count, 26)
    assert features(windows, mel=True).shape == (count, 64, 100 * seconds + 1)
    # The default stays one second.
    assert load_windows(path).shape == (4, SR)


@pytest.mark.parametrize("seconds", [0, -1, 0.00001, 1.00003])
def test_invalid_window_duration_rejected(seconds):
    with pytest.raises(ValueError, match="window-seconds"):
        window_samples(seconds)


def write_manifest(path, rows, header=("path", "label", "group_id", "split")):
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    return path


def test_validation_split_name_is_normalized(tmp_path):
    rows = []
    for index, split in enumerate(["train", "validation", "Valid", "VAL", "test"]):
        sf.write(tmp_path / f"{index}.wav", tone(1), SR)
        rows.append([f"{index}.wav", index % 2, f"g{index}", split])
    clips = read_manifest(write_manifest(tmp_path / "clips.csv", rows))
    assert [c.split for c in clips] == ["train", "val", "val", "val", "test"]
    rows[0][3] = "holdout"
    with pytest.raises(ValueError, match="Invalid split"):
        read_manifest(write_manifest(tmp_path / "clips.csv", rows))


def test_shared_snr_preset_keeps_the_original_options():
    assert SNR_PRESETS["shared"] == ["clean", "20", "10", "5", "0"]
    assert SNR_PRESETS["legacy"] == ["clean", "20", "10", "0", "-10"]
    args = parser().parse_args(["--manifest", "unused.csv"])
    assert args.snrs is None and args.snr_preset == "legacy" and args.window_seconds == 1.0
    assert parser().parse_args(["--manifest", "unused.csv", "--snrs", "clean", "5", "-10"]).snrs == ["clean", "5", "-10"]


def test_five_db_mixture_on_three_second_window():
    rng = np.random.default_rng(3)
    signal, noise = rng.normal(size=3 * SR).astype(np.float32), rng.normal(size=3 * SR).astype(np.float32)
    mixed = mix_at_snr(signal, noise, 5)
    assert 10 * np.log10(np.mean(signal ** 2) / np.mean((mixed - signal) ** 2)) == pytest.approx(5, abs=1e-4)


def test_recording_ids_survive_windowing_and_silent_windows_are_recorded(tmp_path):
    audio = np.concatenate([tone(1), np.zeros(SR), tone(1, 880)])
    sf.write(tmp_path / "a.wav", audio, SR)
    sf.write(tmp_path / "b.wav", tone(2), SR)
    clips = read_manifest(write_manifest(tmp_path / "clips.csv", [["a.wav", 1, "flight-7", "test"],
                                                                  ["b.wav", 0, "field-2", "test"]]))
    windows = make_windows(clips, "test")
    assert list(windows.group_ids) == ["flight-7"] * 3 + ["field-2"] * 2
    kept, excluded = drop_silent(windows, "test")
    assert list(kept.group_ids) == ["flight-7"] * 2 + ["field-2"] * 2
    assert list(kept.window_ids) == [0, 2, 0, 1]
    assert excluded == [{"split": "test", "clip_id": str((tmp_path / "a.wav").resolve()), "group_id": "flight-7",
                         "window_id": 1, "reason": "silent_window", "raw_peak": 0.0}]
    # The undefined case itself still refuses to invent an SNR.
    with pytest.raises(ValueError, match="undefined"):
        mix_at_snr(windows.audio[1], windows.audio[0], 10)


def test_feature_mean_and_score_mean_are_different_procedures():
    x = np.array([[-1.0], [1.0], [2.0], [2.0]])
    labels, ids = np.array([1, 1, 0, 0]), np.array(["r1", "r1", "r2", "r2"])
    means, unit_labels, units = aggregate_features(x, labels, ids)
    np.testing.assert_allclose(means, [[0.0], [2.0]])
    np.testing.assert_array_equal(unit_labels, [1, 0])
    np.testing.assert_array_equal(units, ["r1", "r2"])
    scorer = lambda values: values[:, 0] ** 2
    _, score_mean, _ = aggregate_clips(labels, scorer(x), ids)
    # Same windows, same scorer: averaging features first gives a different recording score.
    np.testing.assert_allclose(score_mean, [1.0, 4.0])
    np.testing.assert_allclose(scorer(means), [0.0, 4.0])
    with pytest.raises(ValueError, match="conflicting labels"):
        aggregate_features(x, np.array([1, 0, 0, 0]), ids)


def test_training_subsets_are_balanced_nested_and_shared():
    labels = group_labels([f"g{i}" for i in range(20)], [i % 2 for i in range(20)])
    first = balanced_subsets(labels, [4, 8, 10], seed=5)
    assert first == balanced_subsets(labels, [4, 8, 10], seed=5)
    assert first != balanced_subsets(labels, [4, 8, 10], seed=6)
    for size, groups in first.items():
        assert len(groups) == size and sum(labels[g] for g in groups) == size // 2
    assert set(first[4]) <= set(first[8]) <= set(first[10])
    with pytest.raises(ValueError, match="exceeds the largest balanced subset"):
        balanced_subsets(labels, [22], seed=5)
    with pytest.raises(ValueError, match="even"):
        balanced_subsets(labels, [5], seed=5)
    with pytest.raises(ValueError, match="mixed labels"):
        group_labels(["g", "g"], [0, 1])


def test_parameter_counts_and_selection_keys():
    assert mlp_parameter_count(4, (4,)) == 25 == (4 + 2) * 4 + 1
    assert mlp_parameter_count(6, (4,)) == 33
    assert mlp_parameter_count(4, (64, 32)) == 2433
    labels, ids = np.array([1, 1, 0, 0]), np.array(["a", "b", "c", "d"])
    scores = np.array([0.9, 0.2, 0.1, 0.8])
    assert selection_key(labels, scores, ids, 0.5) == (0.5,)
    assert selection_key(labels, scores, ids, 0.5, "f1") == (0.5, 0.5, 0.5)
    with pytest.raises(ValueError, match="selection metric"):
        selection_key(labels, scores, ids, 0.5, "accuracy")


@pytest.fixture
def dataset(tmp_path):
    """Six recordings per class and split, each two 6 s clips; one test window is silent."""
    rng = np.random.default_rng(11)
    paths = {}
    for noise in (False, True):
        rows = []
        for split in ("train", "val", "test"):
            for label in (0, 1):
                for group in range(3 if noise else 6):
                    for part in range(1 if noise else 2):
                        name = f"{'n' if noise else 's'}-{split}-{label}-{group}-{part}.wav"
                        audio = 0.05 * rng.normal(size=6 * SR)
                        if not noise:
                            audio += tone(6, 200 + 900 * label + 7 * group)
                        if name == "s-test-1-0-0.wav":
                            audio[3 * SR:] = 0
                        sf.write(tmp_path / name, audio, SR)
                        rows.append([name, label, f"{'n' if noise else 's'}-{split}-{label}-{group}",
                                     "validation" if split == "val" else split])
        paths["noise" if noise else "clips"] = write_manifest(tmp_path / ("noise.csv" if noise else "clips.csv"), rows)
    return paths


def test_dry_run_plans_everything_and_fits_nothing(dataset, tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "fit_tabular", standins.forbidden)
    monkeypatch.setattr(cli, "fit_preprocessor", standins.forbidden)
    output = tmp_path / "dry"
    args = parser().parse_args([
        "--manifest", str(dataset["clips"]), "--noise-manifest", str(dataset["noise"]), "--models", "all",
        "--window-seconds", "3", "--snr-preset", "shared", "--train-recordings", "4", "8", "--seeds", "1", "2",
        "--dry-run", "--output", str(output)])
    assert run(args) == []
    config = json.loads((output / "config.json").read_text())
    assert config["audio"]["window_seconds"] == 3.0
    assert config["snrs"] == ["clean", "20", "10", "5", "0"]
    assert config["thresholds"] == {"A": 0.0, "B": 0.5, "C": 0.5, "D": 0.5}
    assert config["seeds"] == {"split": 42, "train_noise": 43, "test_noise": 44, "runs": [1, 2]}
    assert config["parameter_counts"]["B"] == {"2": 17, "4": 25, "6": 33}
    assert len(config["feature_order"]) == 26 and config["versions"]["scikit-learn"]
    assert "richer-input" in config["preprocessing"]["D"]
    plan = json.loads((output / "plan.json").read_text())
    # Two seeds x two sizes x (three tabular models x three PCA widths + D).
    assert len(plan) == 2 * 2 * (3 * 3 + 1)
    splits = json.loads((output / "split_ids.json").read_text())
    assert not set(splits["train"]) & set(splits["val"]) and not set(splits["val"]) & set(splits["test"])
    subsets = json.loads((output / "subsets.json").read_text())
    assert [(s["seed"], s["train_recordings"]) for s in subsets] == [(1, 4), (1, 8), (2, 4), (2, 8)]
    assert all(set(s["group_ids"]) <= set(splits["train"]) for s in subsets)
    with (output / "exclusions.csv").open() as handle:
        excluded = list(csv.DictReader(handle))
    assert [(r["split"], r["group_id"], r["window_id"], r["reason"]) for r in excluded] == [
        ("test", "s-test-1-0", "1", "silent_window")]
    assert not (output / "results.csv").exists() and not list(output.rglob("*.joblib"))


def read_csv(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def test_sweep_scores_identical_examples_for_every_model_and_condition(dataset, tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "fit_tabular", standins.tabular)
    monkeypatch.setattr(cli, "fit_preprocessor", standins.preprocessor)
    output = tmp_path / "sweep"
    args = parser().parse_args([
        "--manifest", str(dataset["clips"]), "--noise-manifest", str(dataset["noise"]), "--models", "A", "B",
        "--pca", "2", "4", "--window-seconds", "3", "--snr-preset", "shared", "--levels", "window", "clip", "recording",
        "--train-recordings", "4", "--seeds", "3", "4", "--output", str(output)])
    rows = run(args)
    # seeds x PCA widths x models x SNRs x levels
    assert len(rows) == 2 * 2 * 2 * 5 * 3
    assert {r["snr_db"] for r in rows} == {"clean", "20", "10", "5", "0"}
    counts = {(r["level"], r["n_examples"]) for r in rows}
    # 24 test clips of two windows, minus the one silent window, in 12 recordings.
    assert counts == {("window", 47), ("clip", 24), ("recording", 12)}
    predictions = read_csv(output / "predictions.csv")
    scored = {}
    for p in predictions:
        if p["level"] == "window":
            scored.setdefault((p["model"], p["pca_k"], p["seed"], p["snr_db"]), set()).add((p["clip_id"], p["window_id"]))
    assert len(scored) == 2 * 2 * 2 * 5 and len({frozenset(v) for v in scored.values()}) == 1
    silent = read_csv(output / "exclusions.csv")
    assert len(silent) == 1 and (silent[0]["clip_id"], silent[0]["window_id"]) not in next(iter(scored.values()))

    # A recording's score is the mean of its window scores.
    key = lambda p: p["model"] == "A" and p["pca_k"] == "2" and p["seed"] == "3" and p["snr_db"] == "10"
    by_group = {}
    for p in predictions:
        if key(p) and p["level"] == "window":
            by_group.setdefault(p["group_id"], []).append(float(p["score"]))
    for p in predictions:
        if key(p) and p["level"] == "recording":
            assert float(p["score"]) == pytest.approx(np.mean(by_group[p["group_id"]]))

    subsets = {s["seed"]: set(s["group_ids"]) for s in json.loads((output / "subsets.json").read_text())}
    for seed in (3, 4):
        artifacts = output / f"seed{seed}_n4" / "artifacts"
        arrays = np.load(artifacts / "features_k2.npz")
        assert set(arrays["train_group_ids"]) == subsets[seed] and arrays["train_x"].shape[1] == 2
        assert not set(arrays["train_group_ids"]) & (set(arrays["val_group_ids"]) | set(arrays["test_group_ids"]))
        meta = json.loads((artifacts / "model_A_k2.joblib.meta.json").read_text())
        assert meta["versions"]["scikit-learn"] and meta["feature_order"][0] == "mfcc_1_mean"
        assert set(load_bundle(artifacts / "model_A_k2.joblib")) == {"preprocessor", "model", "threshold"}
    noise = read_csv(output / "noise_assignment.csv")
    assert {r["split"] for r in noise} == {"test"} and all(r["noise_group_id"].startswith("n-test") for r in noise)

    # The pickle-free export reproduces the saved clean window scores from raw features.
    portable = load_portable(output / "seed3_n4" / "artifacts" / "model_B_k4.portable.json")
    clip = str((dataset["clips"].parent / "s-test-0-2-0.wav").resolve())
    expected = [float(p["score"]) for p in predictions
                if p["model"] == "B" and p["pca_k"] == "4" and p["seed"] == "3" and p["snr_db"] == "clean"
                and p["level"] == "window" and p["clip_id"] == clip]
    np.testing.assert_allclose(portable.scores(features(load_windows(clip, 3))), expected, rtol=1e-6)
    assert portable.spec["feature_order"][13] == "mfcc_1_std" and portable.spec["model_id"] == "B"


def test_feature_mean_aggregation_trains_and_scores_one_example_per_recording(dataset, tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "fit_tabular", standins.tabular)
    monkeypatch.setattr(cli, "fit_preprocessor", standins.preprocessor)
    output = tmp_path / "feature-mean"
    args = parser().parse_args([
        "--manifest", str(dataset["clips"]), "--noise-manifest", str(dataset["noise"]), "--models", "A",
        "--pca", "2", "--window-seconds", "3", "--snr-preset", "shared", "--train-snrs", "10",
        "--aggregation", "feature-mean", "--output", str(output)])
    rows = run(args)
    assert len(rows) == 5
    assert {(r["level"], r["aggregation"], r["n_examples"]) for r in rows} == {
        ("recording", "feature_mean_per_recording", 12)}
    arrays = np.load(output / "artifacts" / "features_k2.npz")
    # Twelve training recordings, each once clean and once at 10 dB.
    assert arrays["train_x"].shape == (24, 2)
    assert sorted(arrays["train_snr_db"]) == ["10"] * 12 + ["clean"] * 12
    assert len(set(arrays["train_group_ids"])) == 12 and arrays["val_x"].shape == (12, 2)
    config = json.loads((output / "config.json").read_text())
    assert config["selection_metric"] == "clean validation recording balanced accuracy"
    assert config["aggregation"]["A-C"] != config["aggregation"]["D"]


def test_premixed_conditions_and_missing_copies(dataset, tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "fit_tabular", standins.forbidden)
    clips = [c for c in split_clips(read_manifest(dataset["clips"])) if c.split == "test"]
    rows = []
    for clip in clips[1:]:
        mixed = tmp_path / f"mixed-{clip.path.name}"
        sf.write(mixed, 0.5 * sf.read(clip.path)[0], SR, subtype="FLOAT")
        rows.append([mixed.name, clip.path.name])
    premixed = write_manifest(tmp_path / "premixed_10.csv", rows, header=("path", "source_path"))
    common = ["--manifest", str(dataset["clips"]), "--models", "A", "--window-seconds", "3",
              "--snrs", "clean", "10", "--premixed-test", f"10={premixed}", "--dry-run"]
    run(parser().parse_args([*common, "--output", str(tmp_path / "premixed")]))
    excluded = read_csv(tmp_path / "premixed" / "exclusions.csv")
    # The clip without a mixed copy is dropped from every condition, clean included.
    assert {r["reason"] for r in excluded if r["clip_id"] == str(clips[0].path)} == {"no_premixed_copy"}
    assert len([r for r in excluded if r["reason"] == "no_premixed_copy"]) == 2
    with pytest.raises(ValueError, match="no premixed copy"):
        run(parser().parse_args([*common, "--silent-policy", "error", "--output", str(tmp_path / "strict")]))
    with pytest.raises(ValueError, match="not evaluated"):
        run(parser().parse_args(["--manifest", str(dataset["clips"]), "--snrs", "clean",
                                 "--premixed-test", f"10={premixed}", "--output", str(tmp_path / "unused")]))


def test_legacy_error_policy_still_aborts_on_a_silent_window(dataset, tmp_path):
    args = parser().parse_args([
        "--manifest", str(dataset["clips"]), "--noise-manifest", str(dataset["noise"]), "--window-seconds", "3",
        "--silent-policy", "error", "--dry-run", "--output", str(tmp_path / "legacy")])
    with pytest.raises(ValueError, match="undefined"):
        run(args)


def test_cnn_accepts_any_window_length_and_keeps_its_checkpoint_layout():
    """Inference on an untrained network only; nothing is trained."""
    torch = pytest.importorskip("torch")
    from qsky_classical.cnn import CNNModel, GlobalAveragePool, LogMelCNN

    torch.manual_seed(0)
    network = LogMelCNN()
    model = CNNModel(network, mean=-40.0, std=20.0, batch_size=4)
    for frames in (101, 301):  # one- and three-second windows
        x = np.random.default_rng(frames).normal(-40, 20, size=(5, 64, frames)).astype(np.float32)
        assert model.predict_proba(x).shape == (5, 2)
    # The pooling layer has no weights, so earlier checkpoints still load.
    assert list(network.state_dict()) == [f"layers.{i}.{part}" for i in (0, 3, 6, 10, 13) for part in ("weight", "bias")]
    maps = torch.randn(3, 64, 16, 75)
    np.testing.assert_allclose(GlobalAveragePool()(maps).numpy(),
                               torch.nn.AdaptiveAvgPool2d((1, 1))(maps).numpy(), atol=1e-6)
