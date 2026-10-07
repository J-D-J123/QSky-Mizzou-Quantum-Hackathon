"""Matched selected-feature inputs for A-C: reproduction versus the strict label-budget protocol."""

import csv
import json

import numpy as np
import pytest

from qsky_classical import matched
from qsky_classical.artifacts import load_portable
from qsky_classical.dhanya import CANDIDATE_FEATURES, ImportValidationError, import_assets
from qsky_classical.matched import build_inputs, combine_ranks, load_bundle, parser, rank_features, run

import her_assets
import standins

COUNTS = {"val": 6, "test": 9}


@pytest.fixture
def imported(tmp_path):
    root = tmp_path / "her"
    root.mkdir()
    truth = her_assets.build(root)
    import_assets(root, tmp_path / "bundle", feature_counts=[4, 6], expected=COUNTS)
    return tmp_path / "bundle", truth


def recording_means(rows, names, recordings, snr, transform=lambda x: x):
    result = []
    for recording in recordings:
        values = [[r[n] for n in names] for r in rows if r["recording_id"] == recording and str(r["snr_db"]) == snr]
        result.append(transform(np.array(values, dtype=float)).mean(axis=0))
    return np.array(result)


def test_ranks_combine_like_her_feature_selection():
    ranked = combine_ranks(["a", "b", "c", "d"], [0.9, 0.5, 0.5, 0.1], [1.0, 9.0, 3.0, 3.0])
    # Tied scores share the best rank ("min" ranking); the mean of both ranks orders features.
    assert [r["feature"] for r in ranked] == ["b", "c", "a", "d"]
    assert [(r["mutual_information_rank"], r["anova_rank"], r["mean_rank"]) for r in ranked] == [
        (2, 1, 1.5), (2, 2, 2.0), (1, 4, 2.5), (4, 2, 3.0)]
    # Equal mean ranks fall back to the mutual-information rank.
    assert [r["feature"] for r in combine_ranks(["x", "y"], [0.1, 0.9], [9.0, 1.0])] == ["y", "x"]


def test_ranking_prefers_the_informative_feature():
    rng = np.random.default_rng(1)
    y = np.repeat([0, 1], 20)
    x = np.column_stack([rng.normal(size=40), y * 4 + rng.normal(size=40), rng.normal(size=40)])
    assert rank_features(x, y, ["noise_a", "signal", "noise_b"])[0]["feature"] == "signal"
    with pytest.raises(ValueError, match="both classes"):
        rank_features(x, np.zeros(40, dtype=int), ["noise_a", "signal", "noise_b"])


def test_reproduction_inputs_equal_her_values_order_and_ids(imported):
    path, truth = imported
    bundle, rows = load_bundle(path)
    data = build_inputs(bundle, rows, protocol="reproduce", feature_count=4, size=24, seed=42)
    names = her_assets.SELECTED[:4]
    columns = [CANDIDATE_FEATURES.index(n) for n in names]
    standardize = lambda x: (x - truth["mean"][columns]) / truth["scale"][columns]
    assert data["features"] == names and data["protocol"] == "reproduction_dhanya_v1"
    assert list(data["train_ids"]) == truth["subset"] and list(data["val_ids"]) == truth["val_ids"]
    assert list(data["test_ids"]) == truth["test_ids"]
    np.testing.assert_allclose(
        data["train_x"], recording_means(truth["rows"]["train"], names, truth["subset"], "clean", standardize), atol=1e-9)
    for snr in ("clean", "20", "10", "5", "0"):
        np.testing.assert_allclose(
            data["test_x"][snr], recording_means(truth["rows"]["test"], names, truth["test_ids"], snr, standardize), atol=1e-9)
    np.testing.assert_array_equal(data["test_y"], [truth["label_of"][i] for i in truth["test_ids"]])
    assert data["subset_source"] == "dhanya_saved_manifest"
    assert data["fit_scope"] == {"feature_ranking_recordings": 30, "scaler_fit_recordings": 30}
    assert data["comparable_to_published_quantum"] is True
    # Her rule averages both segments of the partly silent recording in the clean condition only.
    assert data["segments"]["test"] == 18
    assert build_inputs(bundle, rows, protocol="reproduce", feature_count=6, size=24, seed=42)["features"] == her_assets.SELECTED


def test_strict_protocol_fits_everything_inside_the_training_subset(imported):
    path, truth = imported
    bundle, rows = load_bundle(path)
    data = build_inputs(bundle, rows, protocol="strict", feature_count=4, size=24, seed=42)
    assert data["protocol"] == "strict_label_budget_v1"
    assert data["fit_scope"] == {"feature_ranking_recordings": 24, "scaler_fit_recordings": 24}
    assert list(data["train_ids"]) == truth["subset"]
    assert len(data["ranking"]) == 9 and data["features"] == [r["feature"] for r in data["ranking"][:4]]
    # The scaler statistics come from the subset's clean rows only, not her 30-recording scaler.
    subset_rows = np.array([[r[n] for n in data["features"]] for r in truth["rows"]["train"]
                            if r["recording_id"] in truth["subset"] and r["snr_db"] == "clean"], dtype=float)
    np.testing.assert_allclose(data["preprocessing"][0]["mean"], subset_rows.mean(axis=0))
    np.testing.assert_allclose(data["preprocessing"][0]["scale"], subset_rows.std(axis=0))
    np.testing.assert_allclose(data["train_x"].mean(axis=0), 0, atol=1e-9)
    assert not np.allclose(data["preprocessing"][0]["mean"],
                           [truth["mean"][CANDIDATE_FEATURES.index(n)] for n in data["features"]])
    # Never presented as comparable with the published quantum scores.
    assert data["comparable_to_published_quantum"] is False and "rerun the quantum models" in data["comparability_note"]
    assert list(data["test_ids"]) == truth["test_ids"]
    # The silent segment is left out of every condition, so each SNR scores the same 17 segments.
    assert data["segments"]["test"] == 17
    names, index = data["features"], truth["test_ids"].index("te1silent0")
    only = [r for r in truth["rows"]["test"] if r["recording_id"] == "te1silent0" and r["snr_db"] == "clean"
            and not r["_silent"]]
    scale = lambda x: (x - np.array(data["preprocessing"][0]["mean"])) / np.array(data["preprocessing"][0]["scale"])
    np.testing.assert_allclose(data["test_x"]["clean"][index], scale(np.array([[only[0][n] for n in names]]))[0])


def test_other_seeds_resample_and_are_not_her_protocol(imported):
    bundle, rows = load_bundle(imported[0])
    data = build_inputs(bundle, rows, protocol="reproduce", feature_count=4, size=24, seed=7)
    assert data["subset_source"] == "resampled_from_her_training_split"
    assert data["comparable_to_published_quantum"] is False
    assert set(data["train_ids"]) <= set(bundle["splits"]["train_all"]) and data["train_y"].sum() == 12
    assert list(data["train_ids"]) != list(build_inputs(bundle, rows, protocol="reproduce", feature_count=4,
                                                        size=24, seed=8)["train_ids"])
    with pytest.raises(ValueError, match="exceeds the largest balanced subset"):
        build_inputs(bundle, rows, protocol="strict", feature_count=4, size=40, seed=7)


def test_noisy_training_donors_outside_the_subset_are_dropped_or_refused(imported):
    bundle, rows = load_bundle(imported[0])
    outside = bundle["outside_subset_information"]["24"]["noise_donor_recordings_outside_subset"]
    assert outside > 0
    with pytest.raises(ValueError, match="donors outside the 24-recording subset"):
        build_inputs(bundle, rows, protocol="strict", feature_count=4, size=24, seed=42, train_snr="10",
                     donor_policy="error")
    kept = build_inputs(bundle, rows, protocol="strict", feature_count=4, size=24, seed=42, train_snr="10",
                        donor_policy="keep")
    assert kept["noisy_training_rows_dropped_for_outside_donors"] == 0 and kept["segments"]["train"] == 48
    try:
        dropped = build_inputs(bundle, rows, protocol="strict", feature_count=4, size=24, seed=42, train_snr="10")
    except ValueError as error:
        # Dropping may leave a recording with no rows; that is reported, never papered over.
        assert "no usable rows" in str(error)
    else:
        assert 0 < dropped["noisy_training_rows_dropped_for_outside_donors"] == 48 - dropped["segments"]["train"]


def test_failed_imports_cannot_be_used(tmp_path):
    root = tmp_path / "her"
    root.mkdir()
    her_assets.build(root)
    with pytest.raises(ImportValidationError):
        import_assets(root, tmp_path / "bundle")
    with pytest.raises(ValueError, match="failed import validation"):
        load_bundle(tmp_path / "bundle")


def test_dry_run_exports_quantum_inputs_and_fits_nothing(imported, tmp_path, monkeypatch):
    monkeypatch.setattr(matched, "fit_tabular", standins.forbidden)
    output = tmp_path / "plan"
    args = parser().parse_args(["--bundle", str(imported[0]), "--protocol", "strict", "--seeds", "42", "7",
                                "--feature-counts", "4", "6", "--dry-run", "--output", str(output)])
    assert run(args) == []
    plan = json.loads((output / "plan.json").read_text())
    assert len(plan) == 2 * 2 * 3 and not any(p["comparable_to_published_quantum"] for p in plan)
    arrays = np.load(output / "artifacts" / "inputs_seed7_n24_k6.npz")
    assert arrays["train_x"].shape == (24, 6) and arrays["val_x"].shape == (6, 6)
    assert {name for name in arrays.files if name.startswith("test_x_")} == {
        "test_x_clean", "test_x_20", "test_x_10", "test_x_5", "test_x_0"}
    summary = json.loads((output / "artifacts" / "inputs_seed7_n24_k6.json").read_text())
    assert summary["split_ids"]["train"] == list(arrays["train_ids"]) and summary["features"] == list(arrays["features"])
    config = json.loads((output / "config.json").read_text())
    assert config["protocol"] == "strict_label_budget_v1" and config["thresholds"] == {"A": 0.0, "B": 0.5, "C": 0.5}
    assert config["parameter_counts"]["B"] == {"4": 25, "6": 33}
    assert not (output / "results.csv").exists()


def test_run_scores_one_example_per_recording_and_exports_portable_models(imported, tmp_path, monkeypatch):
    monkeypatch.setattr(matched, "fit_tabular", standins.tabular)
    path, truth = imported
    output = tmp_path / "run"
    rows = run(parser().parse_args(["--bundle", str(path), "--protocol", "reproduce", "--models", "A", "B",
                                    "--output", str(output)]))
    assert len(rows) == 2 * 5
    assert {(r["level"], r["aggregation"], r["n_examples"], r["protocol"]) for r in rows} == {
        ("recording", "feature_mean_per_recording", 9, "reproduction_dhanya_v1")}
    assert all(r["comparable_to_published_quantum"] for r in rows)
    with (output / "predictions.csv").open() as handle:
        predictions = [p for p in csv.DictReader(handle) if p["model"] == "B" and p["snr_db"] == "5"]
    assert [p["recording_id"] for p in predictions] == truth["test_ids"]
    # From raw feature means of the selected features, the portable model reproduces every score.
    portable = load_portable(output / "artifacts" / "model_B_seed42_n24_k4.portable.json")
    raw = recording_means(truth["rows"]["test"], portable.spec["feature_order"], truth["test_ids"], "5")
    np.testing.assert_allclose(portable.scores(raw), [float(p["score"]) for p in predictions], rtol=1e-7)
    assert portable.spec["model_id"] == "B" and portable.spec["protocol"] == "reproduction_dhanya_v1"
    meta = json.loads((output / "artifacts" / "model_A_seed42_n24_k4.joblib.meta.json").read_text())
    assert meta["versions"]["scikit-learn"] and meta["subset_source"] == "dhanya_saved_manifest"
    with pytest.raises(ValueError, match="new or empty"):
        run(parser().parse_args(["--bundle", str(path), "--protocol", "reproduce", "--output", str(output)]))
