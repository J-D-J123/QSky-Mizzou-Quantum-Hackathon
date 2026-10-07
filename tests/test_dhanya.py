"""Importer checks on a synthetic copy of Dhanya's asset layout; her real assets are not needed."""

import csv
import json

import numpy as np
import pytest

from qsky_classical import cli
from qsky_classical.data import read_manifest, split_clips
from qsky_classical.dhanya import (
    ASSETS, CANDIDATE_FEATURES, ImportValidationError, MissingAssetsError, candidate_features, import_assets,
    inventory, mix_like_dhanya, snr_label,
)

import her_assets
import standins

COUNTS = {"val": 6, "test": 9}


@pytest.fixture
def assets(tmp_path):
    root = tmp_path / "her"
    root.mkdir()
    return root, her_assets.build(root)


def test_inventory_names_every_missing_asset_and_never_raises(tmp_path, assets):
    empty = inventory(tmp_path / "nothing-here")
    assert {a["status"] for a in empty["assets"]} == {"missing"} and len(empty["assets"]) == len(ASSETS)
    assert not any(e["ready"] for e in empty["experiments"].values())
    report = inventory(assets[0])
    status = {a["asset"]: a["status"] for a in report["assets"]}
    assert status["raw_features"] == status["scaled_features"] == status["recording_split"] == "present"
    assert status["train_subsets"] == "partial" and status["processed_audio"] == "missing"
    assert report["experiments"]["abc_reproduce"]["missing"] == [
        "results/small_data_subsets/train_subset_50.csv", "results/small_data_subsets/train_subset_100.csv"]
    assert not report["experiments"]["d_matching_audio"]["ready"]
    assert "data/processed/test/**/*.flac" in report["experiments"]["d_matching_audio"]["missing"]


def test_import_preserves_exact_ids_and_verifies_them(tmp_path, assets):
    root, truth = assets
    bundle = import_assets(root, tmp_path / "bundle", expected=COUNTS)
    assert bundle["status"] == "ok" and bundle["errors"] == []
    splits = bundle["splits"]
    assert splits["train_subsets"]["24"] == truth["subset"]
    assert splits["val"] == truth["val_ids"] and splits["test"] == truth["test_ids"]
    assert splits["train_all"] == truth["train_ids"]
    # Her rule is per recording: one with a silent segment stays, because its other segment is at every SNR.
    assert "te1silent0" in splits["test"] and splits["test_excluded_not_at_every_snr"] == []
    assert bundle["checks"]["counts"] == {"train_all": 30, "val": 6, "test": 9, "test_excluded_not_at_every_snr": 0}
    assert bundle["checks"]["clean_segments_without_all_noise_levels"] == {"train": 0, "val": 0, "test": 1}
    assert bundle["checks"]["recordings_crossing_splits"] == {"train/val": 0, "train/test": 0, "val/test": 0}
    assert bundle["checks"]["published_id_checksums"] == {"k4_n24": True}
    assert bundle["verified_against_published"] is True
    assert bundle["id_sha256"]["test"] == her_assets.her_hash(truth["test_ids"])
    assert bundle["selected_features"]["4"] == her_assets.SELECTED[:4]
    assert bundle["checks"]["scaled_4_rebuilt_from_all_training_rows_max_abs_error"] < 1e-9
    scaler = bundle["scalers_rebuilt_from_tables"]["4"]
    np.testing.assert_allclose(scaler["mean"], [truth["mean"][CANDIDATE_FEATURES.index(n)] for n in scaler["features"]])
    with (tmp_path / "bundle" / "rows.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert {r["split"] for r in rows} == {"train", "val", "test"}
    assert len(rows) == sum(len(v) for v in truth["rows"].values())
    silent = [r for r in rows if r["recording_id"] == "te1silent0"]
    assert sorted(r["complete_noise"] for r in silent if r["snr_db"] == "clean") == ["0", "1"]


def test_outside_subset_information_is_audited(tmp_path, assets):
    root, truth = assets
    audit = import_assets(root, tmp_path / "bundle", expected=COUNTS)["outside_subset_information"]["24"]
    assert audit["training_recordings"] == 24
    # Ranking and scaler both saw all 30 training recordings, not the 24 in the subset.
    assert audit["feature_ranking_recordings"] == 30 and audit["scaler_fit_recordings"] == 30
    outside = set(audit["noise_donors_outside_subset"])
    assert outside and outside <= set(truth["train_ids"]) - set(truth["subset"])
    assert all(truth["label_of"][donor] == 0 for donor in outside)
    assert audit["noise_donor_recordings_outside_subset"] == len(outside)


def test_counts_must_match_the_stated_comparison(tmp_path, assets):
    # Default expectation is her 80 validation / 98 test recordings.
    with pytest.raises(ImportValidationError, match="val has 6 recordings; expected 80"):
        import_assets(assets[0], tmp_path / "bundle")
    report = json.loads((tmp_path / "bundle" / "bundle.json").read_text())
    assert report["status"] == "failed" and "test has 9 recordings; expected 98" in report["errors"]


def test_missing_files_are_listed_and_nothing_is_substituted(tmp_path, assets):
    root, _ = assets
    (root / "data/features/test_features_4.csv").unlink()
    (root / "results/small_data_subsets/train_subset_25.csv").unlink()
    with pytest.raises(MissingAssetsError) as error:
        import_assets(root, tmp_path / "bundle", expected=COUNTS)
    assert "data/features/test_features_4.csv" in str(error.value)
    assert "results/small_data_subsets/train_subset_25.csv" in str(error.value)
    assert not (tmp_path / "bundle").exists()
    with pytest.raises(ValueError, match="her manifests cover"):
        import_assets(root, tmp_path / "bundle", train_sizes=[32], expected=COUNTS)


def break_donor_split(root, truth):
    donor = next(r for r in truth["val_ids"] if truth["label_of"][r] == 0)
    def change(rows):
        next(r for r in rows if r["snr_db"] == "10")["noise_recording_id"] = donor
    her_assets.rewrite(root / "data/features/train_features.csv", change)


def break_donor_class(root, truth):
    drone = next(r for r in truth["train_ids"] if truth["label_of"][r] == 1)
    def change(rows):
        row = next(r for r in rows if r["snr_db"] == "20" and r["recording_id"] != drone)
        row["noise_recording_id"] = drone
    her_assets.rewrite(root / "data/features/train_features.csv", change)


def break_group_separation(root, truth):
    def change(rows):
        rows.append({**rows[0], "sample_id": "leaked__clean__seg0000", "recording_id": truth["train_ids"][0],
                     "binary_label": truth["label_of"][truth["train_ids"][0]]})
    for name in ("test_features.csv", "test_features_4.csv"):
        her_assets.rewrite(root / "data/features" / name, change)


def break_sample_order(root, truth):
    her_assets.rewrite(root / "data/features/validation_features_4.csv", lambda rows: rows.reverse())


def break_scaled_values(root, truth):
    def change(rows):
        rows[3]["mfcc_2_mean"] = repr(float(rows[3]["mfcc_2_mean"]) + 0.25)
    her_assets.rewrite(root / "data/features/test_features_4.csv", change)


def break_published_ids(root, truth):
    her_assets.rewrite(root / "results/quantum/ideal_simulator_results.csv",
                       lambda rows: rows[0].update(test_sample_manifest_hash="0" * 64))


def break_measured_snr(root, truth):
    def change(rows):
        next(r for r in rows if r["snr_db"] == "5")["measured_snr_db"] = "7.5"
    her_assets.rewrite(root / "data/features/test_features.csv", change)


def break_labels(root, truth):
    def change(rows):
        rows[0]["binary_label"] = 1 - int(rows[0]["binary_label"])
    her_assets.rewrite(root / "data/features/validation_features.csv", change)


@pytest.mark.parametrize("damage, message", [
    (break_donor_split, "is in split 'val', not 'train'"),
    (break_donor_class, "is not a class-0 recording"),
    (break_group_separation, "recordings in both train/test"),
    (break_sample_order, "same sample IDs in the same order"),
    (break_scaled_values, "values and preprocessing do not agree"),
    (break_published_ids, "do not match the checksums published"),
    (break_measured_snr, "differs from the requested 5 dB"),
    (break_labels, "conflicting labels"),
])
def test_each_validation_catches_its_problem(tmp_path, assets, damage, message):
    root, truth = assets
    damage(root, truth)
    with pytest.raises(ImportValidationError):
        import_assets(root, tmp_path / "bundle", expected=COUNTS)
    errors = json.loads((tmp_path / "bundle" / "bundle.json").read_text())["errors"]
    assert any(message in e for e in errors), errors[:3]


def test_strict_only_import_needs_no_scaled_tables(tmp_path, assets):
    root, _ = assets
    for path in root.glob("data/features/*_features_[456].csv"):
        path.unlink()
    (root / "results/selected_features.json").unlink()
    bundle = import_assets(root, tmp_path / "bundle", feature_counts=[], expected=COUNTS)
    assert bundle["status"] == "ok" and bundle["selected_features"] == {}


def test_snr_labels_and_her_mixing_rule():
    assert [snr_label(v) for v in ("clean", "", "nan", "20", "5.0", 0)] == ["clean", "clean", "clean", "20", "5", "0"]
    rng = np.random.default_rng(5)
    target = 0.01 * rng.normal(size=8000).astype(np.float32)
    donor = rng.normal(size=8000).astype(np.float32)
    mixed = mix_like_dhanya(target, donor, 10.0)
    residual = mixed.astype(np.float64) - target
    assert 20 * np.log10(np.sqrt(np.mean(target.astype(np.float64) ** 2)) / np.sqrt(np.mean(residual ** 2))) == pytest.approx(10, abs=1e-3)
    loud = mix_like_dhanya(target / np.max(np.abs(target)), donor, 0.0)
    # A mixture that would clip is attenuated as a whole, unlike the audio CLI's mixing.
    assert np.max(np.abs(loud)) == pytest.approx(0.99, abs=1e-6)
    with pytest.raises(ValueError, match="undefined"):
        mix_like_dhanya(np.zeros(8000, dtype=np.float32), donor, 10.0)


def test_candidate_features_keep_her_names_and_order():
    values = candidate_features(np.sin(np.arange(8000) / 8).astype(np.float32))
    assert list(values) == CANDIDATE_FEATURES and all(np.isfinite(v) for v in values.values())


def test_matching_audio_enables_model_d_manifests_and_premixed_conditions(tmp_path, monkeypatch):
    root = tmp_path / "her"
    root.mkdir()
    truth = her_assets.build(root, real_audio=True, train_per_class=12, val_per_class=2, test_per_class=2, segments=1)
    counts = {"val": 4, "test": 4}
    assert inventory(root)["experiments"]["d_matching_audio"]["missing"] == [
        "results/small_data_subsets/train_subset_50.csv", "results/small_data_subsets/train_subset_100.csv"]
    bundle = import_assets(root, tmp_path / "bundle", expected=counts, materialize_noisy=True, verify_audio=2)
    assert bundle["checks"]["audio"]["model_d_ready"] is True
    # Its only segment is silent, so this recording exists in the clean condition alone.
    assert bundle["splits"]["test_excluded_not_at_every_snr"] == ["te1silent0"]
    # Features recomputed from the audio, and from regenerated mixtures, reproduce her rows.
    assert bundle["checks"]["audio_feature_check"]["segments_checked"] == 6
    assert bundle["checks"]["audio_feature_check"]["max_relative_error"] < 1e-3
    assert bundle["checks"]["premixed_test_conditions"] == {"20": 4, "10": 4, "5": 4, "0": 4}
    manifest = tmp_path / "bundle" / "manifest_n24.csv"
    clips = split_clips(read_manifest(manifest))
    assert {c.split for c in clips} == {"train", "val", "test"}
    assert sorted({c.group_id for c in clips if c.split == "train"}) == truth["subset"]
    assert sorted({c.group_id for c in clips if c.split == "test"}) == truth["test_ids"]
    # The audio CLI accepts the bundle as-is; a dry run fits nothing.
    monkeypatch.setattr(cli, "fit_tabular", standins.forbidden)
    args = cli.parser().parse_args([
        "--manifest", str(manifest), "--models", "D", "--window-seconds", "0.5", "--snr-preset", "shared",
        "--levels", "window", "recording", "--dry-run", "--output", str(tmp_path / "plan"),
        "--premixed-test", *[f"{snr}={tmp_path / 'bundle' / f'premixed_test_{snr}.csv'}" for snr in (20, 10, 5, 0)]])
    assert cli.run(args) == []
    with (tmp_path / "plan" / "exclusions.csv").open() as handle:
        assert list(csv.DictReader(handle)) == []


def test_audio_is_required_before_noisy_audio_can_be_regenerated(tmp_path, assets):
    with pytest.raises(ImportValidationError):
        import_assets(assets[0], tmp_path / "bundle", expected=COUNTS, materialize_noisy=True)
    report = json.loads((tmp_path / "bundle" / "bundle.json").read_text())
    assert report["checks"]["audio"]["model_d_ready"] is False
    assert any("needs the processed clean audio" in e for e in report["errors"])
    assert not list((tmp_path / "bundle").glob("manifest_n*.csv"))
