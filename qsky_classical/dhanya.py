"""Inventory, import, and validate Dhanya's experiment assets; nothing is invented.

Her tables are read with the csv module, so recording IDs stay exact strings.
Her pickled scalers/models are only checksummed and version-sniffed, never unpickled.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .artifacts import Standardizer, ids_sha256, pickled_sklearn_versions, sha256_file, versions
from .data import SR, normalize_split

CANDIDATE_FEATURES = [
    "mfcc_1_mean", "mfcc_2_mean", "mfcc_3_mean", "spectral_centroid_mean",
    "spectral_bandwidth_mean", "zero_crossing_rate_mean", "rms_energy_mean",
    "spectral_rolloff_mean", "spectral_flatness_mean",
]
SOURCE_SPLITS = ("train", "validation", "test")
SNRS = ("clean", "20", "10", "5", "0")
# Her manifests are named by the requested size; the balanced size actually used differs.
SUBSET_FILES = {24: 25, 50: 50, 70: 100}
EXPECTED_COUNTS = {"val": 80, "test": 98}
META_COLUMNS = ["sample_id", "recording_id", "binary_label", "split", "external_test", "snr_db",
                "noise_recording_id", "measured_snr_db"]

ASSETS = [
    ("selected_features", ["results/selected_features.json"],
     "selected feature names and their order for 4/5/6 features"),
    ("recording_split", ["results/recording_split.csv"],
     "exact recording_id to train/validation/test assignment"),
    ("train_subsets", ["results/small_data_subsets/train_subset_25.csv",
                       "results/small_data_subsets/train_subset_50.csv",
                       "results/small_data_subsets/train_subset_100.csv"],
     "exact 24/50/70-recording training subsets (files are named 25/50/100)"),
    ("raw_features", [f"data/features/{s}_features.csv" for s in SOURCE_SPLITS],
     "candidate feature values, sample IDs, and noise provenance for every segment"),
    ("scaled_features", [f"data/features/{s}_features_{k}.csv" for k in (4, 5, 6) for s in SOURCE_SPLITS],
     "standardized selected-feature values that were fed to the quantum kernel"),
    ("stage5_manifest", ["results/quantum/stage5_sample_manifest.json"],
     "recordings and sample IDs of the initial 24/80/98 comparison, with the scaler checksum"),
    ("published_results", ["results/quantum/ideal_simulator_results.csv"],
     "published scores and checksums of the train/validation/test ID lists"),
    ("scalers", [f"models/scaler_{k}.pkl" for k in (4, 5, 6)],
     "training-fitted scalers (checksummed only, never unpickled here)"),
    ("processed_audio", [f"data/processed/{s}/**/*.flac" for s in SOURCE_SPLITS],
     "segmented 16 kHz / 3 s audio that the feature rows were computed from"),
    ("master_metadata", ["data/master_metadata.csv"],
     "source dataset and original file of every recording"),
    ("feature_ranking", ["results/feature_selection.csv"],
     "mutual-information and ANOVA scores behind the saved feature ranking"),
    ("classical_results", ["results/svm_results.csv", "results/mlp_results.csv"],
     "her segment-level SVM/MLP results and validation-selected hyperparameters"),
    ("trainable_kernel_parameters", ["models/quantum_ideal/trainable_params_q4_n24_snr_clean_seed42.json"],
     "frozen trainable-kernel parameters of the published quantum result"),
    ("environment_lock", ["requirements.lock"],
     "exact package versions of her runs (requirements.txt only gives ranges)"),
]
EXPERIMENTS = {
    "abc_reproduce": ["selected_features", "recording_split", "train_subsets", "raw_features", "scaled_features"],
    "abc_strict": ["recording_split", "train_subsets", "raw_features"],
    "d_matching_audio": ["recording_split", "train_subsets", "raw_features", "processed_audio"],
    "verify_against_published": ["published_results", "stage5_manifest"],
    "quantum_rerun_on_strict_inputs": ["trainable_kernel_parameters", "environment_lock"],
}


class MissingAssetsError(FileNotFoundError):
    pass


class ImportValidationError(ValueError):
    pass


def inventory(root):
    """Report which of her assets exist under root. Reads nothing but file sizes."""
    root = Path(root)
    assets = []
    for key, patterns, purpose in ASSETS:
        found = []
        for pattern in patterns:
            files = sorted(p for p in root.glob(pattern) if p.is_file())
            found.append({"pattern": pattern, "files": len(files), "bytes": sum(p.stat().st_size for p in files)})
        present = sum(1 for f in found if f["files"])
        status = "present" if present == len(found) else "partial" if present else "missing"
        assets.append({"asset": key, "status": status, "purpose": purpose, "patterns": found})
    by_key = {a["asset"]: a for a in assets}
    experiments = {}
    for name, keys in EXPERIMENTS.items():
        missing = [f["pattern"] for k in keys for f in by_key[k]["patterns"] if not f["files"]]
        experiments[name] = {"ready": not missing, "missing": missing}
    pickles = {str(p.relative_to(root)): {"scikit_learn": pickled_sklearn_versions(p), "sha256": sha256_file(p)}
               for pattern in ("models/*.pkl", "models/**/*.joblib") for p in sorted(root.glob(pattern))}
    return {"root": str(root.resolve()), "assets": assets, "experiments": experiments, "pickles": pickles}


def print_inventory(report):
    print(f"Assets under {report['root']}:")
    for asset in report["assets"]:
        print(f"  [{asset['status']:7}] {asset['asset']}: {asset['purpose']}")
        for found in asset["patterns"]:
            if not found["files"]:
                print(f"             missing: {found['pattern']}")
    print("Experiments:")
    for name, state in report["experiments"].items():
        print(f"  {name}: {'ready' if state['ready'] else 'BLOCKED - ' + str(len(state['missing'])) + ' missing path(s)'}")
    for path, info in report["pickles"].items():
        print(f"  pickle {path}: scikit-learn {', '.join(info['scikit_learn']) or 'unknown'}")


def read_table(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def snr_label(value):
    text = str(value if value is not None else "").strip()
    if text.lower() in ("", "nan", "none", "clean"):
        return "clean"
    return f"{float(text):g}"


def truthy(value):
    return str(value).strip().lower() in ("true", "1", "1.0")


def blank(value):
    return str(value if value is not None else "").strip().lower() in ("", "nan", "none")


def candidate_features(audio):
    """Her nine candidate features, computed with the same librosa calls and defaults."""
    import librosa
    mfcc = librosa.feature.mfcc(y=audio, sr=SR, n_mfcc=3)
    return {
        "mfcc_1_mean": float(np.mean(mfcc[0])),
        "mfcc_2_mean": float(np.mean(mfcc[1])),
        "mfcc_3_mean": float(np.mean(mfcc[2])),
        "spectral_centroid_mean": float(np.mean(librosa.feature.spectral_centroid(y=audio, sr=SR))),
        "spectral_bandwidth_mean": float(np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=SR))),
        "zero_crossing_rate_mean": float(np.mean(librosa.feature.zero_crossing_rate(y=audio))),
        "rms_energy_mean": float(np.mean(librosa.feature.rms(y=audio))),
        "spectral_rolloff_mean": float(np.mean(librosa.feature.spectral_rolloff(y=audio, sr=SR))),
        "spectral_flatness_mean": float(np.mean(librosa.feature.spectral_flatness(y=audio))),
    }


def mix_like_dhanya(target, background, snr_db):
    """Her mixing rule, including attenuation of the whole mixture above a 0.99 peak."""
    target_rms = float(np.sqrt(np.mean(np.square(target, dtype=np.float64))))
    background_rms = float(np.sqrt(np.mean(np.square(background, dtype=np.float64))))
    if target_rms <= 1e-12 or background_rms <= 1e-12:
        raise ValueError("SNR is undefined for a silent target or background.")
    mixed = target.astype(np.float64) + background.astype(np.float64) * (
        target_rms / (10.0 ** (snr_db / 20.0) * background_rms))
    peak = float(np.max(np.abs(mixed)))
    if peak > 0.99:
        mixed *= 0.99 / peak
    return mixed.astype(np.float32)


def resolve_audio(root, audio_root, value):
    if blank(value):
        return None
    path = Path(str(value))
    candidates = []
    if path.is_absolute():
        candidates.append(path)
        if "data" in path.parts:
            # Her tables may hold absolute paths from her machine; re-anchor at data/.
            path = Path(*path.parts[path.parts.index("data"):])
    if not path.is_absolute():
        candidates += [base / path for base in (audio_root, root) if base is not None]
    return next((c.resolve() for c in candidates if c.is_file()), None)


def load_samples(root, feature_counts, selected, errors):
    """Read every feature row of the three internal splits into one validated list."""
    samples, seen = [], set()
    for source_split in SOURCE_SPLITS:
        split = normalize_split(source_split)
        columns, rows = read_table(root / f"data/features/{source_split}_features.csv")
        absent = [c for c in META_COLUMNS + CANDIDATE_FEATURES if c not in columns]
        if absent:
            errors.append(f"{source_split}_features.csv lacks columns: {absent}")
            continue
        scaled = {}
        for k in feature_counts:
            scaled_columns, scaled_rows = read_table(root / f"data/features/{source_split}_features_{k}.csv")
            absent = [c for c in ["sample_id", *selected[k]] if c not in scaled_columns]
            if absent:
                errors.append(f"{source_split}_features_{k}.csv lacks columns: {absent}")
            elif [r["sample_id"] for r in scaled_rows] != [r["sample_id"] for r in rows]:
                errors.append(f"{source_split}_features_{k}.csv does not list the same sample IDs in the "
                              "same order as the raw feature table")
            else:
                scaled[k] = scaled_rows
        for index, row in enumerate(rows):
            where = f"{source_split}_features.csv row {index + 2}"
            sample_id = row["sample_id"].strip()
            if not sample_id or sample_id in seen:
                errors.append(f"{where}: missing or duplicate sample_id {sample_id!r}")
                continue
            seen.add(sample_id)
            if normalize_split(row["split"]) != split:
                errors.append(f"{where}: split column says {row['split']!r}")
            if truthy(row["external_test"]):
                errors.append(f"{where}: external-test (NASA) row inside an internal split")
            try:
                label = int(float(row["binary_label"]))
                raw = [float(row[c]) for c in CANDIDATE_FEATURES]
                snr = snr_label(row["snr_db"])
            except ValueError:
                errors.append(f"{where}: non-numeric label, SNR, or feature value")
                continue
            if label not in (0, 1) or not np.isfinite(raw).all():
                errors.append(f"{where}: label must be 0/1 and features finite")
                continue
            sample = {
                "sample_id": sample_id, "recording_id": row["recording_id"].strip(), "split": split,
                "label": label, "snr_db": snr, "segment_index": row.get("segment_index", ""),
                "noise_recording_id": "" if blank(row["noise_recording_id"]) else row["noise_recording_id"].strip(),
                "measured_snr_db": "" if blank(row["measured_snr_db"]) else float(row["measured_snr_db"]),
                "file_path": row.get("file_path", ""), "base_file_path": row.get("base_file_path", ""),
                "noise_source_file_path": row.get("noise_source_file_path", ""),
                "raw": raw, "scaled": {},
            }
            for k, scaled_rows in scaled.items():
                try:
                    sample["scaled"][k] = [float(scaled_rows[index][c]) for c in selected[k]]
                except ValueError:
                    errors.append(f"{source_split}_features_{k}.csv row {index + 2}: non-numeric value")
            samples.append(sample)
    return samples


def check_noise(samples, split_of, label_of, errors, tolerance):
    """Every noisy row must trace to a clean segment and to a same-split class-0 donor."""
    clean = {s["sample_id"]: s for s in samples if s["snr_db"] == "clean"}
    donors, levels = {}, {}
    for s in samples:
        if s["snr_db"] == "clean":
            if s["noise_recording_id"]:
                errors.append(f"{s['sample_id']}: clean row names a noise donor")
            continue
        base_id, _, suffix = s["sample_id"].rpartition("__snr")
        base, donor = clean.get(base_id), s["noise_recording_id"]
        s["clean_sample_id"] = base_id
        problem = None
        if s["snr_db"] not in SNRS or suffix != s["snr_db"]:
            problem = f"unexpected SNR {s['snr_db']!r}"
        elif base is None or base["recording_id"] != s["recording_id"] or base["split"] != s["split"]:
            problem = "no clean segment of the same recording and split"
        elif not donor:
            problem = "no noise donor recorded"
        elif donor == s["recording_id"]:
            problem = "the noise donor is the target recording"
        elif split_of.get(donor) != s["split"]:
            problem = f"noise donor {donor} is in split {split_of.get(donor)!r}, not {s['split']!r}"
        elif label_of.get(donor) != 0:
            problem = f"noise donor {donor} is not a class-0 recording"
        elif s["measured_snr_db"] == "" or abs(s["measured_snr_db"] - float(s["snr_db"])) > tolerance:
            problem = f"measured SNR {s['measured_snr_db']} differs from the requested {s['snr_db']} dB"
        elif donors.setdefault(base_id, donor) != donor:
            problem = "different donors across SNR levels of one segment"
        if problem:
            errors.append(f"{s['sample_id']}: {problem}")
        levels.setdefault(base_id, set()).add(s["snr_db"])
    for s in clean.values():
        s["clean_sample_id"] = s["sample_id"]
        s["complete_noise"] = levels.get(s["sample_id"], set()) == set(SNRS[1:])
    for s in samples:
        s["complete_noise"] = clean[s["clean_sample_id"]]["complete_noise"] if s.get("clean_sample_id") in clean else False
    return donors


def import_assets(root, output, *, feature_counts=(4,), train_sizes=(24,), expected=None, audio_root=None,
                  measured_tolerance=0.01, materialize_noisy=False, verify_audio=0, feature_tolerance=1e-3):
    """Build a validated bundle from her assets, or raise naming exactly what is wrong."""
    root, output = Path(root), Path(output)
    audio_root = Path(audio_root) if audio_root else None
    expected = dict(EXPECTED_COUNTS if expected is None else expected)
    feature_counts, train_sizes = sorted(set(feature_counts)), sorted(set(train_sizes))
    unknown = [n for n in train_sizes if n not in SUBSET_FILES]
    if unknown:
        raise ValueError(f"No saved subset for {unknown}; her manifests cover {sorted(SUBSET_FILES)} recordings.")
    needed = ["results/recording_split.csv", *[f"data/features/{s}_features.csv" for s in SOURCE_SPLITS],
              *[f"results/small_data_subsets/train_subset_{SUBSET_FILES[n]}.csv" for n in train_sizes]]
    if feature_counts:
        needed += ["results/selected_features.json",
                   *[f"data/features/{s}_features_{k}.csv" for k in feature_counts for s in SOURCE_SPLITS]]
    missing = [p for p in needed if not (root / p).is_file()]
    if missing:
        raise MissingAssetsError("Cannot import: these files are missing under " + str(root) + ":\n  "
                                 + "\n  ".join(missing) + "\nRequest them from Dhanya; nothing is substituted.")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError(f"Output must be a new or empty directory: {output}")

    errors, checks = [], {}
    selection = json.loads((root / "results/selected_features.json").read_text()) if feature_counts else {}
    selected = {k: list(selection["qubit_mappings"][str(k)]["features"]) for k in feature_counts}
    for k, names in selected.items():
        if len(names) != k or set(names) - set(CANDIDATE_FEATURES):
            errors.append(f"selected_features.json: the {k}-feature mapping is not {k} known candidates")

    _, split_rows = read_table(root / "results/recording_split.csv")
    split_of = {}
    for row in split_rows:
        split = normalize_split(row.get("split"))
        if split not in ("train", "val", "test"):
            errors.append(f"recording_split.csv: unsupported split {row.get('split')!r}")
        elif split_of.setdefault(row["recording_id"].strip(), split) != split:
            errors.append(f"recording_split.csv: recording {row['recording_id']} is in two splits")

    samples = load_samples(root, feature_counts, selected, errors)
    label_of = {}
    for s in samples:
        if label_of.setdefault(s["recording_id"], s["label"]) != s["label"]:
            errors.append(f"recording {s['recording_id']} has conflicting labels")
        if split_of.get(s["recording_id"]) != s["split"]:
            errors.append(f"{s['sample_id']}: recording is listed under split "
                          f"{split_of.get(s['recording_id'])!r} in recording_split.csv")
    recordings = {split: {s["recording_id"] for s in samples if s["split"] == split} for split in ("train", "val", "test")}
    crossing = {f"{a}/{b}": sorted(recordings[a] & recordings[b]) for a, b in (("train", "val"), ("train", "test"), ("val", "test"))}
    checks["recordings_crossing_splits"] = {k: len(v) for k, v in crossing.items()}
    errors += [f"recordings in both {pair}: {ids[:5]}" for pair, ids in crossing.items() if ids]

    donors = check_noise(samples, split_of, label_of, errors, measured_tolerance)
    checks["noisy_rows"] = sum(s["snr_db"] != "clean" for s in samples)
    checks["clean_segments_without_all_noise_levels"] = {
        split: sum(s["snr_db"] == "clean" and s["split"] == split and not s["complete_noise"] for s in samples)
        for split in ("train", "val", "test")}

    # Her scaler was fit on every training row (all recordings, all SNR conditions).
    # Rebuilding it from the tables proves raw values, scaled values, and preprocessing agree.
    train_rows = [s for s in samples if s["split"] == "train"]
    scalers = {}
    for k, names in selected.items():
        if not train_rows or any(k not in s["scaled"] for s in samples):
            continue
        columns = [CANDIDATE_FEATURES.index(n) for n in names]
        scaler = Standardizer.from_rows(np.array([s["raw"] for s in train_rows])[:, columns], names)
        rebuilt = scaler.transform(np.array([s["raw"] for s in samples])[:, columns])
        worst = float(np.max(np.abs(rebuilt - np.array([s["scaled"][k] for s in samples]))))
        scalers[k] = scaler.to_dict()
        checks[f"scaled_{k}_rebuilt_from_all_training_rows_max_abs_error"] = worst
        if worst > 1e-6:
            errors.append(f"{k}-feature scaled table is not StandardScaler(all training rows) of the raw "
                          f"table (max error {worst:.3g}); values and preprocessing do not agree")

    subsets = {}
    for size in train_sizes:
        _, rows = read_table(root / f"results/small_data_subsets/train_subset_{SUBSET_FILES[size]}.csv")
        ids = sorted({r["recording_id"].strip() for r in rows})
        counts = np.bincount([label_of.get(i, 0) for i in ids], minlength=2).tolist()
        clean_ids = {s["recording_id"] for s in train_rows if s["snr_db"] == "clean"}
        if len(ids) != size:
            errors.append(f"subset file for {size} holds {len(ids)} recordings")
        if counts[0] != counts[1]:
            errors.append(f"subset {size} is not class-balanced: {counts}")
        if set(ids) - recordings["train"] or set(ids) - clean_ids:
            errors.append(f"subset {size} names recordings outside the clean training split")
        subsets[size] = ids

    val_ids = sorted({s["recording_id"] for s in samples if s["split"] == "val" and s["snr_db"] == "clean"})
    test_by_snr = {snr: {s["recording_id"] for s in samples if s["split"] == "test" and s["snr_db"] == snr} for snr in SNRS}
    test_ids = sorted(set.intersection(*test_by_snr.values()))
    test_excluded = sorted(test_by_snr["clean"] - set(test_ids))
    counts = {"train_all": len(recordings["train"]), "val": len(val_ids), "test": len(test_ids),
              "test_excluded_not_at_every_snr": len(test_excluded)}
    checks["counts"] = counts
    for split, ids in (("val", val_ids), ("test", test_ids)):
        if split in expected and len(ids) != expected[split]:
            errors.append(f"{split} has {len(ids)} recordings; expected {expected[split]}")
        if {label_of[i] for i in ids} != {0, 1}:
            errors.append(f"{split} does not contain both classes")

    # Information from outside each training subset that her saved protocol still used.
    audit = {}
    for size, ids in subsets.items():
        chosen = set(ids)
        outside = sorted({donors[s["clean_sample_id"]] for s in train_rows
                          if s["recording_id"] in chosen and s["snr_db"] != "clean"
                          and s.get("clean_sample_id") in donors} - chosen)
        audit[size] = {
            "training_recordings": size,
            "feature_ranking_recordings": selection.get("recordings_used"),
            "scaler_fit_recordings": len(recordings["train"]),
            "scaler_fit_rows": len(train_rows),
            "noise_donor_recordings_outside_subset": len(outside),
            "noise_donors_outside_subset": outside,
        }

    hashes = {"train": {n: ids_sha256(ids) for n, ids in subsets.items()},
              "val": ids_sha256(val_ids), "test": ids_sha256(test_ids)}
    published = root / "results/quantum/ideal_simulator_results.csv"
    verified = {}
    if published.is_file():
        _, rows = read_table(published)
        for size in train_sizes:
            for k in feature_counts or [4]:
                match = [r for r in rows if r["feature_count"] == str(k) and r["training_size_requested"] == str(size)]
                if not match:
                    continue
                agree = ({r["training_sample_manifest_hash"] for r in match} == {hashes["train"][size]}
                         and {r["validation_sample_manifest_hash"] for r in match} == {hashes["val"]}
                         and {r["test_sample_manifest_hash"] for r in match} == {hashes["test"]})
                verified[f"k{k}_n{size}"] = agree
                if not agree:
                    errors.append(f"ID lists for {k} features / {size} recordings do not match the "
                                  "checksums published in ideal_simulator_results.csv")
    checks["published_id_checksums"] = verified or "not checked: ideal_simulator_results.csv is absent"
    stage5 = root / "results/quantum/stage5_sample_manifest.json"
    if stage5.is_file() and 24 in subsets:
        manifest = json.loads(stage5.read_text())
        sample_ids = {}
        for s in samples:
            if s["snr_db"] == "clean":
                sample_ids.setdefault(s["recording_id"], set()).add(s["sample_id"])
        expected_ids = {"training": subsets[24], "validation": val_ids, "test": test_ids}
        agree = True
        for part, ids in expected_ids.items():
            listed = {r["recording_id"]: set(r["sample_ids"]) for r in manifest.get(part, [])}
            agree &= sorted(listed) == ids and all(listed[i] == sample_ids.get(i) for i in listed)
        checks["stage5_manifest_ids_and_sample_ids_match"] = bool(agree)
        if not agree:
            errors.append("stage5_sample_manifest.json lists different recordings or sample IDs")
        scaler_path = root / "models/scaler_4.pkl"
        if scaler_path.is_file() and manifest.get("scaler_sha256"):
            checks["scaler_4_checksum_matches_stage5"] = sha256_file(scaler_path) == manifest["scaler_sha256"]

    # Model D needs the audio itself; record what exists without guessing at the rest.
    used = {"train": set().union(*subsets.values()) if subsets else set(), "val": set(val_ids), "test": set(test_ids)}
    audio = {"clean_segments": 0, "clean_segments_with_audio": 0}
    for s in samples:
        s["audio_path"] = ""
        if s["recording_id"] in used[s["split"]] and s["snr_db"] == "clean":
            path = resolve_audio(root, audio_root, s["file_path"])
            audio["clean_segments"] += 1
            audio["clean_segments_with_audio"] += path is not None
            s["audio_path"] = str(path or "")
    audio["model_d_ready"] = bool(audio["clean_segments"]) and audio["clean_segments"] == audio["clean_segments_with_audio"]
    checks["audio"] = audio

    output.mkdir(parents=True, exist_ok=True)
    if audio["model_d_ready"] and verify_audio:
        checks["audio_feature_check"] = verify_clean_audio(samples, verify_audio, feature_tolerance, errors)
    premixed = {}
    if materialize_noisy and audio["model_d_ready"]:
        premixed = write_premixed(root, audio_root, output, samples, used["test"], feature_tolerance, errors)
        checks["premixed_test_conditions"] = {snr: len(rows) for snr, rows in premixed.items()}
    elif materialize_noisy:
        errors.append("--materialize-noisy needs the processed clean audio, which is missing")

    status = "failed" if errors else "ok"
    bundle = {
        "status": status, "source_root": str(root.resolve()), "errors": errors, "checks": checks,
        "candidate_features": CANDIDATE_FEATURES, "selected_features": {str(k): v for k, v in selected.items()},
        "scalers_rebuilt_from_tables": {str(k): v for k, v in scalers.items()},
        "feature_ranking_recordings": selection.get("recordings_used"),
        "splits": {"train_all": sorted(recordings["train"]), "train_subsets": {str(n): v for n, v in subsets.items()},
                   "val": val_ids, "test": test_ids, "test_excluded_not_at_every_snr": test_excluded},
        "labels": {i: label_of[i] for i in sorted(label_of)},
        "id_sha256": {"train": {str(n): h for n, h in hashes["train"].items()}, "val": hashes["val"], "test": hashes["test"]},
        "verified_against_published": bool(verified) and all(verified.values()),
        "outside_subset_information": {str(n): v for n, v in audit.items()},
        "audio": {"sample_rate": SR, "segment_seconds": 3.0, "segmentation": "non-overlapping, zero-padded tail",
                  "normalization": "per-segment peak", "noise": "same-split class-0 donor, one per segment across SNRs",
                  "mixing": "RMS SNR, whole mixture attenuated to 0.99 peak when it would clip"},
        "aggregation": "mean of standardized segment features per recording, then one prediction per recording",
        "test_rule": "recordings present at every SNR condition",
        "silent_policy": "silent targets keep their clean row and have no noisy rows",
        "versions": versions(),
    }
    (output / "bundle.json").write_text(json.dumps(bundle, indent=2) + "\n")
    write_rows(output / "rows.csv", samples, selected)
    if audio["model_d_ready"] and not errors:
        write_audio_manifests(output, samples, subsets, used)
    if errors:
        raise ImportValidationError(f"{len(errors)} validation problem(s); see {output / 'bundle.json'}. First: {errors[0]}")
    return bundle


def write_rows(path, samples, selected):
    scaled_columns = [f"scaled{k}:{name}" for k, names in selected.items() for name in names]
    header = ["sample_id", "clean_sample_id", "recording_id", "split", "label", "snr_db", "segment_index",
              "noise_recording_id", "measured_snr_db", "complete_noise", "audio_path",
              *CANDIDATE_FEATURES, *scaled_columns]
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for s in samples:
            scaled = [repr(v) for k in selected for v in s["scaled"].get(k, [""] * k)]
            writer.writerow([s["sample_id"], s.get("clean_sample_id", ""), s["recording_id"], s["split"], s["label"],
                             s["snr_db"], s["segment_index"], s["noise_recording_id"], s["measured_snr_db"],
                             int(s["complete_noise"]), s["audio_path"], *[repr(v) for v in s["raw"]], *scaled])


def write_audio_manifests(output, samples, subsets, used):
    """One manifest per training size for the audio CLI; validation and test never change."""
    held_out = [s for s in samples if s["snr_db"] == "clean" and s["split"] != "train"
                and s["recording_id"] in used[s["split"]]]
    for size, ids in subsets.items():
        train = [s for s in samples if s["snr_db"] == "clean" and s["split"] == "train" and s["recording_id"] in set(ids)]
        with (output / f"manifest_n{size}.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["path", "label", "group_id", "split"])
            for s in train + held_out:
                writer.writerow([s["audio_path"], s["label"], s["recording_id"], s["split"]])


def load_audio(path):
    import librosa
    return librosa.load(path, sr=SR, mono=True)[0]


def feature_error(audio, sample):
    values = candidate_features(audio)
    return max(abs(values[name] - want) / max(abs(want), 1e-9) for name, want in zip(CANDIDATE_FEATURES, sample["raw"]))


def verify_clean_audio(samples, limit, tolerance, errors):
    """Recompute her features from the audio for a few segments per split."""
    worst, checked = 0.0, 0
    for split in ("train", "val", "test"):
        for s in [s for s in samples if s["split"] == split and s["audio_path"]][:limit]:
            error = feature_error(load_audio(s["audio_path"]), s)
            worst, checked = max(worst, error), checked + 1
            if error > tolerance:
                errors.append(f"{s['sample_id']}: features recomputed from the audio differ by {error:.3g} (relative)")
    return {"segments_checked": checked, "max_relative_error": worst, "tolerance": tolerance}


def write_premixed(root, audio_root, output, samples, test_ids, tolerance, errors):
    """Regenerate her noisy test audio from recorded donors; each mixture must reproduce its feature row."""
    import librosa
    import soundfile as sf
    clean = {s["sample_id"]: s for s in samples if s["snr_db"] == "clean"}
    written = {}
    for s in samples:
        if s["split"] != "test" or s["snr_db"] == "clean" or s["recording_id"] not in test_ids:
            continue
        target = clean.get(s.get("clean_sample_id"))
        donor = resolve_audio(root, audio_root, s["noise_source_file_path"])
        if target is None or not target["audio_path"] or donor is None:
            errors.append(f"{s['sample_id']}: target or donor audio is missing; cannot regenerate the mixture")
            continue
        signal = load_audio(target["audio_path"])
        mixed = mix_like_dhanya(signal, librosa.util.fix_length(load_audio(donor), size=len(signal)), float(s["snr_db"]))
        error = feature_error(mixed, s)
        if error > tolerance:
            errors.append(f"{s['sample_id']}: the regenerated mixture differs from its feature row by {error:.3g} (relative)")
            continue
        path = output / "premixed" / f"snr_{s['snr_db']}" / f"{s['sample_id']}.wav"
        path.parent.mkdir(parents=True, exist_ok=True)
        # Float WAV keeps the in-memory mixture exactly; her features never saw 16-bit rounding.
        sf.write(path, mixed, SR, subtype="FLOAT")
        written.setdefault(s["snr_db"], []).append((str(path.resolve()), target["audio_path"]))
    for snr, rows in written.items():
        with (output / f"premixed_test_{snr}.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["path", "source_path"])
            writer.writerows(rows)
    return written


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest="command", required=True)
    inv = commands.add_parser("inventory", help="List which of her assets are present or missing")
    inv.add_argument("--root", required=True, type=Path, help="Copy of her project directory")
    inv.add_argument("--output", type=Path, help="Also write the report as JSON")
    imp = commands.add_parser("import", help="Validate her assets and build an experiment bundle")
    imp.add_argument("--root", required=True, type=Path)
    imp.add_argument("--output", required=True, type=Path, help="New/empty bundle directory")
    imp.add_argument("--feature-counts", nargs="*", type=int, choices=[4, 5, 6], default=[4],
                     help="Selected-feature sets to import; give none for a strict-only bundle")
    imp.add_argument("--train-sizes", nargs="+", type=int, choices=sorted(SUBSET_FILES), default=[24])
    imp.add_argument("--expect-val", type=int, default=EXPECTED_COUNTS["val"])
    imp.add_argument("--expect-test", type=int, default=EXPECTED_COUNTS["test"])
    imp.add_argument("--audio-root", type=Path, help="Project root that holds data/processed, if elsewhere")
    imp.add_argument("--materialize-noisy", action="store_true", help="Regenerate her noisy test audio for model D")
    imp.add_argument("--verify-audio", type=int, default=0, metavar="N",
                     help="Recompute her features from N clean segments per split and compare")
    return p


def main():
    p = parser()
    args = p.parse_args()
    if args.command == "inventory":
        report = inventory(args.root)
        print_inventory(report)
        if args.output:
            args.output.write_text(json.dumps(report, indent=2) + "\n")
        return
    try:
        bundle = import_assets(args.root, args.output, feature_counts=args.feature_counts,
                               train_sizes=args.train_sizes, audio_root=args.audio_root,
                               expected={"val": args.expect_val, "test": args.expect_test},
                               materialize_noisy=args.materialize_noisy, verify_audio=args.verify_audio)
    except (MissingAssetsError, ImportValidationError, ValueError) as error:
        p.error(str(error))
    print(f"Bundle written to {args.output}: {bundle['checks']['counts']}")
    print(f"Verified against published ID checksums: {bundle['verified_against_published']}")
    print(f"Model D audio ready: {bundle['checks']['audio']['model_d_ready']}")


if __name__ == "__main__":
    main()
