"""Build a small synthetic copy of Dhanya's asset layout for importer tests.

Nothing here is her data: IDs, features, and audio are generated, in her file formats.
"""

import csv
import hashlib
import json

import numpy as np

from qsky_classical.dhanya import CANDIDATE_FEATURES, candidate_features, mix_like_dhanya

SELECTED = ["mfcc_2_mean", "spectral_flatness_mean", "spectral_bandwidth_mean", "mfcc_1_mean",
            "rms_energy_mean", "spectral_centroid_mean"]
META = ["sample_id", "file_path", "source_dataset", "original_class", "binary_label", "scenario",
        "recording_id", "external_test", "split", "segment_index", "snr_db", "noise_recording_id",
        "measured_snr_db", "base_file_path", "condition_audio_path", "noise_source_file_path", "audio_persisted"]
SNRS = (20, 10, 5, 0)
SR = 16000


def her_hash(ids):
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def write_csv(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


def build(root, *, real_audio=False, train_per_class=15, val_per_class=3, test_per_class=4, segments=2, seed=0):
    rng = np.random.default_rng(seed)
    recordings = [(f"{split[:2]}{label}{index:02d}c0ffee", split, label)
                  for split, count in (("train", train_per_class), ("validation", val_per_class), ("test", test_per_class))
                  for label in (0, 1) for index in range(count)]
    # A test drone recording whose last segment is silent: that segment has a clean row only,
    # as in her pipeline. With one segment the recording is therefore absent from noisy conditions.
    recordings.append(("te1silent0", "test", 1))
    label_of = {rid: label for rid, _, label in recordings}
    rows = {"train": [], "validation": [], "test": []}
    audio = {}
    if real_audio:
        import librosa
        import soundfile as sf
    for rid, split, label in recordings:
        kind = "drone" if label else "background"
        for index in range(segments):
            piece = hashlib.sha256(rid.encode()).hexdigest()[:8]
            sample_id = f"{rid}__{piece}__clean__seg{index:04d}"
            path = f"data/processed/{split}/svanstrom/{kind}/{sample_id}.flac"
            silent = rid == "te1silent0" and index == segments - 1
            if real_audio:
                t = np.arange(SR // 2) / SR
                wave = np.sin(2 * np.pi * (180 if label else 1300) * t + rng.uniform(0, 6)) + 0.2 * rng.normal(size=len(t))
                wave = np.zeros_like(wave) if silent else wave / np.max(np.abs(wave))
                (root / path).parent.mkdir(parents=True, exist_ok=True)
                sf.write(root / path, wave, SR, format="FLAC", subtype="PCM_16")
                audio[sample_id] = librosa.load(root / path, sr=SR, mono=True)[0]
                values = candidate_features(audio[sample_id])
            else:
                values = dict(zip(CANDIDATE_FEATURES, rng.normal(size=9) + label * np.linspace(2, 0, 9)))
            rows[split].append({
                "sample_id": sample_id, "file_path": path, "source_dataset": "svanstrom", "original_class": kind,
                "binary_label": label, "scenario": "", "recording_id": rid, "external_test": False, "split": split,
                "segment_index": index, "snr_db": "clean", "noise_recording_id": "", "measured_snr_db": "",
                "base_file_path": path, "condition_audio_path": path, "noise_source_file_path": "",
                "audio_persisted": True, "_silent": silent, **values})
    for split, clean_rows in rows.items():
        backgrounds = [r for r in clean_rows if r["binary_label"] == 0 and r["segment_index"] == 0]
        noisy = []
        for row in clean_rows:
            if row["_silent"]:
                continue
            choices = [b for b in backgrounds if b["recording_id"] != row["recording_id"]]
            donor = choices[int(rng.integers(len(choices)))]
            for snr in SNRS:
                if real_audio:
                    mixed = mix_like_dhanya(audio[row["sample_id"]], audio[donor["sample_id"]], float(snr))
                    values = candidate_features(mixed)
                else:
                    values = {name: row[name] + rng.normal() / (1 + snr) for name in CANDIDATE_FEATURES}
                noisy.append({**row, **values, "sample_id": f"{row['sample_id']}__snr{snr}", "file_path": "",
                              "snr_db": snr, "noise_recording_id": donor["recording_id"],
                              "measured_snr_db": snr + 1e-6, "condition_audio_path": "",
                              "noise_source_file_path": donor["file_path"], "audio_persisted": False})
        clean_rows.extend(noisy)
    train = np.array([[r[name] for name in CANDIDATE_FEATURES] for r in rows["train"]], dtype=float)
    mean, scale = train.mean(axis=0), train.std(axis=0)
    header = [*META, *CANDIDATE_FEATURES]
    for split, split_rows in rows.items():
        plain = [{k: v for k, v in r.items() if k != "_silent"} for r in split_rows]
        write_csv(root / f"data/features/{split}_features.csv", header, plain)
        for k in (4, 5, 6):
            names = SELECTED[:k]
            scaled = [{**{c: r[c] for c in META},
                       **{n: repr(float((r[n] - mean[CANDIDATE_FEATURES.index(n)]) / scale[CANDIDATE_FEATURES.index(n)]))
                          for n in names}} for r in plain]
            write_csv(root / f"data/features/{split}_features_{k}.csv", [*META, *names], scaled)
    write_csv(root / "results/recording_split.csv", ["recording_id", "split"],
              [{"recording_id": rid, "split": split} for rid, split, _ in recordings])
    train_ids = sorted(rid for rid, split, _ in recordings if split == "train")
    subset = sorted([r for r in train_ids if label_of[r] == 0][:12] + [r for r in train_ids if label_of[r] == 1][:12])
    write_csv(root / "results/small_data_subsets/train_subset_25.csv",
              ["requested_recordings", "actual_recordings", "recording_id", "sample_id", "binary_label"],
              [{"requested_recordings": 25, "actual_recordings": 24, "recording_id": r["recording_id"],
                "sample_id": r["sample_id"], "binary_label": r["binary_label"]}
               for r in rows["train"] if r["recording_id"] in subset])
    (root / "results/selected_features.json").write_text(json.dumps({
        "ranking_unit": "recording_id mean across training segments", "recordings_used": len(train_ids),
        "qubit_mappings": {str(k): {"feature_count": k, "qubits": k, "features": SELECTED[:k]} for k in (4, 5, 6)}}))
    val_ids = sorted(rid for rid, split, _ in recordings if split == "validation")
    test_ids = sorted(rid for rid, split, _ in recordings
                      if split == "test" and (rid != "te1silent0" or segments > 1))
    write_csv(root / "results/quantum/ideal_simulator_results.csv",
              ["model", "feature_count", "training_size_requested", "training_sample_manifest_hash",
               "validation_sample_manifest_hash", "test_sample_manifest_hash"],
              [{"model": "fixed_qsvc", "feature_count": 4, "training_size_requested": 24,
                "training_sample_manifest_hash": her_hash(subset), "validation_sample_manifest_hash": her_hash(val_ids),
                "test_sample_manifest_hash": her_hash(test_ids)}])
    return {"rows": rows, "subset": subset, "val_ids": val_ids, "test_ids": test_ids, "train_ids": train_ids,
            "label_of": label_of, "mean": mean, "scale": scale}


def rewrite(path, change):
    """Apply change(rows) to a CSV in place."""
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        header, rows = reader.fieldnames, list(reader)
    change(rows)
    write_csv(path, header, rows)
