"""Train one baseline or compare A/B/C/D on shared, reproducible data."""

import argparse
import csv
from dataclasses import fields, replace
import importlib.util
import json
from pathlib import Path
import time

import numpy as np
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from .artifacts import (
    PortableExportUnsupported, export_portable, portable_model, portable_steps, save_bundle, versions,
)
from .data import (
    SILENCE_PEAK, Windows, corrupt, drop_silent, features, load_windows, make_windows,
    noise_indices, read_manifest, read_premixed, split_clips, take, window_samples, write_manifest,
)
from .models import (
    AGGREGATIONS, SELECTION_METRICS, aggregate_clips, aggregate_features, fit_tabular, metrics,
    mlp_parameter_count, scores_for,
)
from .subsets import balanced_subsets, group_labels

# "legacy" is the original default; "shared" is the protocol of Dhanya's quantum experiments.
SNR_PRESETS = {"legacy": ["clean", "20", "10", "0", "-10"], "shared": ["clean", "20", "10", "5", "0"]}
LEVELS = ("window", "clip", "recording")
FEATURE_ORDER = [f"mfcc_{i}_{stat}" for stat in ("mean", "std") for i in range(1, 14)]
THRESHOLDS = {"A": 0.0, "B": 0.5, "C": 0.5, "D": 0.5}
INPUTS = {
    "tabular": "MFCC mean/std -> StandardScaler -> PCA -> [0, pi]",
    "D": "64-bin log-mel spectrogram (richer-input classical reference, not feature-matched)",
}


def parse_snr(value):
    if value.lower() == "clean":
        return "clean"
    try:
        number = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("SNR must be 'clean' or a number in dB.") from error
    if not np.isfinite(number) or not -100 <= number <= 100:
        raise argparse.ArgumentTypeError("SNR must be finite and between -100 and 100 dB.")
    return f"{number:g}"


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("Value must be a positive integer.")
    return number


def premixed_item(value):
    snr, _, path = value.partition("=")
    snr = parse_snr(snr)
    if snr == "clean" or not path:
        raise argparse.ArgumentTypeError("Use SNR=MANIFEST with a noisy SNR, for example 10=premixed_10.csv.")
    return snr, Path(path)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True, type=Path, help="CSV: path,label[,group_id,split]")
    p.add_argument("--noise-manifest", type=Path, help="Separate background CSV: path[,group_id,split]")
    p.add_argument("--models", nargs="+", type=str.upper, choices=["A", "B", "C", "D", "ALL"], default=["A"])
    p.add_argument("--pca", nargs="+", type=int, choices=[2, 4, 6], default=[2, 4, 6])
    p.add_argument("--snrs", nargs="+", type=parse_snr, default=None,
                   help="Test conditions; default is the --snr-preset list")
    p.add_argument("--snr-preset", choices=sorted(SNR_PRESETS), default="legacy",
                   help="legacy: clean 20 10 0 -10; shared: clean 20 10 5 0")
    p.add_argument("--train-snrs", nargs="*", type=parse_snr, default=[], help="Additional noisy training copies; clean always included")
    p.add_argument("--window-seconds", type=float, default=1.0, help="Window duration, for example 1 or 3")
    p.add_argument("--silent-policy", choices=["exclude", "error"], default="exclude",
                   help="exclude: drop and record silent windows in every split; error: abort noisy runs on one")
    p.add_argument("--silence-peak", type=float, default=SILENCE_PEAK,
                   help="A window is silent when its peak amplitude is at or below this value")
    p.add_argument("--premixed-test", nargs="*", type=premixed_item, default=[], metavar="SNR=MANIFEST",
                   help="Use existing noisy test audio for a condition (CSV: path,source_path) instead of mixing")
    p.add_argument("--aggregation", choices=sorted(AGGREGATIONS), default="score-mean",
                   help="A-C recording-level procedure; D always averages window scores")
    p.add_argument("--levels", nargs="+", choices=LEVELS, default=["window", "clip"],
                   help="Evaluation units to report for window-scored models")
    p.add_argument("--selection-metric", choices=SELECTION_METRICS, default="balanced_accuracy",
                   help="Validation-only selection of A-C hyperparameters")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--seeds", nargs="*", type=int, default=[],
                   help="Run seeds for a sweep (subset draw and model initialization); --seed still fixes the split and noise")
    p.add_argument("--train-recordings", nargs="*", type=positive_int, default=[],
                   help="Balanced training-set sizes in recordings; the same subsets are used by every model")
    p.add_argument("--val-size", type=float, default=0.2)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--small-width", type=positive_int, default=4, help="Model B hidden units; adjust to your quantum parameter budget")
    p.add_argument("--max-iter", type=positive_int, default=500, help="MLP optimizer iteration limit")
    p.add_argument("--epochs", type=positive_int, default=30, help="Model D training epochs")
    p.add_argument("--batch-size", type=positive_int, default=32, help="Model D batch size")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto",
                   help="Model D only: auto uses a usable visible CUDA GPU, otherwise CPU")
    p.add_argument("--dry-run", action="store_true",
                   help="Validate inputs, load audio, and write the plan; fit and evaluate nothing")
    p.add_argument("--output", type=Path, default=Path("runs/classical"), help="New/empty output directory")
    return p


def save_csv(path, rows, fieldnames=None):
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plain(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value


def fit_preprocessor(train_raw, k):
    transform = make_pipeline(StandardScaler(), PCA(n_components=k, svd_solver="full"),
                              MinMaxScaler(feature_range=(0, np.pi), clip=True))
    return transform, transform.fit_transform(train_raw)


def premixed_windows(test, mapping, window_seconds):
    """Windows of already-mixed audio, aligned to the clean test windows."""
    cache, audio = {}, []
    for clip, position in zip(test.clip_ids, test.window_ids):
        if clip not in cache:
            # Mixtures are used as supplied: no normalization after mixing.
            cache[clip] = load_windows(mapping[clip], window_seconds, normalize=False)
        if position >= len(cache[clip]):
            raise ValueError(f"Premixed audio is shorter than its clean clip: {mapping[clip]}")
        audio.append(cache[clip][position])
    return replace(test, audio=np.stack(audio))


def noise_rows(split, windows, pool, seed):
    return [{"split": split, "clip_id": c, "window_id": int(w), "group_id": g, "noise_clip_id": pool.clip_ids[i],
             "noise_window_id": int(pool.window_ids[i]), "noise_group_id": pool.group_ids[i]}
            for c, w, g, i in zip(windows.clip_ids, windows.window_ids, windows.group_ids,
                                  noise_indices(len(windows.audio), len(pool.audio), seed))]


def run(args):
    models = list("ABCD") if "ALL" in args.models else list(dict.fromkeys(args.models))
    snrs = list(dict.fromkeys(args.snrs or SNR_PRESETS[args.snr_preset]))
    train_snrs = [s for s in dict.fromkeys(args.train_snrs) if s != "clean"]
    seeds = list(dict.fromkeys(args.seeds or [args.seed]))
    sizes = list(dict.fromkeys(args.train_recordings))
    sweep = bool(args.seeds or sizes)
    levels = [level for level in LEVELS if level in args.levels]
    feature_mean = args.aggregation == "feature-mean"
    if any(seed < 0 or seed >= 2**32 - 100 for seed in (args.seed, *seeds)):
        raise ValueError("seed must be in [0, 2**32 - 100).")
    window_samples(args.window_seconds)
    premixed = dict(args.premixed_test)
    if set(premixed) - set(snrs):
        raise ValueError(f"--premixed-test names conditions that are not evaluated: {sorted(set(premixed) - set(snrs))}")
    if "D" in models and not args.dry_run and importlib.util.find_spec("torch") is None:
        raise ValueError('Model D requires PyTorch: pip install -e ".[cnn]"')
    cnn_device = None
    if "D" in models and not args.dry_run:
        from .cnn import device_name, resolve_device
        cnn_device = resolve_device(args.device)
        print(f"Model D device: {cnn_device} ({device_name(cnn_device)})", flush=True)
    clips = split_clips(read_manifest(args.manifest), args.seed, args.val_size, args.test_size)
    mixed_snrs = [s for s in snrs if s != "clean" and s not in premixed]
    need_noise = bool(mixed_snrs) or bool(train_snrs)
    noises = []
    if need_noise:
        if args.noise_manifest is None:
            raise ValueError("Noisy evaluation requires --noise-manifest; use --snrs clean for a clean-only run.")
        noises = split_clips(read_manifest(args.noise_manifest, noise=True), args.seed,
                             args.val_size, args.test_size, noise=True)
        if {c.path for c in clips} & {c.path for c in noises}:
            raise ValueError("Signal and noise manifests must not share audio files.")
        if {c.group_id for c in clips} & {c.group_id for c in noises}:
            raise ValueError("Signal and noise manifests must not share source recording groups.")
        required_noise_splits = (["train"] if train_snrs else []) + (["test"] if mixed_snrs else [])
        for split in required_noise_splits:
            if not any(c.split == split for c in noises):
                raise ValueError(f"Noise manifest needs {split} recordings.")
    premixed = {snr: read_premixed(path) for snr, path in premixed.items()}
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        raise ValueError(f"Output must be a new or empty directory: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    write_manifest(args.output / "splits.csv", clips)
    if noises:
        write_manifest(args.output / "noise_splits.csv", noises)
    # Recording-level selection whenever recordings, not clips, are the reported unit.
    selection_unit = "recording" if feature_mean or ("recording" in levels and "clip" not in levels) else "clip"
    config = {k: plain(v) for k, v in vars(args).items()}
    config["versions"] = versions()
    if cnn_device is not None:
        config["cnn_device"] = str(cnn_device)
        config["cnn_device_name"] = device_name(cnn_device)
    config["audio"] = {"sample_rate": 16000, "window_seconds": args.window_seconds, "n_fft": 512,
                       "hop_length": 160, "n_mels": 64, "n_mfcc": 13, "tail": "zero-pad",
                       "window_normalization": "per-window peak", "after_mixing": "no clipping or normalization"}
    config["selection_metric"] = f"clean validation {selection_unit} {args.selection_metric.replace('_', ' ')}"
    config["selection"] = {"A-C": config["selection_metric"],
                           "D": f"best epoch by clean validation {selection_unit} balanced accuracy",
                           "test": "never used for selection"}
    config["snrs"] = snrs
    config["feature_mode"] = "mfcc_pca_angle"
    config["feature_order"] = FEATURE_ORDER
    config["preprocessing"] = {"A-C": INPUTS["tabular"] + ", fit on training rows only", "D": INPUTS["D"]}
    config["aggregation"] = {"A-C": AGGREGATIONS[args.aggregation], "D": AGGREGATIONS["score-mean"]}
    config["thresholds"] = {m: THRESHOLDS[m] for m in models}
    config["seeds"] = {"split": args.seed, "train_noise": args.seed + 1, "test_noise": args.seed + 2, "runs": seeds}
    config["silent_windows"] = {"policy": args.silent_policy, "peak_at_or_below": args.silence_peak}
    config["parameter_counts"] = {"B": {str(k): mlp_parameter_count(k, (args.small_width,)) for k in args.pca}}
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print("Loading audio after recording-level splitting...", flush=True)
    windows = {s: make_windows(clips, s, args.window_seconds) for s in ("train", "val", "test")}
    exclusions = []
    lacking = np.array([any(clip not in mapping for mapping in premixed.values()) for clip in windows["test"].clip_ids])
    if args.silent_policy == "exclude":
        # One exclusion set, fixed before any model runs: every model and every SNR
        # condition is then scored on identical examples.
        exclusions += [{"split": "test", "clip_id": c, "group_id": g, "window_id": int(w),
                        "reason": "no_premixed_copy", "raw_peak": float(p)}
                       for c, g, w, p in zip(*(getattr(windows["test"], a)[lacking]
                                               for a in ("clip_ids", "group_ids", "window_ids", "raw_peaks")))]
        windows["test"] = take(windows["test"], ~lacking)
        for split in windows:
            windows[split], rows = drop_silent(windows[split], split, args.silence_peak)
            exclusions += rows
    elif lacking.any():
        raise ValueError(f"{int(lacking.sum())} test windows have no premixed copy; use --silent-policy exclude to drop them.")
    save_csv(args.output / "exclusions.csv", exclusions,
             ["split", "clip_id", "group_id", "window_id", "reason", "raw_peak"])
    for split, data in windows.items():
        if set(np.unique(data.labels)) != {0, 1}:
            raise ValueError(f"{split} must contain both labels 0 and 1 after excluding windows.")
    noise_pools = {}
    for split in ("train", "test"):
        if (split == "train" and train_snrs) or (split == "test" and mixed_snrs):
            pool = make_windows(noises, split, args.window_seconds)
            pool = take(pool, np.max(np.abs(pool.audio), axis=1) > 1e-12)
            if not len(pool.audio):
                raise ValueError(f"All {split} noise windows are silent.")
            noise_pools[split] = pool
    clean_train = windows["train"]
    augmented = [clean_train] + [corrupt(clean_train, noise_pools["train"].audio, snr, args.seed + 1) for snr in train_snrs]
    train = Windows(*(np.concatenate([getattr(w, f.name) for w in augmented]) for f in fields(Windows)))
    train_snr = np.repeat(["clean", *train_snrs], len(clean_train.labels))
    val = windows["val"]
    test = windows["test"]
    conditions = {s: premixed_windows(test, premixed[s], args.window_seconds) if s in premixed
                  else corrupt(test, noise_pools["test"].audio if "test" in noise_pools else None, s, args.seed + 2)
                  for s in snrs}
    assignment = ((noise_rows("train", clean_train, noise_pools["train"], args.seed + 1) if train_snrs else [])
                  + (noise_rows("test", test, noise_pools["test"], args.seed + 2) if mixed_snrs else []))
    if assignment:
        save_csv(args.output / "noise_assignment.csv", assignment)
    for split, data in windows.items():
        print(f"  {split}: {len(np.unique(data.clip_ids))} clips / {len(data.labels)} windows", flush=True)
    if exclusions:
        print(f"  excluded {len(exclusions)} windows (see exclusions.csv)", flush=True)
    # The same recordings train every model of a given seed and size.
    subsets = {seed: balanced_subsets(group_labels(clean_train.group_ids, clean_train.labels), sizes, seed)
               if sizes else {None: None} for seed in seeds}
    plan = [(seed, size) for seed in seeds for size in subsets[seed]]
    (args.output / "split_ids.json").write_text(json.dumps(
        {s: sorted({c.group_id for c in clips if c.split == s}) for s in ("train", "val", "test")}, indent=2) + "\n")
    (args.output / "subsets.json").write_text(json.dumps(
        [{"seed": seed, "train_recordings": size, "group_ids": subsets[seed][size]} for seed, size in plan
         if size is not None], indent=2) + "\n")
    tabular = [m for m in models if m != "D"]
    planned = [{"model": m, "pca_k": k, "seed": seed, "train_recordings": size}
               for seed, size in plan for m in tabular for k in dict.fromkeys(args.pca)]
    planned += [{"model": "D", "pca_k": "", "seed": seed, "train_recordings": size} for seed, size in plan if "D" in models]
    (args.output / "plan.json").write_text(json.dumps(planned, indent=2) + "\n")
    if args.dry_run:
        print(f"Dry run: {len(planned)} fits planned, none started. Plan written to {args.output / 'plan.json'}", flush=True)
        return []
    results, predictions, selections = [], [], []
    clip_groups = dict(zip(test.clip_ids, test.group_ids))
    recordings_total = len(np.unique(clean_train.group_ids))

    def run_paths(seed, size):
        directory = args.output / f"seed{seed}_n{size or 'all'}" if sweep else args.output
        (directory / "artifacts").mkdir(parents=True, exist_ok=True)
        return directory / "artifacts"

    def train_mask(seed, size):
        if size is None:
            return np.ones(len(train.labels), dtype=bool)
        return np.isin(train.group_ids, subsets[seed][size])

    def evaluate(name, k, model, info, test_features, units, seed, size):
        common = {"model": name, "pca_k": k, "seed": seed, "train_recordings": size or recordings_total}
        selections.append({**common, **info})
        for snr, x in test_features.items():
            start = time.perf_counter()
            scores = scores_for(model, x)
            predict_seconds = time.perf_counter() - start
            for level in units["levels"]:
                if units["aggregated"]:
                    # One feature vector per recording was classified; nothing is averaged here.
                    labels, values, groups = units["labels"], scores, units["group_ids"]
                    ids, window_ids, how = [""] * len(labels), [""] * len(labels), "feature_mean_per_recording"
                elif level == "window":
                    labels, values, ids, groups = test.labels, scores, test.clip_ids, test.group_ids
                    window_ids, how = test.window_ids, "none"
                elif level == "clip":
                    labels, values, ids = aggregate_clips(test.labels, scores, test.clip_ids)
                    groups, window_ids, how = [clip_groups[c] for c in ids], [""] * len(labels), "score_mean_per_clip"
                else:
                    labels, values, groups = aggregate_clips(test.labels, scores, test.group_ids)
                    ids, window_ids, how = [""] * len(labels), [""] * len(labels), "score_mean_per_recording"
                results.append({
                    "model": name, "pca_k": k, "snr_db": snr, "level": level,
                    **metrics(labels, values, info["threshold"]),
                    "val_clip_balanced_accuracy": info["val_clip_balanced_accuracy"],
                    "fit_seconds": info["fit_seconds"], "predict_seconds": predict_seconds,
                    "parameter_count": info["parameter_count"], "support_vectors": info["support_vectors"],
                    "seed": seed, "train_recordings": common["train_recordings"],
                    "window_seconds": args.window_seconds, "aggregation": how,
                    "input": INPUTS["D" if name == "D" else "tabular"],
                })
                for label, score, clip_id, position, group in zip(labels, values, ids, window_ids, groups):
                    predictions.append({"model": name, "pca_k": k, "snr_db": snr, "level": level,
                                        "clip_id": clip_id, "window_id": position, "label": int(label),
                                        "score": float(score), "prediction": int(score >= info["threshold"]),
                                        "group_id": group, "seed": seed,
                                        "train_recordings": common["train_recordings"]})
        # Keep completed model results available during long comparisons.
        save_csv(args.output / "results.csv", results)
        save_csv(args.output / "predictions.csv", predictions)
        (args.output / "selection.json").write_text(json.dumps(selections, indent=2) + "\n")

    window_units = {"aggregated": False, "levels": levels}
    val_ids = val.group_ids if selection_unit == "recording" else val.clip_ids
    if tabular:
        print("Extracting 26-dimensional MFCC mean/std features...", flush=True)
        train_raw, val_raw = features(train.audio), features(val.audio)
        test_raw = {s: features(w.audio) for s, w in conditions.items()}
        val_y, val_units, units = val.labels, val_ids, window_units
        if feature_mean:
            val_raw, val_y, val_units = aggregate_features(val_raw, val.labels, val.group_ids)
            test_y, test_groups = aggregate_features(test_raw[snrs[0]], test.labels, test.group_ids)[1:]
            test_raw = {s: aggregate_features(x, test.labels, test.group_ids)[0] for s, x in test_raw.items()}
            units = {"aggregated": True, "levels": ["recording"], "labels": test_y, "group_ids": test_groups}
        for seed, size in plan:
            artifacts, mask = run_paths(seed, size), train_mask(seed, size)
            fit_raw, fit_y, fit_snr = train_raw[mask], train.labels[mask], train_snr[mask]
            fit_clips, fit_positions, fit_groups = train.clip_ids[mask], train.window_ids[mask], train.group_ids[mask]
            if feature_mean:
                # A recording contributes one example per training condition.
                keys = np.array([f"{g}\x1f{s}" for g, s in zip(fit_groups, fit_snr)])
                fit_raw, fit_y, keys = aggregate_features(fit_raw, fit_y, keys)
                fit_groups, fit_snr = (np.array(part) for part in zip(*(key.split("\x1f") for key in keys)))
                fit_clips = fit_positions = np.array([""] * len(keys))
            for k in dict.fromkeys(args.pca):
                if k > min(fit_raw.shape):
                    raise ValueError(f"PCA k={k} requires at least {k} training windows.")
                transform, train_x = fit_preprocessor(fit_raw, k)
                val_x = transform.transform(val_raw)
                test_x = {s: transform.transform(x) for s, x in test_raw.items()}
                save_bundle(artifacts / f"preprocessor_k{k}.joblib", transform, feature_order=FEATURE_ORDER)
                arrays = {"train_x": train_x, "train_y": fit_y, "train_clip_ids": fit_clips,
                          "train_window_ids": fit_positions, "train_group_ids": fit_groups,
                          "train_snr_db": fit_snr, "val_x": val_x, "val_y": val_y, "test_y": test.labels,
                          "val_clip_ids": val.clip_ids, "val_window_ids": val.window_ids,
                          "val_group_ids": val.group_ids, "test_clip_ids": test.clip_ids,
                          "test_window_ids": test.window_ids, "test_group_ids": test.group_ids}
                if feature_mean:
                    arrays.update(val_group_ids=val_units, test_y=units["labels"], test_group_ids=units["group_ids"])
                    for dropped in ("val_clip_ids", "val_window_ids", "test_clip_ids", "test_window_ids"):
                        del arrays[dropped]
                arrays.update({f"test_x_{s}": x for s, x in test_x.items()})
                np.savez_compressed(artifacts / f"features_k{k}.npz", **arrays)
                for name in tabular:
                    print(f"Training model {name}, PCA k={k}, seed {seed}, "
                          f"{size or recordings_total} training recordings...", flush=True)
                    model, info = fit_tabular(name, train_x, fit_y, val_x, val_y, val_units, seed=seed,
                                              small_width=args.small_width, max_iter=args.max_iter,
                                              selection=args.selection_metric)
                    described = {"model_id": name, "pca_k": k, "seed": seed, "feature_order": FEATURE_ORDER,
                                 "aggregation": AGGREGATIONS[args.aggregation], "audio": config["audio"]}
                    save_bundle(artifacts / f"model_{name}_k{k}.joblib",
                                {"preprocessor": transform, "model": model, "threshold": info["threshold"]},
                                **described)
                    evaluate(name, k, model, info, test_x, units, seed, size)
                    try:
                        export_portable(artifacts / f"model_{name}_k{k}.portable.json",
                                        preprocessing=portable_steps(transform), model=portable_model(model),
                                        threshold=info["threshold"], **described)
                    except PortableExportUnsupported as error:
                        print(f"  no portable export for model {name}: {error}", flush=True)
    if "D" in models:
        from .cnn import fit_cnn
        print("Training model D on log-mel spectrograms (no PCA)...", flush=True)
        train_x, val_x = features(train.audio, mel=True), features(val.audio, mel=True)
        test_x = {s: features(w.audio, mel=True) for s, w in conditions.items()}
        # D has no per-recording feature vector: its recording score is the mean window score.
        d_levels = [level for level in LEVELS if level in levels or (feature_mean and level == "recording")]
        for seed, size in plan:
            mask = train_mask(seed, size)
            model, info = fit_cnn(train_x[mask], train.labels[mask], val_x, val.labels, val_ids,
                                  seed=seed, epochs=args.epochs, batch_size=args.batch_size,
                                  device=cnn_device)
            model.save(run_paths(seed, size) / "model_D.pt")
            evaluate("D", "", model, info, test_x, {"aggregated": False, "levels": d_levels}, seed, size)
    shown = "clip" if any(row["level"] == "clip" for row in results) else "recording"
    print(f"\n{shown.capitalize()}-level comparison (test):")
    print("model  PCA    SNR       accuracy  balanced_acc  F1")
    for row in results:
        if row["level"] == shown:
            print(f"{row['model']:5}  {str(row['pca_k']) or '-':5}  {row['snr_db']:8}  "
                  f"{row['accuracy']:.3f}     {row['balanced_accuracy']:.3f}         {row['f1']:.3f}")
    print(f"\nSaved results to {args.output / 'results.csv'}", flush=True)
    return results


def main():
    p = parser()
    args = p.parse_args()
    try:
        run(args)
    except (ValueError, FileNotFoundError) as error:
        p.error(str(error))
