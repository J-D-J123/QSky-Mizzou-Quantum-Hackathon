"""Train one baseline or compare A/B/C/D on shared, reproducible data."""

import argparse
import csv
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import time

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from .data import (
    Windows, corrupt, features, make_windows, read_manifest, split_clips, write_manifest,
)
from .models import aggregate_clips, fit_tabular, metrics, scores_for


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


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True, type=Path, help="CSV: path,label[,group_id,split]")
    p.add_argument("--noise-manifest", type=Path, help="Separate background CSV: path[,group_id,split]")
    p.add_argument("--models", nargs="+", type=str.upper, choices=["A", "B", "C", "D", "ALL"], default=["A"])
    p.add_argument("--pca", nargs="+", type=int, choices=[2, 4, 6], default=[2, 4, 6])
    p.add_argument("--snrs", nargs="+", type=parse_snr, default=["clean", "20", "10", "0", "-10"])
    p.add_argument("--train-snrs", nargs="*", type=parse_snr, default=[], help="Additional noisy training copies; clean always included")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-size", type=float, default=0.2)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--small-width", type=positive_int, default=4, help="Model B hidden units; adjust to your quantum parameter budget")
    p.add_argument("--max-iter", type=positive_int, default=500, help="MLP optimizer iteration limit")
    p.add_argument("--epochs", type=positive_int, default=30, help="Model D training epochs")
    p.add_argument("--batch-size", type=positive_int, default=32, help="Model D batch size")
    p.add_argument("--output", type=Path, default=Path("runs/classical"), help="New/empty output directory")
    return p


def save_csv(path, rows):
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(args):
    models = list("ABCD") if "ALL" in args.models else list(dict.fromkeys(args.models))
    snrs = list(dict.fromkeys(args.snrs))
    train_snrs = [s for s in dict.fromkeys(args.train_snrs) if s != "clean"]
    if args.seed < 0 or args.seed >= 2**32 - 100:
        raise ValueError("seed must be in [0, 2**32 - 100).")
    if "D" in models and importlib.util.find_spec("torch") is None:
        raise ValueError('Model D requires PyTorch: pip install -e ".[cnn]"')
    clips = split_clips(read_manifest(args.manifest), args.seed, args.val_size, args.test_size)
    need_noise = any(s != "clean" for s in snrs) or bool(train_snrs)
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
        required_noise_splits = (["train"] if train_snrs else []) + (["test"] if any(s != "clean" for s in snrs) else [])
        for split in required_noise_splits:
            if not any(c.split == split for c in noises):
                raise ValueError(f"Noise manifest needs {split} recordings.")
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        raise ValueError(f"Output must be a new or empty directory: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    artifacts = args.output / "artifacts"
    artifacts.mkdir()
    write_manifest(args.output / "splits.csv", clips)
    if noises:
        write_manifest(args.output / "noise_splits.csv", noises)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config["versions"] = {p: importlib.metadata.version(p) for p in ("numpy", "librosa", "scikit-learn")}
    if "D" in models:
        config["versions"]["torch"] = importlib.metadata.version("torch")
    config["audio"] = {"sample_rate": 16000, "window_seconds": 1, "n_fft": 512,
                       "hop_length": 160, "n_mels": 64, "n_mfcc": 13, "tail": "zero-pad"}
    config["selection_metric"] = "clean validation clip balanced accuracy"
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print("Loading audio after recording-level splitting...", flush=True)
    windows = {s: make_windows(clips, s) for s in ("train", "val", "test")}
    noise_pools = {}
    for split in ("train", "test"):
        if (split == "train" and train_snrs) or (split == "test" and any(s != "clean" for s in snrs)):
            pool = make_windows(noises, split).audio
            pool = pool[np.max(np.abs(pool), axis=1) > 1e-12]
            if not len(pool):
                raise ValueError(f"All {split} noise windows are silent.")
            noise_pools[split] = pool
    clean_train = windows["train"]
    augmented = [clean_train] + [corrupt(clean_train, noise_pools["train"], snr, args.seed + 1) for snr in train_snrs]
    train = Windows(*(np.concatenate([getattr(w, attr) for w in augmented])
                      for attr in ("audio", "labels", "clip_ids", "window_ids")))
    val = windows["val"]
    test = windows["test"]
    conditions = {s: corrupt(test, noise_pools.get("test"), s, args.seed + 2) for s in snrs}
    for split, data in windows.items():
        print(f"  {split}: {len(np.unique(data.clip_ids))} clips / {len(data.labels)} windows", flush=True)
    results, predictions, selections = [], [], []

    def evaluate(name, k, model, info, test_features):
        selections.append({"model": name, "pca_k": k, **info})
        for snr, x in test_features.items():
            start = time.perf_counter()
            scores = scores_for(model, x)
            predict_seconds = time.perf_counter() - start
            for level in ("window", "clip"):
                if level == "clip":
                    labels, values, ids = aggregate_clips(test.labels, scores, test.clip_ids)
                    window_ids = [""] * len(labels)
                else:
                    labels, values, ids = test.labels, scores, test.clip_ids
                    window_ids = test.window_ids
                results.append({
                    "model": name, "pca_k": k, "snr_db": snr, "level": level,
                    **metrics(labels, values, info["threshold"]),
                    "val_clip_balanced_accuracy": info["val_clip_balanced_accuracy"],
                    "fit_seconds": info["fit_seconds"], "predict_seconds": predict_seconds,
                    "parameter_count": info["parameter_count"], "support_vectors": info["support_vectors"],
                    "seed": args.seed,
                })
                for label, score, clip_id, position in zip(labels, values, ids, window_ids):
                    predictions.append({"model": name, "pca_k": k, "snr_db": snr, "level": level,
                                        "clip_id": clip_id, "window_id": position, "label": int(label),
                                        "score": float(score), "prediction": int(score >= info["threshold"])})
        # Keep completed model results available during long comparisons.
        save_csv(args.output / "results.csv", results)
        save_csv(args.output / "predictions.csv", predictions)
        (args.output / "selection.json").write_text(json.dumps(selections, indent=2) + "\n")

    tabular = [m for m in models if m != "D"]
    if tabular:
        print("Extracting 26-dimensional MFCC mean/std features...", flush=True)
        train_raw, val_raw = features(train.audio), features(val.audio)
        test_raw = {s: features(w.audio) for s, w in conditions.items()}
        for k in dict.fromkeys(args.pca):
            if k > min(train_raw.shape):
                raise ValueError(f"PCA k={k} requires at least {k} training windows.")
            transform = make_pipeline(StandardScaler(), PCA(n_components=k, svd_solver="full"),
                                      MinMaxScaler(feature_range=(0, np.pi), clip=True))
            train_x = transform.fit_transform(train_raw)
            val_x = transform.transform(val_raw)
            test_x = {s: transform.transform(x) for s, x in test_raw.items()}
            joblib.dump(transform, artifacts / f"preprocessor_k{k}.joblib")
            arrays = {"train_x": train_x, "train_y": train.labels, "train_clip_ids": train.clip_ids,
                      "train_window_ids": train.window_ids,
                      "train_snr_db": np.repeat(["clean", *train_snrs], len(clean_train.labels)),
                      "val_x": val_x, "val_y": val.labels, "val_clip_ids": val.clip_ids,
                      "val_window_ids": val.window_ids,
                      "test_y": test.labels, "test_clip_ids": test.clip_ids, "test_window_ids": test.window_ids}
            arrays.update({f"test_x_{s}": x for s, x in test_x.items()})
            np.savez_compressed(artifacts / f"features_k{k}.npz", **arrays)
            for name in tabular:
                print(f"Training model {name}, PCA k={k}...", flush=True)
                model, info = fit_tabular(name, train_x, train.labels, val_x, val.labels, val.clip_ids,
                                          seed=args.seed, small_width=args.small_width, max_iter=args.max_iter)
                joblib.dump({"preprocessor": transform, "model": model, "threshold": info["threshold"]},
                            artifacts / f"model_{name}_k{k}.joblib")
                evaluate(name, k, model, info, test_x)
    if "D" in models:
        from .cnn import fit_cnn
        print("Training model D on log-mel spectrograms (no PCA)...", flush=True)
        train_x, val_x = features(train.audio, mel=True), features(val.audio, mel=True)
        test_x = {s: features(w.audio, mel=True) for s, w in conditions.items()}
        model, info = fit_cnn(train_x, train.labels, val_x, val.labels, val.clip_ids,
                              seed=args.seed, epochs=args.epochs, batch_size=args.batch_size)
        model.save(artifacts / "model_D.pt")
        evaluate("D", "", model, info, test_x)
    print("\nClip-level comparison (test):")
    print("model  PCA    SNR       accuracy  balanced_acc  F1")
    for row in results:
        if row["level"] == "clip":
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
