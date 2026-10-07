"""Models A-C on Dhanya-matched selected features, one example per recording.

Two protocols, never mixed in one comparison:

reproduce  Her saved protocol: features ranked on all training recordings, her
           scaler fit on every training row, then a 24/50/70-recording subset.
           Inputs are her own standardized values, so they equal the quantum inputs.
strict     Label-budget protocol: features are ranked and the scaler is fit inside
           each actual training subset. Published quantum scores do not apply to
           it; the exported inputs let the quantum models be rerun on the same data.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .artifacts import Standardizer, export_portable, ids_sha256, portable_model, save_bundle, versions
from .models import AGGREGATIONS, SELECTION_METRICS, aggregate_features, fit_tabular, metrics, mlp_parameter_count, scores_for
from .subsets import balanced_subsets

PROTOCOLS = {"reproduce": "reproduction_dhanya_v1", "strict": "strict_label_budget_v1"}
HER_SUBSET_SEED = 42
SNRS = ("clean", "20", "10", "5", "0")
MODEL_NAMES = {"A": "RBF SVM", "B": "small MLP", "C": "larger MLP"}


def load_bundle(path):
    path = Path(path)
    bundle = json.loads((path / "bundle.json").read_text())
    if bundle.get("status") != "ok":
        raise ValueError(f"{path} failed import validation ({len(bundle.get('errors', []))} problem(s)); "
                         "fix the assets and import again.")
    with (path / "rows.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return bundle, rows


def matrix(rows, columns):
    return np.array([[float(r[c]) for c in columns] for r in rows], dtype=np.float64)


def recording_units(rows, columns, transform=None):
    """Mean feature vector per recording, in sorted recording order."""
    if not rows:
        raise ValueError("No feature rows to aggregate.")
    x = matrix(rows, columns)
    if transform is not None:
        x = transform(x)
    return aggregate_features(x, np.array([int(r["label"]) for r in rows]),
                              np.array([r["recording_id"] for r in rows]))


def combine_ranks(names, mutual_information, anova_f):
    """Her rule: mean of the two ranks, ties broken by MI rank, ANOVA rank, then name."""
    def ranks(scores):
        scores = np.asarray(scores, dtype=float)
        return [1 + int((scores > s).sum()) for s in scores]
    rows = [{"feature": n, "mutual_information": float(m), "anova_f_score": float(f),
             "mutual_information_rank": a, "anova_rank": b, "mean_rank": (a + b) / 2}
            for n, m, f, a, b in zip(names, mutual_information, anova_f, ranks(mutual_information), ranks(anova_f))]
    rows.sort(key=lambda r: (r["mean_rank"], r["mutual_information_rank"], r["anova_rank"], r["feature"]))
    return rows


def rank_features(x, y, names, seed=HER_SUBSET_SEED):
    from sklearn.feature_selection import f_classif, mutual_info_classif
    if len(np.unique(y)) != 2:
        raise ValueError("Feature ranking needs both classes.")
    anova, _ = f_classif(x, y)
    return combine_ranks(names, mutual_info_classif(x, y, random_state=seed),
                         np.nan_to_num(anova, nan=0.0, posinf=0.0, neginf=0.0))


def training_subset(bundle, size, seed):
    saved = bundle["splits"]["train_subsets"].get(str(size))
    if seed == HER_SUBSET_SEED and saved:
        return saved, "dhanya_saved_manifest"
    labels = {i: bundle["labels"][i] for i in bundle["splits"]["train_all"]}
    return balanced_subsets(labels, [size], seed)[size], "resampled_from_her_training_split"


def build_inputs(bundle, rows, *, protocol, feature_count, size, seed, train_snr="clean", donor_policy="drop"):
    """Recording-level train/val/test matrices plus everything needed to audit them."""
    subset, subset_source = training_subset(bundle, size, seed)
    chosen = set(subset)
    val_ids, test_ids = set(bundle["splits"]["val"]), set(bundle["splits"]["test"])
    train_rows = [r for r in rows if r["split"] == "train" and r["recording_id"] in chosen and r["snr_db"] == train_snr]
    dropped = 0
    if protocol == "strict" and train_snr != "clean":
        outside = [r for r in train_rows if r["noise_recording_id"] not in chosen]
        if outside and donor_policy == "error":
            raise ValueError(f"{len(outside)} noisy training rows use donors outside the {size}-recording subset.")
        if donor_policy == "drop":
            train_rows, dropped = [r for r in train_rows if r["noise_recording_id"] in chosen], len(outside)
    if {r["recording_id"] for r in train_rows} != chosen:
        raise ValueError(f"Some of the {size} training recordings have no usable rows at training SNR {train_snr}.")

    val_rows = [r for r in rows if r["split"] == "val" and r["snr_db"] == "clean" and r["recording_id"] in val_ids]
    test_rows = {snr: [r for r in rows if r["split"] == "test" and r["snr_db"] == snr and r["recording_id"] in test_ids]
                 for snr in SNRS}
    if protocol == "strict":
        # Score the same segments under every condition: her rule keeps a recording whose
        # silent segments exist only in the clean table.
        test_rows = {snr: [r for r in rs if r["complete_noise"] == "1"] for snr, rs in test_rows.items()}

    ranking = None
    if protocol == "reproduce":
        names = bundle["selected_features"].get(str(feature_count))
        if not names:
            raise ValueError(f"The bundle holds no {feature_count}-feature set; import it with --feature-counts.")
        columns, transform = [f"scaled{feature_count}:{n}" for n in names], None
        preprocessing = [bundle["scalers_rebuilt_from_tables"][str(feature_count)]]
        fit_scope = {"feature_ranking_recordings": bundle["feature_ranking_recordings"],
                     "scaler_fit_recordings": len(bundle["splits"]["train_all"])}
    else:
        candidates = bundle["candidate_features"]
        clean_rows = [r for r in rows if r["split"] == "train" and r["recording_id"] in chosen and r["snr_db"] == "clean"]
        ranking_x, ranking_y, _ = recording_units(clean_rows, candidates)
        ranking = rank_features(ranking_x, ranking_y, candidates, seed)
        names = [r["feature"] for r in ranking[:feature_count]]
        scaler = Standardizer.from_rows(matrix(train_rows, names), names)
        columns, transform, preprocessing = names, scaler.transform, [scaler.to_dict()]
        fit_scope = {"feature_ranking_recordings": size, "scaler_fit_recordings": size}

    train_x, train_y, train_ids = recording_units(train_rows, columns, transform)
    val_x, val_y, val_order = recording_units(val_rows, columns, transform)
    tests = {snr: recording_units(rs, columns, transform) for snr, rs in test_rows.items()}
    test_order = tests["clean"][2]
    for snr, (_, _, ids) in tests.items():
        if list(ids) != list(test_order):
            raise ValueError(f"Test recordings differ at SNR {snr}; conditions are not comparable.")
    if set(train_ids) & (set(val_order) | set(test_order)) or set(val_order) & set(test_order):
        raise ValueError("Recordings overlap between train, validation, and test.")

    published = bundle["id_sha256"]
    same_ids = (subset_source == "dhanya_saved_manifest" and ids_sha256(train_ids) == published["train"].get(str(size))
                and ids_sha256(val_order) == published["val"] and ids_sha256(test_order) == published["test"])
    comparable = (protocol == "reproduce" and train_snr == "clean" and same_ids
                  and bool(bundle["verified_against_published"]))
    return {
        "protocol": PROTOCOLS[protocol], "features": names, "preprocessing": preprocessing, "ranking": ranking,
        "train_x": train_x, "train_y": train_y, "train_ids": train_ids,
        "val_x": val_x, "val_y": val_y, "val_ids": val_order,
        "test_x": {snr: t[0] for snr, t in tests.items()}, "test_y": tests["clean"][1], "test_ids": test_order,
        "subset_source": subset_source, "fit_scope": fit_scope,
        "noisy_training_rows_dropped_for_outside_donors": dropped,
        "segments": {"train": len(train_rows), "val": len(val_rows), "test": len(test_rows["clean"])},
        "comparable_to_published_quantum": bool(comparable),
        "comparability_note": (
            "Same recordings, values, feature order, preprocessing, and aggregation as her published runs."
            if comparable else
            "Not the protocol of the published quantum scores; rerun the quantum models on the exported inputs."),
    }


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("Value must be a positive integer.")
    return number


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bundle", required=True, type=Path, help="Directory written by qsky-dhanya import")
    p.add_argument("--protocol", required=True, choices=sorted(PROTOCOLS))
    p.add_argument("--models", nargs="+", type=str.upper, choices=["A", "B", "C", "ALL"], default=["ALL"])
    p.add_argument("--feature-counts", nargs="+", type=int, choices=[4, 5, 6], default=[4])
    p.add_argument("--train-recordings", nargs="+", type=positive_int, default=[24])
    p.add_argument("--seeds", nargs="+", type=int, default=[HER_SUBSET_SEED],
                   help=f"Seed {HER_SUBSET_SEED} uses her saved subsets; other seeds resample from her training split")
    p.add_argument("--train-snr", default="clean", choices=SNRS)
    p.add_argument("--donor-policy", choices=["drop", "error", "keep"], default="drop",
                   help="strict protocol with noisy training: rows whose noise donor lies outside the subset")
    p.add_argument("--selection-metric", choices=SELECTION_METRICS, default="f1",
                   help="Validation-only model selection; f1 (then balanced accuracy, recall) is her rule")
    p.add_argument("--small-width", type=positive_int, default=4, help="Model B hidden units")
    p.add_argument("--max-iter", type=positive_int, default=500, help="MLP optimizer iteration limit")
    p.add_argument("--dry-run", action="store_true", help="Build and export inputs, list the planned fits, train nothing")
    p.add_argument("--output", type=Path, default=Path("runs/matched"), help="New/empty output directory")
    return p


def save_csv(path, rows):
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(args):
    models = list("ABC") if "ALL" in args.models else list(dict.fromkeys(args.models))
    bundle, rows = load_bundle(args.bundle)
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        raise ValueError(f"Output must be a new or empty directory: {args.output}")
    plan = [(seed, size, k) for seed in dict.fromkeys(args.seeds) for size in dict.fromkeys(args.train_recordings)
            for k in dict.fromkeys(args.feature_counts)]
    # Build every input before fitting anything, so a bad configuration fails early.
    inputs = {key: build_inputs(bundle, rows, protocol=args.protocol, feature_count=key[2], size=key[1], seed=key[0],
                                train_snr=args.train_snr, donor_policy=args.donor_policy) for key in plan}
    args.output.mkdir(parents=True, exist_ok=True)
    artifacts = args.output / "artifacts"
    artifacts.mkdir()
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update({
        "versions": versions(), "protocol": PROTOCOLS[args.protocol], "feature_mode": "matched_selected",
        "aggregation": {"A-C": AGGREGATIONS["feature-mean"]}, "evaluation_unit": "recording",
        "audio": bundle["audio"], "thresholds": {"A": 0.0, "B": 0.5, "C": 0.5},
        "selection": f"validation only, {args.selection_metric}; test scores never select anything",
        "angle_scaling": "none: standardized values are used directly, as in her quantum feature maps",
        "source_bundle": str(Path(args.bundle).resolve()),
        "parameter_counts": {"B": {str(k): mlp_parameter_count(k, (args.small_width,)) for k in args.feature_counts},
                             "note": "B is not parameter-matched to the quantum model; see BENCHMARK_PROTOCOL.md"},
    })
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    planned = []
    for (seed, size, k), data in inputs.items():
        tag = f"seed{seed}_n{size}_k{k}"
        np.savez_compressed(
            artifacts / f"inputs_{tag}.npz", train_x=data["train_x"], train_y=data["train_y"],
            train_ids=data["train_ids"], val_x=data["val_x"], val_y=data["val_y"], val_ids=data["val_ids"],
            test_y=data["test_y"], test_ids=data["test_ids"], features=np.array(data["features"]),
            **{f"test_x_{snr}": x for snr, x in data["test_x"].items()})
        summary = {k2: data[k2] for k2 in ("protocol", "features", "preprocessing", "ranking", "subset_source",
                                           "fit_scope", "segments", "noisy_training_rows_dropped_for_outside_donors",
                                           "comparable_to_published_quantum", "comparability_note")}
        summary.update(seed=seed, train_recordings=size, feature_count=k,
                       split_ids={s: [str(i) for i in data[f"{s}_ids"]] for s in ("train", "val", "test")})
        (artifacts / f"inputs_{tag}.json").write_text(json.dumps(summary, indent=2) + "\n")
        planned += [{"model": m, "seed": seed, "train_recordings": size, "feature_count": k,
                     "comparable_to_published_quantum": data["comparable_to_published_quantum"]} for m in models]
    (args.output / "plan.json").write_text(json.dumps(planned, indent=2) + "\n")
    if args.dry_run:
        print(f"Dry run: {len(planned)} fits planned, none started. Inputs are in {artifacts}.", flush=True)
        return []

    results, predictions, selections = [], [], []
    for (seed, size, k), data in inputs.items():
        tag = f"seed{seed}_n{size}_k{k}"
        for name in models:
            print(f"Training model {name} ({MODEL_NAMES[name]}), {tag}...", flush=True)
            model, info = fit_tabular(name, data["train_x"], data["train_y"], data["val_x"], data["val_y"],
                                      data["val_ids"], seed=seed, small_width=args.small_width,
                                      max_iter=args.max_iter, selection=args.selection_metric)
            common = {"model": name, "feature_mode": "matched_selected", "protocol": data["protocol"],
                      "feature_count": k, "train_recordings": size, "seed": seed,
                      "subset_source": data["subset_source"]}
            save_bundle(artifacts / f"model_{name}_{tag}.joblib",
                        {"model": model, "threshold": info["threshold"], "features": data["features"],
                         "preprocessing": data["preprocessing"]}, **common)
            export_portable(artifacts / f"model_{name}_{tag}.portable.json", preprocessing=data["preprocessing"],
                            model=portable_model(model), threshold=info["threshold"], feature_order=data["features"],
                            aggregation=AGGREGATIONS["feature-mean"], audio=bundle["audio"],
                            **{("model_id" if key == "model" else key): value for key, value in common.items()})
            selections.append({**common, **info})
            for snr, x in data["test_x"].items():
                scores = scores_for(model, x)
                results.append({
                    **common, "snr_db": snr, "level": "recording", "aggregation": "feature_mean_per_recording",
                    **metrics(data["test_y"], scores, info["threshold"]),
                    "val_selection_key": json.dumps(info["val_selection_key"]),
                    "fit_seconds": info["fit_seconds"], "parameter_count": info["parameter_count"],
                    "support_vectors": info["support_vectors"],
                    "comparable_to_published_quantum": data["comparable_to_published_quantum"],
                })
                predictions += [{**common, "snr_db": snr, "recording_id": i, "label": int(y), "score": float(s),
                                 "prediction": int(s >= info["threshold"])}
                                for i, y, s in zip(data["test_ids"], data["test_y"], scores)]
            save_csv(args.output / "results.csv", results)
            save_csv(args.output / "predictions.csv", predictions)
            (args.output / "selection.json").write_text(json.dumps(selections, indent=2) + "\n")
    print(f"Saved results to {args.output / 'results.csv'}", flush=True)
    return results


def main():
    p = parser()
    args = p.parse_args()
    try:
        run(args)
    except (ValueError, FileNotFoundError) as error:
        p.error(str(error))


if __name__ == "__main__":
    main()
