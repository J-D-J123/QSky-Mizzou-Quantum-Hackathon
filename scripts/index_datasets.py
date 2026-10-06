"""Inventory completed local downloads without preprocessing or training."""

import csv
import json
from collections import Counter
from pathlib import Path
import re

import soundfile as sf

ROOT = Path(__file__).resolve().parents[1] / "data"
FIELDS = ["path", "dataset", "label", "group_id", "category", "fold", "role",
          "sample_rate", "channels", "frames", "duration_seconds", "usable_audio"]


def inspect(path, dataset, label, group, category, fold="", role=""):
    row = dict.fromkeys(FIELDS, "")
    row.update(path=str(path.relative_to(ROOT)), dataset=dataset, label=label,
               group_id=group, category=category, fold=fold, role=role)
    if path.suffix.lower() == ".wav":
        info = sf.info(path)
        row.update(sample_rate=info.samplerate, channels=info.channels, frames=info.frames,
                   duration_seconds=info.duration, usable_audio=info.frames > 0)
    return row


def write_inventory(dataset, rows):
    destination = ROOT / "inventories" / f"{dataset}.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "files": len(rows), "categories": dict(Counter(r["category"] for r in rows)),
        "labels": dict(Counter(str(r["label"]) for r in rows)),
        "source_groups": len({r["group_id"] for r in rows}),
        "empty_audio_files": sum(r["usable_audio"] is False for r in rows),
        "duration_seconds": sum(r["duration_seconds"] or 0 for r in rows),
    }
    destination.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(dataset, json.dumps(summary), flush=True)


def main():
    for dataset in ("ddl", "esc50", "urbansound8k", "nasa_external"):
        directory = ROOT / "raw" / dataset
        if not (directory / "download_complete.json").exists():
            print(f"{dataset}: download not yet complete; skipping", flush=True)
            continue
        rows = []
        if dataset == "ddl":
            for path in sorted(directory.rglob("*.wav")):
                match = re.match(r"^\d{14}(MINI|PRO4|XXXX)", path.stem)
                if not match:
                    raise ValueError(f"Unknown DDL filename; refusing to guess label: {path}")
                category = match[1]
                rows.append(inspect(path, dataset, int(category != "XXXX"),
                                    "ddl-" + path.parent.name, category, role="main_raw_fragment"))
        elif dataset in ("esc50", "urbansound8k"):
            metadata = directory / ("meta/selected.csv" if dataset == "esc50" else "metadata/selected.csv")
            with metadata.open(newline="") as handle:
                for item in csv.DictReader(handle):
                    if dataset == "esc50":
                        path = directory / "audio" / item["filename"]
                        source, category = item["src_file"], item["category"]
                    else:
                        path = directory / "audio" / f"fold{item['fold']}" / item["slice_file_name"]
                        source, category = item["fsID"], item["class"]
                    # A common namespace detects sources shared across both datasets.
                    rows.append(inspect(path, dataset, 0, "freesound-" + source, category,
                                        item["fold"], "background_pool_unassigned"))
        else:
            for path in sorted(directory.rglob("*.mat")):
                rows.append(inspect(path, dataset, "", "nasa-" + path.stem,
                                    path.stem.split("_")[0], role="external_test_only"))
        write_inventory(dataset, rows)


if __name__ == "__main__":
    main()
