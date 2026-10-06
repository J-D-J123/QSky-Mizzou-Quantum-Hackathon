"""Download QSky datasets without training. Archives resume in data/downloads.

Usage: .venv/bin/python scripts/download_datasets.py ddl --workers 12
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import threading
import time
import zipfile

import requests

ROOT = Path(__file__).resolve().parents[1] / "data"
ESC_CLASSES = {"helicopter", "airplane", "wind", "rain", "thunderstorm", "chirping_birds",
               "crow", "crickets", "insects", "siren", "footsteps", "fireworks"}
URBAN_CLASSES = {"engine_idling", "siren", "car_horn", "gun_shot", "street_music", "drilling", "jackhammer"}
ARCHIVES = {
    "ddl": ("https://zenodo.org/records/6459183/files/MLSP_2022_Real_Data.zip?download=1",
            "MLSP_2022_Real_Data.zip", 12628837686, "4a6d4da4e1c732550c1ccd8d29dd16f8"),
    "urbansound8k": ("https://zenodo.org/records/1203745/files/UrbanSound8K.tar.gz?download=1",
                     "UrbanSound8K.tar.gz", 6023741708, "9aa69802bbf37fb986f71ec1483a196e"),
    "nasa_external": ("https://data.nasa.gov/docs/datasets/rfk401li/small_uav_acoustics.zip",
                      "small_uav_acoustics.zip", 1693926297, None),
}


def get_bytes(url):
    for attempt in range(5):
        try:
            response = requests.get(url, timeout=(30, 90))
            response.raise_for_status()
            return response.content
        except requests.RequestException:
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def archive_download(name, workers):
    url, filename, size, checksum = ARCHIVES[name]
    directory = ROOT / "downloads"
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / filename
    partial = directory / (filename + ".part")
    journal = directory / (filename + ".progress.json")
    chunk_size = 32 * 1024 * 1024
    completed = set()
    if target.exists() and target.stat().st_size != size:
        if partial.exists():
            raise RuntimeError(f"Both incomplete final and partial files exist: {target}")
        target.rename(partial)
        # Adopt a sequential curl download from byte zero.
        completed = set(range(partial.stat().st_size // chunk_size))
    if not target.exists():
        if journal.exists():
            progress = json.loads(journal.read_text())
            if progress["size"] != size or progress["chunk_size"] != chunk_size or not partial.exists():
                raise RuntimeError(f"Incompatible download checkpoint: {journal}")
            completed = set(progress["completed"])
        elif partial.exists() and not completed:
            raise RuntimeError(f"Partial file without checkpoint: {partial}")
        fd = os.open(partial, os.O_CREAT | os.O_RDWR, 0o644)
        os.ftruncate(fd, size)
        lock = threading.Lock()

        def checkpoint():
            atomic_json(journal, {"url": url, "size": size, "chunk_size": chunk_size,
                                  "completed": sorted(completed)})

        checkpoint()

        def fetch(index):
            start = index * chunk_size
            end = min(size, start + chunk_size) - 1
            for attempt in range(8):
                try:
                    with requests.get(url, headers={"Range": f"bytes={start}-{end}",
                                                    "Accept-Encoding": "identity"},
                                      stream=True, timeout=(30, 90)) as response:
                        response.raise_for_status()
                        expected = f"bytes {start}-{end}/{size}"
                        if response.status_code != 206 or response.headers.get("Content-Range") != expected:
                            raise ValueError(f"Server did not honor requested range {expected}")
                        offset = start
                        for block in response.iter_content(1024 * 1024):
                            if offset + len(block) > end + 1:
                                raise ValueError("Server sent more bytes than requested")
                            view = memoryview(block)
                            while view:
                                written = os.pwrite(fd, view, offset)
                                offset += written
                                view = view[written:]
                        if offset != end + 1:
                            raise ValueError("Truncated range response")
                    with lock:
                        os.fsync(fd)
                        completed.add(index)
                        checkpoint()
                        downloaded = sum(min(chunk_size, size - i * chunk_size) for i in completed)
                        print(f"{name}: {downloaded / 1e9:.2f}/{size / 1e9:.2f} GB ({100 * downloaded / size:.1f}%)", flush=True)
                    return
                except (requests.RequestException, ValueError) as error:
                    if attempt == 7:
                        raise
                    print(f"{name}: retry chunk {index}: {type(error).__name__}", flush=True)
                    time.sleep(min(60, 2 ** attempt))

        try:
            pending = [i for i in range((size + chunk_size - 1) // chunk_size) if i not in completed]
            with ThreadPoolExecutor(max_workers=workers) as pool:
                for future in as_completed([pool.submit(fetch, i) for i in pending]):
                    future.result()
        finally:
            os.close(fd)
        partial.replace(target)
        journal.unlink()
    print(f"{name}: checking archive integrity...", flush=True)
    if checksum:
        digest = hashlib.md5()
        with target.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
        actual = digest.hexdigest()
        if actual != checksum:
            raise ValueError(f"Archive MD5 mismatch: {target}: {actual}")
    return target


def safe_path(root, name):
    parts = PurePosixPath(name).parts
    if PurePosixPath(name).is_absolute() or ".." in parts or "\\" in name:
        raise ValueError(f"Unsafe archive member: {name}")
    destination = root.joinpath(*parts)
    if not destination.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Archive path leaves destination: {name}")
    return destination


def save_stream(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    with temporary.open("wb") as handle:
        shutil.copyfileobj(source, handle, 1024 * 1024)
    temporary.replace(destination)


def extract_zip(archive, root):
    count = 0
    with zipfile.ZipFile(archive) as zf:
        for item in zf.infolist():
            if item.is_dir() or "__MACOSX" in PurePosixPath(item.filename).parts:
                continue
            destination = safe_path(root, item.filename)
            # Reading to EOF also verifies each member's ZIP CRC.
            with zf.open(item) as source:
                save_stream(source, destination)
            count += 1
            if count % 5000 == 0:
                print(f"{root.name}: extracted {count} files", flush=True)
    return count


def download_esc(workers):
    root = ROOT / "raw/esc50"
    for folder in (root / "audio", root / "meta"):
        folder.mkdir(parents=True, exist_ok=True)
    commit_path = root / "revision.txt"
    if commit_path.exists():
        revision = commit_path.read_text().strip()
    else:
        revision = json.loads(get_bytes("https://api.github.com/repos/karolpiczak/ESC-50/commits/master"))["sha"]
        commit_path.write_text(revision + "\n")
    base = f"https://raw.githubusercontent.com/karolpiczak/ESC-50/{revision}/"
    metadata = get_bytes(base + "meta/esc50.csv")
    (root / "meta/esc50.csv").write_bytes(metadata)
    for name in ("README.md", "LICENSE"):
        (root / name).write_bytes(get_bytes(base + name))
    rows = [r for r in csv.DictReader(io.StringIO(metadata.decode())) if r["category"] in ESC_CLASSES]
    hashes = {}

    def fetch(row):
        destination = root / "audio" / row["filename"]
        if not destination.exists():
            data = get_bytes(base + "audio/" + row["filename"])
            if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
                raise ValueError(f"Invalid WAV download: {destination}")
            save_stream(io.BytesIO(data), destination)
        return row["filename"], hashlib.sha256(destination.read_bytes()).hexdigest()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for index, result in enumerate(pool.map(fetch, rows), 1):
            hashes[result[0]] = result[1]
            if index % 40 == 0:
                print(f"esc50: {index}/{len(rows)} selected recordings", flush=True)
    with (root / "meta/selected.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    atomic_json(root / "sha256.json", hashes)
    return len(rows)


def extract_urban(archive, root):
    # Read metadata before choosing audio; preserve upstream source IDs and folds.
    with tarfile.open(archive, "r:gz") as tar:
        meta = tar.extractfile("UrbanSound8K/metadata/UrbanSound8K.csv").read()
    rows = list(csv.DictReader(io.StringIO(meta.decode())))
    selected = [r for r in rows if r["class"] in URBAN_CLASSES]
    selected_paths = {f"audio/fold{r['fold']}/{r['slice_file_name']}" for r in selected}
    count = 0
    with tarfile.open(archive, "r|gz") as tar:
        for item in tar:
            if not item.isfile():
                continue
            path = PurePosixPath(item.name)
            if not path.parts or path.parts[0] != "UrbanSound8K":
                continue
            relative = str(PurePosixPath(*path.parts[1:]))
            if path.name.startswith("."):
                continue
            if relative.startswith("audio/") and relative not in selected_paths:
                continue
            with tar.extractfile(item) as source:
                save_stream(source, safe_path(root, relative))
            if relative in selected_paths:
                count += 1
    if count != len(selected):
        raise ValueError(f"Expected {len(selected)} selected UrbanSound8K WAVs; found {count}")
    with (root / "metadata/selected.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selected[0]))
        writer.writeheader()
        writer.writerows(selected)
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=["ddl", "esc50", "urbansound8k", "nasa_external"])
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.workers <= 16:
        parser.error("workers must be between 1 and 16")
    root = ROOT / "raw" / args.dataset
    root.mkdir(parents=True, exist_ok=True)
    complete = root / "download_complete.json"
    if complete.exists():
        print(f"Already downloaded: {root}")
        return
    if args.dataset == "esc50":
        count = download_esc(args.workers)
        source = "https://github.com/karolpiczak/ESC-50"
    else:
        archive = archive_download(args.dataset, args.workers)
        print(f"{args.dataset}: extracting...", flush=True)
        count = extract_urban(archive, root) if args.dataset == "urbansound8k" else extract_zip(archive, root)
        source = ARCHIVES[args.dataset][0]
        if args.dataset == "nasa_external":
            (root / "Data_Description_20160203.pdf").write_bytes(get_bytes(
                "https://data.nasa.gov/docs/datasets/rfk401li/Data_Description_20160203.pdf"))
    atomic_json(complete, {"source": source, "files_extracted": count,
                           "role": "external_test_only" if args.dataset == "nasa_external" else args.dataset,
                           "downloaded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    print(f"DONE: {args.dataset}: {count} files stored in {root}", flush=True)


if __name__ == "__main__":
    main()
