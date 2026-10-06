"""Manifest validation, recording-level splitting, and audio preparation."""

import csv
from dataclasses import dataclass, replace
from pathlib import Path

import librosa
import numpy as np
from sklearn.model_selection import train_test_split

SR = 16_000
SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class Clip:
    path: Path
    label: int
    group_id: str
    split: str = ""


def read_manifest(path, *, noise=False):
    path = Path(path).resolve()
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"path"} if noise else {"path", "label"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{path}: required columns: {sorted(required)}")
        clips = []
        seen = set()
        for line, row in enumerate(reader, 2):
            raw_path = row["path"].strip()
            audio = (path.parent / raw_path).resolve()
            if not raw_path or not audio.is_file():
                raise ValueError(f"{path}:{line}: audio file does not exist: {audio}")
            if audio in seen:
                raise ValueError(f"Duplicate audio path: {audio}")
            seen.add(audio)
            label = -1 if noise else int(row["label"])
            if not noise and label not in (0, 1):
                raise ValueError("Labels must be 0 (background) or 1 (drone).")
            split = (row.get("split") or "").strip().lower()
            if split and split not in SPLITS:
                raise ValueError(f"Invalid split {split!r}; use train, val, or test.")
            group = (row.get("group_id") or "").strip() or str(audio)
            clips.append(Clip(audio, label, group, split))
    if not clips:
        raise ValueError(f"Empty manifest: {path}")
    return clips


def split_clips(clips, seed=42, val_size=0.2, test_size=0.2, *, noise=False):
    if not (0 < val_size < 1 and 0 < test_size < 1 and val_size + test_size < 1):
        raise ValueError("val-size and test-size must be positive and sum to less than 1.")
    explicit = [bool(c.split) for c in clips]
    if any(explicit) and not all(explicit):
        raise ValueError("Specify split for every row, or leave all splits blank.")
    if all(explicit):
        groups = {}
        for clip in clips:
            if groups.setdefault(clip.group_id, clip.split) != clip.split:
                raise ValueError(f"Recording group crosses splits: {clip.group_id}")
        result = clips
    else:
        labels = {}
        for clip in clips:
            if labels.setdefault(clip.group_id, clip.label) != clip.label:
                raise ValueError("Mixed-label groups require explicit split assignments.")
        groups = sorted(labels)
        stratify = None if noise else [labels[g] for g in groups]
        try:
            trainval, test = train_test_split(
                groups, test_size=test_size, random_state=seed, stratify=stratify
            )
            train, val = train_test_split(
                trainval, test_size=val_size / (1 - test_size), random_state=seed,
                stratify=None if noise else [labels[g] for g in trainval],
            )
        except ValueError as error:
            raise ValueError(
                "Not enough independent recordings for stratified train/val/test splits. "
                "Add recordings or supply explicit split assignments. " + str(error)
            ) from error
        assignments = {g: s for s, subset in zip(SPLITS, (train, val, test)) for g in subset}
        result = [replace(c, split=assignments[c.group_id]) for c in clips]
    if not noise:
        for split in SPLITS:
            if {c.label for c in result if c.split == split} != {0, 1}:
                raise ValueError(f"{split} must contain both labels 0 and 1.")
    return result


def write_manifest(path, clips):
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "label", "group_id", "split"])
        writer.writeheader()
        for c in clips:
            writer.writerow(dict(path=str(c.path), label=c.label, group_id=c.group_id, split=c.split))


def load_windows(path):
    audio, _ = librosa.load(path, sr=SR, mono=True)
    if not len(audio) or not np.isfinite(audio).all():
        raise ValueError(f"Empty or non-finite audio: {path}")
    # Keep every sample, including a zero-padded final partial window.
    audio = np.pad(audio, (0, (-len(audio)) % SR)).reshape(-1, SR)
    peaks = np.max(np.abs(audio), axis=1, keepdims=True)
    return (audio / np.maximum(peaks, 1e-12)).astype(np.float32)


@dataclass
class Windows:
    audio: np.ndarray
    labels: np.ndarray
    clip_ids: np.ndarray
    window_ids: np.ndarray


def make_windows(clips, split):
    waves, labels, ids, positions = [], [], [], []
    for clip in clips:
        if clip.split != split:
            continue
        windows = load_windows(clip.path)
        waves.extend(windows)
        labels.extend([clip.label] * len(windows))
        ids.extend([str(clip.path)] * len(windows))
        positions.extend(range(len(windows)))
    if not waves:
        raise ValueError(f"No audio windows for {split}")
    return Windows(np.stack(waves), np.array(labels), np.array(ids), np.array(positions))


def mix_at_snr(signal, noise, snr):
    signal_rms = float(np.sqrt(np.mean(np.square(signal, dtype=np.float64))))
    noise_rms = float(np.sqrt(np.mean(np.square(noise, dtype=np.float64))))
    if signal_rms < 1e-12 or noise_rms < 1e-12:
        raise ValueError("SNR is undefined for a silent signal or noise window.")
    # Do not clip or normalize after mixing: that would alter the experiment.
    return (signal + noise * (signal_rms / noise_rms) * 10 ** (-snr / 20)).astype(np.float32)


def corrupt(windows, noise_pool, snr, seed):
    if snr == "clean":
        return windows
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(noise_pool), size=len(windows.audio))
    audio = np.stack([
        mix_at_snr(wave, noise_pool[index], float(snr))
        for wave, index in zip(windows.audio, indices)
    ])
    return replace(windows, audio=audio)


def features(audio, *, mel=False):
    result = []
    for wave in audio:
        power = librosa.feature.melspectrogram(
            y=wave, sr=SR, n_fft=512, hop_length=160, n_mels=64, power=2.0
        )
        logmel = librosa.power_to_db(power, ref=1.0)
        if mel:
            result.append(logmel)
        else:
            mfcc = librosa.feature.mfcc(S=logmel, n_mfcc=13)
            result.append(np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)]))
    return np.asarray(result, dtype=np.float32)
