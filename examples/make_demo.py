"""Create small synthetic WAVs to exercise the pipeline, not benchmark accuracy."""

import csv
from pathlib import Path

import numpy as np
import soundfile as sf


def main():
    root = Path("data/demo")
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    t = np.arange(32000) / 16000
    for noise in (False, True):
        manifest = root / ("noise.csv" if noise else "clips.csv")
        if manifest.exists():
            raise SystemExit(f"Demo already exists: {manifest}")
        with manifest.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["path", "label", "group_id"])
            for label in (0, 1):
                for index in range(12):
                    name = f"{'noise' if noise else 'signal'}-{label}-{index}.wav"
                    wave = 0.05 * rng.normal(size=len(t))
                    if not noise:
                        frequency = (180 if label else 1200) + rng.uniform(-30, 30)
                        wave += 0.3 * np.sin(2 * np.pi * frequency * t)
                    sf.write(root / name, wave, 16000)
                    writer.writerow([name, label, name])
        print(manifest)


if __name__ == "__main__":
    main()
