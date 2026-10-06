# QSky classical audio baselines

Choose an individual model or compare all four on the same recordings and noise
mixtures. This implements the requested **1-second / 16 kHz** pipeline with test
SNRs **clean, 20, 10, 0, and -10 dB**. These settings supersede the older experiment
settings in `QubitSky_DATASETS.md` for this implementation.

| Model | Input | Classifier |
| --- | --- | --- |
| A | MFCC mean/std → StandardScaler → PCA → [0, π] | RBF SVM |
| B | Same features as A | Small MLP, one hidden layer of 4 units by default |
| C | Same features as A | Larger MLP, validation-selected (64, 32) or (128, 64) |
| D | 64-bin log-mel spectrogram, no PCA | Three convolution layers and a small dense head |

B is a configurable small-model reference: use `--small-width` to adjust its
capacity. Its trainable parameter count is `(k + 2) * width + 1` for binary output.
An exact quantum parameter budget has not been supplied, so no exact match is
claimed. C and D are stronger reference architectures, not guaranteed upper bounds
on accuracy. Parameter counts and SVM support-vector counts are recorded.

## Install

Python 3.10 or newer:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[test]"
```

A–C work with the base installation. For D or `--models all`, also install PyTorch.
The implementation runs the CNN on CPU:

```bash
pip install "torch>=2.2,<3" --index-url https://download.pytorch.org/whl/cpu
```

Alternatively, `pip install -e ".[cnn,test]"` uses the default PyTorch distribution.

## Describe your recordings

Use a CSV manifest; this avoids assuming a particular DADS/DDL directory layout.
No datasets are included or downloaded automatically. For binary drone detection,
label drone recordings **1** and background/confuser recordings **0**.

For the requested local dataset downloads, folder layout, resume commands, and
raw-format details, see [DATASET_SETUP.md](DATASET_SETUP.md). Local data is ignored
by Git and is not bundled with the source code.

For example, `data/clips.csv`:

```csv
path,label,group_id
raw/dads/drone/flight01.wav,1,dads-flight01
raw/dads/drone/flight02.wav,1,dads-flight02
raw/dads/background/field01.wav,0,dads-field01
raw/esc50/audio/example-confuser.wav,0,esc50-source123
```

Paths are relative to the CSV's directory, or absolute. Include enough independent
recordings of each label for all three splits; the four illustrative rows above
are not a complete dataset. Automatic splitting defaults to 60% train, 20% val,
20% test and stratifies recording groups by label.

`group_id` is optional when every file is an independent recording. If several
files come from the same original recording, flight, or recording session, give
them the same ID. Group IDs must be globally unique across datasets. For ESC-50,
use the metadata source-recording identifier (`src_file`) with an `esc50-` prefix
so excerpts of one source stay together. Select the relevant confuser categories
described in [the dataset notes](QubitSky_DATASETS.md).

To supply an existing split, add a `split` column containing `train`, `val`, or
`test` **for every row**. Both labels must occur in each signal split. Groups with
mixed labels require explicit splits. Predefined ESC-50 folds can be mapped into
these split names in your manifest.

Create a **separate** `data/noise.csv` for recordings used as additive noise:

```csv
path,group_id
raw/esc50/audio/noise-example1.wav,esc50-source456
raw/esc50/audio/noise-example2.wav,esc50-source789
```

Noise has no class-label requirement. Its source recordings are also split before
windowing; use enough recordings for automatic splitting, or supply explicit
`split` values. Only test noise is used for noisy testing, and only training noise
for training augmentation. An explicit test-only noise manifest is allowed when
training is clean. Reserve distinct ESC-50 sources for noise versus labeled
background examples: shared paths or group IDs between manifests are rejected.
Keep any NASA external-test recordings out of training and validation.

## Select a model or compare

Run A with four PCA components:

```bash
qsky-classical --manifest data/clips.csv --noise-manifest data/noise.csv \
  --models A --pca 4 --output runs/svm
```

Compare A, B, and C at all requested PCA dimensions:

```bash
qsky-classical --manifest data/clips.csv --noise-manifest data/noise.csv \
  --models A B C --pca 2 4 6 --output runs/tabular
```

Compare all four models (D runs once, independently of PCA):

```bash
qsky-classical --manifest data/clips.csv --noise-manifest data/noise.csv \
  --models all --pca 2 4 6 --output runs/comparison
```

Run just D, or run without noise data:

```bash
qsky-classical --manifest data/clips.csv --models D --snrs clean --output runs/cnn-clean
qsky-classical --manifest data/clips.csv --models B --snrs clean --output runs/mlp-clean
```

Optional training augmentation adds noisy copies alongside clean training windows:

```bash
qsky-classical --manifest data/clips.csv --noise-manifest data/noise.csv \
  --models all --train-snrs 20 10 0 -10 --small-width 4 --epochs 30 \
  --output runs/augmented
```

`--snrs clean 20 10 0 -10` is the default. `--seed`, `--val-size`, `--test-size`,
`--max-iter`, and `--batch-size` are also configurable. See `qsky-classical --help`.
`python -m qsky_classical` is an equivalent entry point. Use a new output directory
for each run; existing results are never overwritten by a new run.

## Processing and comparison protocol

1. Validate the manifest and split source-recording groups before reading windows.
2. Convert to mono at 16 kHz, cut non-overlapping 1-second windows, zero-pad the
   final partial window, and peak-normalize each window independently.
3. Mix an independently reserved background window at each requested SNR using
   RMS power. The same seeded noise assignment is used for each model and each
   SNR; only the noise gain changes with SNR. Mixing is in floating point, with
   no post-mix clipping or normalization. Silent noise windows are excluded;
   silent signal windows produce a clear error for noisy runs because SNR is
   undefined (they remain usable in clean-only runs).
4. Compute 13 MFCCs and their population mean/std over frames: 26 features. Both
   feature branches use 64 mel bands, a 512-sample FFT, and a 160-sample hop.
5. Fit StandardScaler, PCA (`k = 2, 4, 6`), and MinMaxScaler **only on training
   features**, including augmentation if requested. Transform validation/test
   with those fitted objects. Held-out values outside training extrema are
   clipped into [0, π]. The exact angle features are exported for quantum use.
6. Select SVM/MLP hyperparameters using **clean validation clip balanced accuracy**.
   SVM searches C ∈ {0.1, 1, 10} and gamma ∈ {scale, 0.1, 1}; each MLP searches
   alpha ∈ {0.0001, 0.01}. MLPs use L-BFGS without an internal random validation
   split. Model D standardizes log-mels using a training-only global mean/std and
   selects its best epoch on the same validation metric. Ties retain the first
   candidate/earliest epoch. Validation is never merged into training.
7. Evaluate every requested model/dimension on the identical test conditions.
   Clip predictions average window scores. SVM uses its decision score and a
   zero threshold; MLP/CNN use positive-class probability and a 0.5 threshold.
   Every requested PCA width is reported; test scores do not select a winner.

The implementation keeps audio/features in memory and is intended for manageable
research subsets. Larger datasets may need a streaming feature cache. Training
time includes candidate fitting and validation; prediction time covers window
scoring only, excluding audio preprocessing and feature extraction. A short
`--max-iter` may produce scikit-learn convergence warnings; increase it before
treating the result as a trained benchmark.

The preprocessing APIs are documented in the official
[librosa feature reference](https://librosa.org/doc/0.10.2/feature.html) and
[MinMaxScaler reference](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html).

## Outputs

Each output directory contains:

- `results.csv`: model, PCA width, SNR, evaluation level, accuracy, balanced
  accuracy, precision, recall, F1, ROC-AUC, confusion counts, timing, model size,
  validation score, and seed. Precision/recall/F1 refer to drone label 1.
- `predictions.csv`: individual window and clip scores/predictions.
- `splits.csv`, `noise_splits.csv`: exact recording assignments (noise file when used).
- `config.json`: arguments, audio settings, selection rule, and key package versions.
- `selection.json`: candidate validation scores and selected hyperparameters/epoch.
- `artifacts/model_A_k*.joblib`, `model_B_k*.joblib`, `model_C_k*.joblib`: fitted
  classifier, fitted preprocessing pipeline, and decision threshold.
- `artifacts/model_D.pt`: CNN state, training normalization, and batch size.
- `artifacts/preprocessor_k*.joblib` and `features_k*.npz`: fitted preprocessing
  and the exact train/val/test angle features used by A–C.

Feature archives contain `train_x`, `train_y`, `val_x`, `val_y`, `test_y`,
`test_x_clean`, `test_x_20`, etc. (only requested test conditions), plus clip IDs,
window IDs, and `train_snr_db`. Reuse these for a quantum kernel to preserve the
same feature representation and split. Angle exports exist when any of A–C runs.

## Try the complete flow without downloading data

```bash
python examples/make_demo.py
qsky-classical --manifest data/demo/clips.csv --noise-manifest data/demo/noise.csv \
  --models all --pca 2 4 6 --epochs 2 --output runs/demo
pytest -q
```

The generated tone/noise dataset is **only a functional smoke test**. Its accuracy
does not measure real drone detection. Install PyTorch for the complete test suite;
without it the tabular comparison is tested and the dedicated CNN test is skipped.
