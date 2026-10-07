# Comparing the classical models with the quantum experiments

This document describes how models A–D are compared with Dhanya's quantum-kernel
experiments (`dhanya` branch, commit `80e0046`), which of her files are needed, and the
commands to run once they arrive.

**Status: nothing has been trained or benchmarked under this protocol.** The code and
its non-training tests are in place. Every command in "Running it" is still to be run.

## The two pipelines

| | Classical default (`qsky-classical`) | Dhanya's quantum experiments |
| --- | --- | --- |
| Audio | 16 kHz mono, 1 s windows | 16 kHz mono, 3 s segments |
| Window normalization | Per-window peak | Per-segment peak |
| Features | 13 MFCC mean + std (26) | 9 candidates: MFCC 1–3 means, spectral centroid, bandwidth, rolloff, flatness, zero-crossing rate, RMS |
| Reduction | StandardScaler → PCA (2/4/6) → [0, π] | Top 4/5/6 by mutual information + ANOVA rank → StandardScaler; no angle scaling |
| Noise | Separate noise recordings, split-local | A class-0 recording of the same split, one donor per segment across SNRs |
| Mixing | No clipping or normalization after mixing | Whole mixture attenuated to 0.99 peak if it would clip |
| Test SNRs | clean, 20, 10, 0, −10 dB | clean, 20, 10, 5, 0 dB |
| Evaluation unit | Window, and clip (mean of window scores) | Recording (mean of segment **features**, then one prediction) |
| Selection metric | Validation balanced accuracy | Validation F1, then balanced accuracy, then drone recall |

`--window-seconds 3 --snr-preset shared` moves the classical pipeline to her window
length and SNR list. `--snr-preset legacy` (the default) keeps the original list.

## Feature modes

**A–C, MFCC mode** (`qsky-classical`): the existing MFCC → StandardScaler → PCA →
angle-scaling pipeline. It shares a protocol with the quantum experiments, not inputs.

**A–C, matched mode** (`qsky-matched`): her selected features, read from her own tables.
Matching the number of features is not enough, so the importer checks that the values,
feature order, preprocessing, and sample IDs all agree:

- the scaled table lists the same `sample_id`s in the same order as the raw table;
- rebuilding StandardScaler from all her training rows reproduces her scaled values;
- the train/validation/test recording lists hash to the checksums published in her
  `ideal_simulator_results.csv`.

**D** always uses 64-bin log-mel spectrograms of the same recordings. It sees far more
of the signal than 4–6 numbers, so every output labels it a *richer-input classical
reference*. It is not a feature-matched comparison.

## Two protocols, kept apart

Her saved feature ranking used all **301** labeled training recordings, and her scaler
was fit on every training row of all those recordings at every SNR. The 24-, 50-, and
70-recording subsets were chosen afterwards. A model trained on 24 recordings has
therefore already benefited from 301 labels. Noise donors for noisy training rows also
come from the whole training split.

| | `--protocol reproduce` | `--protocol strict` |
| --- | --- | --- |
| Label in results | `reproduction_dhanya_v1` | `strict_label_budget_v1` |
| Feature ranking | Hers, from 301 recordings | Inside each actual training subset |
| Scaler | Hers, from all training rows | Inside each actual training subset |
| Noisy training rows with a donor outside the subset | Kept, as she ran it | Dropped by default (`--donor-policy`) |
| Test segments | Her rule: recordings present at every SNR | The same segments under every SNR |
| Comparable with her published quantum scores | Yes, if the ID checksums verify | **No** |

A strict classical result must not be set beside the published quantum scores. The
quantum models have to be rerun on the strict inputs first. Each `qsky-matched` run
writes those inputs to `artifacts/inputs_seed*_n*_k*.npz` (with recording IDs and feature
order) for that purpose, and each result row carries `comparable_to_published_quantum`.

Seed 42 uses her saved training subsets. Any other seed draws a new balanced subset from
her training split, which is no longer her protocol and is marked not comparable.

`bundle.json` records, per training size, how many recordings the ranking and scaler
saw and which noise-donor recordings lie outside the subset.

## Evaluation units

These are different procedures and are never reported under one name:

| Name in `results.csv` | Procedure | Used by |
| --- | --- | --- |
| `feature_mean_per_recording` | Average the features of a recording's windows, classify once | A–C matched mode; A–C with `--aggregation feature-mean` |
| `score_mean_per_recording` | Classify every window, average the scores per recording | D; A–C with `--levels recording` |
| `score_mean_per_clip` | Classify every window, average the scores per file | Existing clip level |

D has no per-recording feature vector, so its recording-level prediction is always the
mean window probability. With one 3 s segment per recording the two procedures coincide;
with several segments they do not.

## Silent windows

SNR is undefined for a silent target. Policy (`--silent-policy exclude`, the default):
a window whose peak amplitude is at or below `--silence-peak` (10⁻¹²) is removed from
every split before any model runs and listed in `exclusions.csv`. The exclusion set is
independent of the model and of the SNR, so all models and all conditions, clean
included, are scored on identical examples. A test clip with no copy in a
`--premixed-test` condition is excluded the same way. `--silent-policy error` keeps the
earlier behavior of stopping the run.

Her pipeline differs: a silent segment keeps its clean row and gets no noisy rows, and a
recording is tested if it appears at every SNR. A recording with one silent segment is
then averaged over more segments in the clean condition than in the noisy ones.
`reproduce` keeps that rule; `strict` scores the same segments everywhere.

## Model selection

Hyperparameters and checkpoints are chosen on the validation split only. Thresholds are
fixed (SVM decision score 0, probabilities 0.5). No test score selects anything, and the
published test F1 must not be used to tune further.

What her code does, so nothing is mislabeled as untuned:

- Segment-level SVM: grid over C ∈ {0.1, 1, 10} and gamma ∈ {scale, auto}, selected on
  the fixed validation split.
- MLP: hidden layers (16, 8), early stopping on the same validation split. On 4 inputs
  it has 225 parameters.
- Recording-level comparison with the quantum kernel: the SVM uses C = 0.1,
  gamma = scale; the MLP is refit with validation early stopping; QSVC uses a fixed
  C = 0.1; the trainable kernel optimizes one parameter per qubit on training data.

Model B has `(k + 2) × width + 1` parameters: 25 at `k = 4` with the default width of 4.
It is the smallest one-hidden-layer reference, chosen for low capacity. It is **not**
parameter-matched to the quantum model, whose trainable kernel has 4 parameters at
4 qubits; no MLP on 4 inputs can be that small.

`--train-recordings` and `--seeds` run training-size and multi-seed sweeps. Within a
seed and size, every model trains on the same recordings, and smaller subsets are
contained in larger ones. `--seed` fixes the split and the noise, so the test set is the
same for every run.

## What is needed from Dhanya

Her branch contains code, the selected-feature list, result tables, one scaler, and one
SVM. It contains no recording IDs, feature tables, or audio. `qsky-dhanya inventory`
prints this list against any copy of her project:

| Needed for | Files (relative to her project root) |
| --- | --- |
| A–C, both protocols | `results/recording_split.csv` |
| | `results/small_data_subsets/train_subset_25.csv`, `_50.csv`, `_100.csv` (24/50/70 recordings) |
| | `data/features/train_features.csv`, `validation_features.csv`, `test_features.csv` |
| A–C reproduction | `data/features/{train,validation,test}_features_{4,5,6}.csv` |
| Verifying the exact 24/80/98 comparison | `results/quantum/stage5_sample_manifest.json` |
| D | `data/processed/{train,validation,test}/**/*.flac` |
| Quantum rerun on strict inputs | `models/quantum_ideal/trainable_params_q4_n24_snr_clean_seed42.json`, and her agreement to rerun |
| Reproducibility | Output of `pip freeze` from her environment |
| Context, optional | `data/master_metadata.csv`, `results/feature_selection.csv`, `results/svm_results.csv`, `results/mlp_results.csv`, `models/scaler_4.pkl`, `models/scaler_5.pkl` |

Her noisy audio was not saved by default. It is not needed: with the clean segments and
the donor IDs in the feature tables, `--materialize-noisy` regenerates each mixture and
accepts it only if it reproduces her feature row.

Published checksums the import must match (4 features): training, 24 recordings
`c72a008d…4b5281`; validation `08c970aa…6a31ec`; test `cfc4bc8c…7524cf`.

## Running it

Install once, then check the environment. The full suite fits small models; the first
command does not.

```bash
pip install -e ".[test]"            # registers qsky-dhanya and qsky-matched
pytest -q -m "not fits"             # no training
pytest -q                           # trains tiny models; run on the training machine
```

Import her assets. This validates counts (24 / 80 / 98), labels, group separation, and
noise provenance, and refuses to continue if anything is missing or inconsistent.

```bash
qsky-dhanya inventory --root /path/to/Qsky
qsky-dhanya import --root /path/to/Qsky --output data/dhanya_bundle \
  --feature-counts 4 5 6 --train-sizes 24 50 70 --verify-audio 20 --materialize-noisy
```

A–C on matched features. Add `--dry-run` first to build the inputs and list the fits.

```bash
qsky-matched --bundle data/dhanya_bundle --protocol reproduce --models all \
  --feature-counts 4 --train-recordings 24 --seeds 42 --output runs/matched-reproduce

qsky-matched --bundle data/dhanya_bundle --protocol strict --models all \
  --feature-counts 4 5 6 --train-recordings 24 50 70 --seeds 42 1 2 3 4 \
  --output runs/matched-strict
```

D on her audio and her exact noisy mixtures, one run per training size:

```bash
qsky-classical --manifest data/dhanya_bundle/manifest_n24.csv --models D \
  --window-seconds 3 --snr-preset shared --levels window recording \
  --premixed-test 20=data/dhanya_bundle/premixed_test_20.csv \
                  10=data/dhanya_bundle/premixed_test_10.csv \
                  5=data/dhanya_bundle/premixed_test_5.csv \
                  0=data/dhanya_bundle/premixed_test_0.csv \
  --seeds 42 1 2 3 4 --device cuda --output runs/d-n24
```

All four models on our own recordings under the shared protocol:

```bash
qsky-classical --manifest data/clips.csv --noise-manifest data/noise.csv --models all \
  --window-seconds 3 --snr-preset shared --levels window clip recording \
  --train-recordings 24 50 70 --seeds 1 2 3 4 5 --device cuda --output runs/shared
```

`--device cuda` fails immediately if no usable GPU is found. `--device auto` falls back
to CPU with a warning, which is not wanted for a benchmark.

## Artifacts and the scikit-learn versions

This package requires scikit-learn below 1.8. Her saved scaler and SVM were written by
1.9.1. scikit-learn does not support loading pickles across versions, and a mismatch can
give wrong numbers without an error. The resolution:

1. No pickle crosses between the two environments. Her artifacts are checksummed and
   their version is read from the file bytes; they are never unpickled here.
2. Her preprocessing is taken from her CSV tables and verified numerically.
3. Every artifact saved here has a `.meta.json` with its checksum and package versions.
   `load_bundle` refuses a pickle from another scikit-learn minor release, or one with no
   recorded version, unless `allow_version_mismatch=True` is passed.
4. Every A–C model is also written as `*.portable.json`: scaler, PCA, and classifier as
   plain numbers, with feature order, threshold, aggregation rule, and audio settings.
   `PortablePredictor` runs it with NumPy alone, in any environment.

The version pin itself is unchanged. Whether to widen it is listed below.

## Decisions still open

1. **Rerun the quantum models under the strict protocol.** Without that there is no
   fair small-data comparison, only a reproduction of the existing one.
2. **Which comparison is the headline:** the reproduction, or the strict rerun.
3. **Frontend integration:** read `*.portable.json` in her app (no version coupling), or
   move this package to scikit-learn 1.9 after running the full test suite there.
4. **Selection metric for the shared runs:** `qsky-matched` defaults to her F1 rule;
   `qsky-classical` defaults to balanced accuracy. Pick one for the final table.
5. **Hyperparameter grids:** A searches gamma ∈ {scale, 0.1, 1}; hers searched
   {scale, auto}. The grids are recorded but not aligned.
6. **Noise for our own recordings** comes from separate unlabeled noise recordings. In a
   strict label budget that is extra data, though not extra labels.
