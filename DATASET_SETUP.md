# Local dataset storage

All raw recordings, archives, metadata, and generated manifests live under
`data/`, which is excluded by the repository's `.gitignore`. Downloading or
indexing these datasets does not train a model.

| Dataset | Local folder | Selection | Role |
| --- | --- | --- | --- |
| [DDL](https://zenodo.org/records/6459183) | `data/raw/ddl/` | Real field-recording archive; synthetic archive excluded | Main drone / non-drone data |
| [ESC-50](https://github.com/karolpiczak/ESC-50) | `data/raw/esc50/` | 12 requested categories, 480 WAVs | Environmental noise / confusers |
| [UrbanSound8K](https://urbansounddataset.weebly.com/urbansound8k.html) | `data/raw/urbansound8k/` | 7 requested categories | Urban noise |
| [NASA Small UAS](https://data.nasa.gov/dataset/small-uas-flyover-acoustics-data) | `data/raw/nasa_external/` | Original flyover archive and description | External testing only |

ESC-50 category `chirping_birds` corresponds to the requested birds category.
The selected categories are helicopter, airplane, wind, rain, thunderstorm,
chirping_birds, crow, crickets, insects, siren, footsteps, and fireworks.

UrbanSound8K categories are engine_idling, siren, car_horn, gun_shot, street_music,
drilling, and jackhammer. Its publisher explicitly supports downloading through
soundata; the downloader uses the same official Zenodo archive URL and MD5 hash
listed by [soundata's loader](https://github.com/soundata/soundata/blob/main/soundata/datasets/urbansound8k.py).
Only the selected UrbanSound8K WAVs are extracted. Full upstream metadata and
attribution documents are retained alongside the selected audio.

## Download / resume

The project virtual environment supplies the `requests` dependency. From the
repository directory, these commands download data only:

```bash
.venv/bin/python scripts/download_datasets.py ddl --workers 12
.venv/bin/python scripts/download_datasets.py esc50 --workers 8
.venv/bin/python scripts/download_datasets.py urbansound8k --workers 8
.venv/bin/python scripts/download_datasets.py nasa_external --workers 4
```

Run at most one downloader per dataset at a time. The archive downloader records
completed byte ranges under `data/downloads/` and resumes them after interruption.
Sparse `.part` files have their final logical size from the beginning; use `du`
or the progress JSON, not `ls -lh`, to judge download progress. A
`download_complete.json` inside each raw dataset folder marks successful download
and extraction. Final DDL and UrbanSound8K archives are checked against the
publisher's MD5, and ZIP members are checked against their CRC during extraction.
ESC-50 downloads are pinned to a recorded Git revision and have a local SHA-256
inventory. The original large archives are retained under `data/downloads/`.

## Preserve recording boundaries before preparing training audio

DDL's real archive contains short multichannel WAV samples. The inspected sample
has 8 channels, a 96 kHz sample rate, and a 0.1-second duration. Its filename
identifies the drone type (`MINI`, `PRO4`, or `XXXX` for no drone), session, and
sample sequence. Keep source-session IDs together, and concatenate consecutive
samples within their recording session before creating 1- or 3-second windows.
Do not treat separate 0.1-second samples as independent full-length clips.

**The real archive's directory lists 62,103 WAV files with MINI or PRO4 names,
and no XXXX files.** Do not assume this archive supplies negative examples just
because its naming convention documents a no-drone code. Some entries are empty
WAV headers; the inventory identifies them. Reserve distinct background source
recordings from ESC-50/UrbanSound8K for the negative class before training.

Keep ESC-50's `src_file` and UrbanSound8K's `fsID` when grouping audio, along with
their published folds. Both datasets derive from Freesound, so a shared Freesound
ID across the two datasets refers to the same source and must remain in one split.
Do not use the same source as a labeled background example and as test-side
additive noise.

NASA supplies MATLAB `.mat` files, not training-ready WAV files. The archive also
contains MATLAB examples, and `Data_Description_20160203.pdf` describes the
`acoustics.incident_pascals` signal and `acoustics.utc_time` timestamps. Preserve
these originals; any later conversion should remain under an external-test-only
directory and never be added to the training/validation manifest.

The existing classical pipeline currently uses the originally requested 1-second
windows and SNRs clean/20/10/0/-10. The later dataset notes describe a separate
3-second, clean/20/10/5/0 experiment. Downloading the data does not alter either
experiment's preprocessing configuration or run training.

## Inspect downloaded files

```bash
.venv/bin/python scripts/index_datasets.py
```

This writes CSV inventories and JSON summaries to `data/inventories/` for datasets
with a completed download marker. WAV headers are checked for sample rate,
channels, duration, and empty data. Paths are relative to `data/`. These are raw
inventories, **not training manifests**: no splits or background-versus-noise
assignments are guessed, DDL fragments are not concatenated, and NASA MAT labels
remain unset until external-test preparation. NASA is always marked
`external_test_only`.
