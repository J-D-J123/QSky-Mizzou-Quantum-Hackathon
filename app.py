from __future__ import annotations

import base64
import hashlib
import json
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import librosa
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from extract_features import FEATURE_COLUMNS, extract_features, standardize_clip


RESULTS_DIR = PROJECT_ROOT / "results"
QUANTUM_DIR = RESULTS_DIR / "quantum"
MODELS_DIR = PROJECT_ROOT / "models"
ASSETS_DIR = PROJECT_ROOT / "assets"


@dataclass
class ModelBundle:
    label: str
    model_path: Path
    scaler_path: Path
    feature_names: list[str]
    feature_count: int
    validation_f1: float


def _inject_style(background_image: Path) -> None:
    if background_image.is_file():
        image_data = base64.b64encode(background_image.read_bytes()).decode("ascii")
        hero_background = (
            "background-image: linear-gradient(110deg, rgba(4, 18, 49, 0.76), rgba(24, 77, 150, 0.48) 48%, rgba(87, 57, 158, 0.42)), "
            "linear-gradient(0deg, rgba(5, 21, 45, 0.30), rgba(3, 14, 35, 0.20)), "
            f"url('data:image/jpeg;base64,{image_data}');"
        )
    else:
        hero_background = "background-image: linear-gradient(135deg, #0a1620, #112a35 55%, #1a3e47);"
    st.markdown(
        f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=IBM+Plex+Mono:wght@400;600&display=swap');

:root {{
    --ink: #102342;
    --muted: #526982;
    --surface: #f4f8ff;
    --navy: #071b3d;
    --blue: #2877d4;
    --cyan: #38c9e8;
    --teal: #119b91;
    --violet: #7657d6;
    --purple: #a18af0;
    --orange: #ffad42;
    --accent: #ffad42;
    --accent-alt: #14a89d;
    --ring: rgba(63, 156, 232, 0.24);
}}

html, body, [class*="stApp"] {{
  font-family: 'Space Grotesk', sans-serif;
  color: var(--ink);
    background: linear-gradient(150deg, #f5fbff 0%, #f1f4ff 58%, #f7f4ff 100%);
}}

section[data-testid="stMain"] .block-container {{
    max-width: none;
    padding: 4rem 0 2rem 0;
}}

section[data-testid="stSidebar"], button[data-testid="stExpandSidebarButton"] {{
    display: none;
}}

[data-testid="stSidebar"] {{
  background: linear-gradient(180deg, #112531, #0b1a24);
}}

[data-testid="stSidebar"] * {{
  color: #eff7fb;
}}

.qk-hero {{
  position: relative;
    display: flex;
    align-items: flex-start;
    min-height: 42vh;
    width: 100%;
    border-radius: 0;
  {hero_background}
  background-size: cover;
  background-position: center;
    padding: clamp(2.4rem, 7vh, 4.4rem) clamp(1.5rem, 8vw, 8rem) 2rem;
  color: #f6fbff;
  overflow: hidden;
    box-sizing: border-box;
}}

.qk-hero-content {{
    position: relative;
    z-index: 4;
    width: min(740px, 78%);
    margin-top: 1vh;
    text-shadow: 0 2px 18px rgba(0, 0, 0, 0.38);
}}

.qk-hero::after {{
    content: "";
  position: absolute;
    inset: 0;
    z-index: 1;
    pointer-events: none;
    background: linear-gradient(180deg, rgba(3, 11, 20, 0.10), transparent 52%, rgba(3, 11, 20, 0.18));
}}

.qk-hero-title {{
    margin: 0.2rem 0 0.2rem;
    max-width: 12ch;
    font-size: clamp(2.5rem, 5vw, 4.3rem);
    line-height: 1;
    letter-spacing: 0;
}}

.qk-hero-copy {{
    max-width: 58ch;
    color: rgba(245, 251, 255, 0.94);
    font-size: 1rem;
    line-height: 1.5;
    margin: 0;
}}

.qk-hero-cta {{
    display: inline-block;
    margin-top: 0.9rem;
    padding: 0.66rem 1rem;
    border-radius: 8px;
    background: linear-gradient(110deg, #4bd3ed, #9f8af1 76%, #ffb454);
    color: #071b3d !important;
    font-weight: 700;
    text-decoration: none !important;
    text-shadow: none;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.20);
}}

.qk-radar {{
    position: absolute;
    z-index: 2;
    width: min(28vw, 330px);
    aspect-ratio: 1;
    right: 8%;
    top: 16%;
    border: 1px solid rgba(219, 249, 255, 0.34);
    border-radius: 50%;
    opacity: 0.5;
    pointer-events: none;
}}

.qk-radar-sweep {{
    position: absolute;
    inset: 0;
    border-radius: 50%;
    background: conic-gradient(from 0deg, transparent 0deg 300deg, rgba(115, 238, 226, 0.30) 350deg, transparent 360deg);
    animation: radar-spin 18s linear infinite;
}}

.qk-bird {{
    position: absolute;
    z-index: 3;
    width: 30px;
    height: 12px;
    color: rgba(9, 25, 36, 0.78);
    pointer-events: none;
    animation: bird-glide 24s linear infinite;
}}

.qk-bird::before, .qk-bird::after {{
    content: "";
    position: absolute;
    top: 0;
    width: 12px;
    height: 7px;
    border-top: 2px solid currentColor;
    border-radius: 50%;
}}

.qk-bird::before {{ right: 8px; transform: rotate(18deg); }}
.qk-bird::after {{ left: 8px; transform: rotate(-18deg); }}
.qk-bird-one {{ top: 36%; left: -5%; }}

.qk-drone {{
    position: absolute;
    z-index: 3;
    right: 21%;
    top: 39%;
    width: 44px;
    height: 28px;
    opacity: 0.72;
    pointer-events: none;
    animation: drone-hover 8s ease-in-out infinite;
}}

.qk-drone-body {{
    position: absolute;
    left: 15px;
    top: 10px;
    width: 15px;
    height: 8px;
    border-radius: 4px;
    background: #111e27;
    box-shadow: 0 1px 3px rgba(255,255,255,0.6);
}}

.qk-drone-body::before, .qk-drone-body::after {{
    content: "";
    position: absolute;
    left: -9px;
    top: 3px;
    width: 33px;
    height: 2px;
    background: #172a35;
    transform: rotate(25deg);
}}

.qk-drone-body::after {{ transform: rotate(-25deg); }}

.qk-prop {{
    position: absolute;
    top: 3px;
    width: 12px;
    height: 3px;
    border-radius: 50%;
    background: rgba(12, 31, 43, 0.85);
    opacity: 0.72;
}}

.qk-prop-left {{ left: 0; }}
.qk-prop-right {{ right: 0; }}

.qk-drone-beam {{
    position: absolute;
    top: 21px;
    left: 21px;
    width: 1px;
    height: 42px;
    background: linear-gradient(rgba(149, 255, 229, 0.56), transparent);
}}

@keyframes radar-spin {{ to {{ transform: rotate(360deg); }} }}
@keyframes bird-glide {{ to {{ transform: translateX(120vw) translateY(-18px); }} }}
@keyframes drone-hover {{ 50% {{ transform: translateY(-7px) translateX(5px); }} }}
}}

.qk-kicker {{
  text-transform: uppercase;
  font-family: 'IBM Plex Mono', monospace;
  letter-spacing: 0.14em;
  font-size: 0.76rem;
  color: rgba(247, 253, 255, 0.90);
}}

.qk-title {{
  margin: 0.35rem 0 0.35rem 0;
  font-size: clamp(1.7rem, 4vw, 2.6rem);
  line-height: 1.1;
  letter-spacing: -0.03em;
}}

.qk-sub {{
  max-width: 70ch;
  color: rgba(241, 249, 255, 0.92);
  margin-bottom: 0;
}}

.qk-card {{
    background: linear-gradient(145deg, rgba(255,255,255,0.92), rgba(242,248,255,0.84));
    border: 1px solid rgba(133, 183, 222, 0.46);
  border-radius: 16px;
  padding: 1rem 1rem 0.9rem 1rem;
  box-shadow: 0 10px 28px rgba(16, 34, 43, 0.08);
}}

.qk-result {{
        border: 1px solid rgba(48, 178, 185, 0.28);
        border-left: 7px solid var(--accent-alt);
        background: linear-gradient(110deg, rgba(211, 249, 247, 0.98), rgba(221, 241, 255, 0.94));
    padding: 1.25rem 1.4rem;
    margin: 1rem 0;
    color: #102833;
        box-shadow: 0 12px 32px rgba(25, 88, 129, 0.10);
}}

.qk-result.drone {{ border-color: #ed9a38; border-left-color: #26c5e3; background: linear-gradient(110deg, rgba(204, 247, 255, 0.98), rgba(225, 226, 255, 0.96) 68%, rgba(255, 226, 186, 0.92)); }}
.qk-result.no-drone {{ border-left-color: #119b91; background: linear-gradient(110deg, rgba(207, 249, 239, 0.98), rgba(212, 236, 255, 0.96)); }}
.qk-result-title {{ font-size: clamp(1.8rem, 5vw, 3rem); line-height: 1.05; font-weight: 700; }}

.st-key-detect-card {{
    padding: clamp(1.1rem, 3vw, 2rem);
    border: 1px solid rgba(139, 182, 235, 0.58);
    border-radius: 18px;
    background: linear-gradient(135deg, rgba(10, 35, 79, 0.97), rgba(29, 91, 156, 0.92) 52%, rgba(106, 78, 178, 0.91));
    color: #f4fbff;
    box-shadow: 0 18px 48px rgba(21, 58, 111, 0.18);
}}

.st-key-detect-card h2, .st-key-detect-card h3, .st-key-detect-card p, .st-key-detect-card label {{ color: #f4fbff !important; }}
.st-key-detect-card [data-testid="stCaptionContainer"] {{ color: #cfdef4 !important; }}
.st-key-detect-card [data-testid="stFileUploader"] {{ background: rgba(239, 249, 255, 0.94); border: 1px dashed #5ecfea; border-radius: 12px; padding: 0.55rem; }}
.st-key-detect-card [data-testid="stFileUploader"] * {{ color: #153450 !important; }}
.st-key-detect-card audio {{ width: 100%; }}

.qk-stat {{
  font-size: 1.5rem;
  font-weight: 700;
  color: #102833;
}}

.qk-muted {{
  color: var(--muted);
  font-size: 0.93rem;
}}

.qk-chip {{
  display: inline-block;
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.76rem;
  border-radius: 999px;
  padding: 0.22rem 0.6rem;
  background: #e6f8fb;
  color: #10566f;
  border: 1px solid #bcecf2;
  margin-right: 0.45rem;
  margin-bottom: 0.35rem;
}}

.qk-alert {{
  border-left: 4px solid var(--accent);
  background: #fff4ea;
  border-radius: 10px;
  padding: 0.75rem 0.9rem;
  color: #6b3c17;
}}

.stTabs [data-baseweb="tab-list"] {{
  gap: 0.4rem;
}}

.stTabs [data-baseweb="tab"] {{
  border-radius: 999px;
  padding: 0.35rem 0.9rem;
}}

.stButton button, .stDownloadButton button {{
  border-radius: 10px;
  border: 1px solid #d6dde0;
  background: #ffffff;
}}

.stButton button:hover, .stDownloadButton button:hover {{
  border-color: #ffb487;
  box-shadow: 0 0 0 0.2rem var(--ring);
}}

.stButton button[kind="primary"] {{
    border: 0;
    color: #071b3d;
    font-weight: 700;
    background: linear-gradient(105deg, #52d6ed, #8e89ed 72%, #ffb454);
    box-shadow: 0 8px 22px rgba(55, 148, 206, 0.25);
}}

@keyframes fade-slide {{
  from {{ opacity: 0; transform: translateY(8px); }}
  to {{ opacity: 1; transform: translateY(0); }}
}}

@media (max-width: 900px) {{
    .qk-hero {{ min-height: 37vh; padding: 2.8rem 1.25rem 1.6rem; background-position: center; }}
    .qk-hero-content {{ width: 100%; margin-top: 0; }}
    .qk-hero-title {{ max-width: 11ch; font-size: clamp(2.5rem, 10vw, 4rem); }}
    .qk-radar {{ width: 34vw; right: 5%; top: 48%; }}
    .qk-drone {{ right: 16%; top: 60%; }}
}}

@media (prefers-reduced-motion: reduce) {{
    .qk-bird, .qk-radar-sweep, .qk-drone {{ animation: none !important; }}
    * {{ transition: none !important; }}
}}
</style>
        """,
        unsafe_allow_html=True,
    )


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_csv(path)


def _safe_read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


@st.cache_data(show_spinner=False, ttl=10)
def load_results() -> dict[str, Any]:
    return {
        "final": _safe_read_csv(QUANTUM_DIR / "final_quantum_comparison.csv"),
        "ideal": _safe_read_csv(QUANTUM_DIR / "ideal_simulator_results.csv"),
        "noisy": _safe_read_csv(QUANTUM_DIR / "noisy_simulator_results.csv"),
        "svm": _safe_read_csv(RESULTS_DIR / "svm_results.csv"),
        "mlp": _safe_read_csv(RESULTS_DIR / "mlp_results.csv"),
        "qpu_jobs": _safe_read_json(QUANTUM_DIR / "qpu_jobs.json"),
        "selected_features": _safe_read_json(RESULTS_DIR / "selected_features.json"),
        "best_svm": _safe_read_json(RESULTS_DIR / "best_svm_configuration.json"),
        "best_mlp": _safe_read_json(RESULTS_DIR / "best_mlp_configuration.json"),
        "stage7": _safe_read_json(MODELS_DIR / "quantum" / "stage7_frozen_config.json"),
        "qpu_results": _safe_read_csv(QUANTUM_DIR / "qpu_results.csv"),
    }


@st.cache_resource(show_spinner=False)
def _load_model(model_path: str) -> Any:
    return joblib.load(model_path)


@st.cache_resource(show_spinner=False)
def _load_scaler(scaler_path: str) -> Any:
    return joblib.load(scaler_path)


def _build_bundle(config: dict[str, Any], selected_features: dict[str, Any], label: str) -> ModelBundle | None:
    best = config.get("best_external_model", {})
    model_path = best.get("model_path")
    feature_count = int(best.get("feature_count", 0) or 0)
    if not model_path or feature_count <= 0:
        return None
    mapping = selected_features.get("qubit_mappings", {}).get(str(feature_count), {})
    names = mapping.get("features", [])
    scaler_path = MODELS_DIR / f"scaler_{feature_count}.pkl"
    if not names:
        return None
    model_candidate = Path(model_path)
    if not model_candidate.is_file():
        candidates = [
            MODELS_DIR / model_candidate.name,
            MODELS_DIR / "svm" / model_candidate.name,
            MODELS_DIR / "mlp" / model_candidate.name,
        ]
        model_candidate = next((candidate for candidate in candidates if candidate.is_file()), model_candidate)
    if not model_candidate.is_file() or not scaler_path.is_file():
        return None
    return ModelBundle(
        label=label,
        model_path=model_candidate,
        scaler_path=scaler_path,
        feature_names=list(names),
        feature_count=feature_count,
        validation_f1=float(best.get("validation_f1", 0.0) or 0.0),
    )


def demo_audio_samples() -> dict[str, Path]:
    """Find one clean held-out drone and background example for UI demonstration only."""
    split_path = RESULTS_DIR / "recording_split.csv"
    test_ids: set[str] = set()
    if split_path.is_file():
        split_table = pd.read_csv(split_path, dtype={"recording_id": str, "split": str})
        test_ids = set(split_table.loc[split_table["split"].eq("test"), "recording_id"])
    examples: dict[str, Path] = {}
    for label, folder in (
        ("Drone Sample", PROJECT_ROOT / "data/processed/test/svanstrom/drone"),
        ("Environmental / No-Drone Sample", PROJECT_ROOT / "data/processed/test/svanstrom/background"),
    ):
        for audio_path in sorted(folder.glob("*__clean__seg0000.flac")):
            recording_id = audio_path.name.split("__", maxsplit=1)[0]
            if recording_id in test_ids:
                examples[label] = audio_path
                break
    if len(examples) < 2:
        manifest_path = ASSETS_DIR / "demo_audio" / "demo_samples.json"
        manifest = _safe_read_json(manifest_path)
        for label, sample in manifest.get("samples", {}).items():
            if label in examples or sample.get("split") != "test":
                continue
            candidate = ASSETS_DIR / "demo_audio" / str(sample.get("file", ""))
            if candidate.is_file():
                examples[label] = candidate
    return examples


def predict_audio(audio_source: Any, bundle: ModelBundle | None = None) -> dict[str, Any]:
    """Run the same local preprocessing and saved-model inference for uploads and demos."""
    if isinstance(audio_source, Path):
        source_path = audio_source
        source_name = source_path.name
        if source_path.suffix.lower() not in {".wav", ".flac"}:
            raise ValueError("Supported audio formats are WAV and FLAC.")
        audio_bytes = source_path.read_bytes()
    else:
        source_name = str(getattr(audio_source, "name", "uploaded.wav"))
        if hasattr(audio_source, "getvalue"):
            audio_bytes = bytes(audio_source.getvalue())
        else:
            audio_bytes = bytes(audio_source.getbuffer())
        suffix = Path(source_name).suffix.lower()
        if suffix not in {".wav", ".flac"}:
            raise ValueError("Supported audio formats are WAV and FLAC.")
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temporary_audio:
            temporary_audio.write(audio_bytes)
            source_path = Path(temporary_audio.name)

    try:
        clip = standardize_clip(source_path)
    finally:
        if not isinstance(audio_source, Path):
            try:
                os.remove(source_path)
            except OSError:
                pass

    features = extract_features(clip)
    if not np.isfinite(clip).all() or not all(np.isfinite(value) for value in features.values()):
        raise ValueError("Audio preprocessing produced a non-finite value.")

    if bundle is None:
        results = load_results()
        bundle = _build_bundle(results["best_svm"], results["selected_features"], "SVM")
    if bundle is None:
        raise FileNotFoundError("The saved default SVM model configuration is unavailable.")
    if not bundle.model_path.is_file() or not bundle.scaler_path.is_file():
        raise FileNotFoundError("A saved model or scaler artifact is missing.")
    missing_columns = set(bundle.feature_names).difference(features)
    if missing_columns:
        raise ValueError(f"Expected model features are missing: {sorted(missing_columns)}")

    feature_row = np.asarray([[features[name] for name in bundle.feature_names]], dtype=float)
    if feature_row.shape != (1, bundle.feature_count) or not np.isfinite(feature_row).all():
        raise ValueError("Selected feature row has an invalid shape or non-finite values.")
    model = _load_model(str(bundle.model_path))
    scaler = _load_scaler(str(bundle.scaler_path))
    scaled_row = scaler.transform(feature_row)
    if not np.isfinite(scaled_row).all():
        raise ValueError("Scaler produced a non-finite feature value.")
    prediction_values = np.asarray(model.predict(scaled_row)).reshape(-1)
    if prediction_values.size != 1 or int(prediction_values[0]) not in {0, 1}:
        raise ValueError("The saved model did not return a binary prediction.")

    return {
        "source_name": source_name,
        "source_sha256": hashlib.sha256(audio_bytes).hexdigest(),
        "clip": clip,
        "features": features,
        "feature_names": bundle.feature_names,
        "feature_count": bundle.feature_count,
        "model_name": bundle.label,
        "model_path": str(bundle.model_path),
        "prediction": int(prediction_values[0]),
    }


def analysis_detail_data(result: dict[str, Any]) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """Prepare waveform, spectrogram, and feature table for the collapsed details panel."""
    clip = np.asarray(result["clip"], dtype=float)
    waveform = pd.DataFrame({"Amplitude": clip[::80]})
    spectrum = librosa.amplitude_to_db(np.abs(librosa.stft(clip)), ref=np.max)
    feature_table = pd.DataFrame(
        {
            "Feature": result["feature_names"],
            "Value": [result["features"][name] for name in result["feature_names"]],
        }
    )
    if waveform.empty or spectrum.size == 0 or feature_table.empty:
        raise ValueError("Audio analysis details could not be prepared.")
    if not np.isfinite(waveform.to_numpy()).all() or not np.isfinite(spectrum).all():
        raise ValueError("Audio analysis details contain non-finite values.")
    return waveform, spectrum, feature_table


def _hero() -> None:
    st.markdown(
        """
<section class="qk-hero">
  <div class="qk-radar"><span class="qk-radar-sweep"></span></div>
  <span class="qk-bird qk-bird-one" aria-hidden="true"></span>
  <div class="qk-drone" aria-hidden="true"><span class="qk-prop qk-prop-left"></span><span class="qk-prop qk-prop-right"></span><span class="qk-drone-body"></span><span class="qk-drone-beam"></span></div>
  <div class="qk-hero-content">
        <div class="qk-kicker">Audio-Based Drone Detection</div>
    <h1 class="qk-hero-title">QubitSky</h1>
        <div class="qk-kicker">LISTEN BEYOND THE NOISE.</div>
        <p class="qk-hero-copy">Upload an environmental audio recording and QubitSky analyzes its acoustic characteristics to determine whether a drone may be present.</p>
  </div>
</section>
        """,
        unsafe_allow_html=True,
    )


def _navigate_to_research() -> None:
    st.session_state["main_navigation"] = "RESEARCH"


def page_detect(data: dict[str, Any]) -> None:
    _hero()
    hero_action, hero_research = st.columns([1, 1], gap="small")
    with hero_action:
        st.markdown('<a class="qk-hero-cta" href="#audio-upload">ANALYZE AUDIO</a>', unsafe_allow_html=True)
    with hero_research:
        st.button("VIEW RESEARCH", on_click=_navigate_to_research, key="hero-view-research")

    model_options = {
        "Classical SVM": _build_bundle(data["best_svm"], data["selected_features"], "SVM"),
        "Small MLP": _build_bundle(data["best_mlp"], data["selected_features"], "MLP"),
    }
    available = {name: bundle for name, bundle in model_options.items() if bundle is not None}
    if not available:
        st.error("The saved local detection models could not be loaded.")
        return
    default_model = "Classical SVM" if "Classical SVM" in available else next(iter(available))

    with st.container(key="detect-card"):
        st.subheader("UPLOAD AUDIO")
        input_mode = st.radio(
            "Audio source",
            ["Upload your own audio", "Try a Demo"],
            horizontal=True,
            label_visibility="collapsed",
            key="audio_source_mode",
        )
        demos = demo_audio_samples()
        audio_source: Any | None = None
        is_demo = False
        if input_mode == "Upload your own audio":
            uploaded = st.file_uploader(
                "Choose a WAV or FLAC recording",
                type=["wav", "flac"],
                accept_multiple_files=False,
                key="audio_upload",
                help="WAV and FLAC are enabled. MP3 decoding is unavailable in this environment.",
            )
            audio_source = uploaded
            if uploaded is None:
                st.caption("Supported formats: WAV, FLAC")
        elif demos:
            demo_label = st.selectbox("Demo example", options=list(demos), key="demo_audio_choice")
            audio_source = demos[demo_label]
            is_demo = True
            st.caption("Demo example only. Its prediction is not new scientific evaluation evidence.")
        else:
            st.info("No split-verified local demo audio is available. Upload a WAV or FLAC recording instead.")

        if audio_source is not None:
            suffix = audio_source.suffix.lower() if isinstance(audio_source, Path) else Path(audio_source.name).suffix.lower()
            audio_bytes = audio_source.read_bytes() if isinstance(audio_source, Path) else bytes(audio_source.getvalue())
            st.audio(audio_bytes, format="audio/flac" if suffix == ".flac" else "audio/wav")

            with st.expander("Advanced model selection"):
                st.caption("Choose a saved local model. Quantum models are compared on the Research page.")
                model_name = st.selectbox(
                    "Detection model",
                    options=list(available),
                    index=list(available).index(default_model),
                    label_visibility="collapsed",
                    key="advanced_detection_model",
                )
            bundle = available[model_name]
            source_hash = hashlib.sha256(audio_bytes).hexdigest()
            if st.button("ANALYZE AUDIO", type="primary", width="stretch", key="analyze-audio"):
                try:
                    with st.spinner("Analyzing recording…"):
                        result = predict_audio(audio_source, bundle)
                        result["demo_example"] = is_demo
                        st.session_state["analysis_result"] = result
                except Exception as exc:
                    st.error(f"Audio analysis failed: {exc}")

            result = st.session_state.get("analysis_result", {})
            if result.get("source_sha256") == source_hash and result.get("model_name") == bundle.label:
                detected = result["prediction"] == 1
                result_text = "DRONE DETECTED" if detected else "NO DRONE DETECTED"
                result_class = "drone" if detected else "no-drone"
                st.markdown(
                    f"<div class='qk-result {result_class}'><div class='qk-kicker'>Prediction</div><div class='qk-result-title'>{result_text}</div><div>Model: {result['model_name']}</div></div>",
                    unsafe_allow_html=True,
                )
                if result.get("demo_example"):
                    st.caption("Demo example; this output is not a new evaluation result.")
                with st.expander("View analysis details"):
                    waveform, spectrum, feature_table = analysis_detail_data(result)
                    st.markdown("**Waveform**")
                    st.line_chart(waveform, height=180)
                    st.markdown("**Spectrogram**")
                    fig = px.imshow(
                        spectrum,
                        origin="lower",
                        aspect="auto",
                        color_continuous_scale=[[0, "#102b61"], [0.45, "#27c4df"], [0.76, "#7860d8"], [1, "#e9a347"]],
                    )
                    fig.update_layout(height=300, margin=dict(l=8, r=8, t=10, b=8))
                    st.plotly_chart(fig, width="stretch")
                    st.dataframe(feature_table, width="stretch", hide_index=True)
                    st.caption(f"Local {result['model_name']} · {result['feature_count']} selected features · {Path(result['model_path']).name}")

        st.session_state["audio_uploaded"] = audio_source is not None

    with st.expander("How does this work?"):
        st.write("Audio → Acoustic Features → ML / Quantum Model → Drone or No Drone")
        st.write("The quantum model encodes four selected acoustic features into four qubits and compares their quantum-state similarity.")


def _matched_model_results(data: dict[str, Any]) -> pd.DataFrame:
    ideal = data.get("ideal", pd.DataFrame())
    if ideal.empty:
        return pd.DataFrame()
    matched = ideal[
        ideal["evaluation_split"].eq("test")
        & ideal["experiment"].eq("matched_sample_clean_model_robustness")
        & ideal["evaluation_snr"].eq("clean")
        & ideal["feature_count"].eq(4)
        & ideal["training_size_requested"].eq(24)
    ].copy()
    names = {
        "rbf_svm": "SVM",
        "small_mlp": "MLP",
        "fixed_qsvc": "Fixed QSVC",
        "trainable_qsvc": "Trainable QSVC",
    }
    matched["Model"] = matched["model"].map(names)
    return matched[matched["Model"].notna()]


def _validated_qpu_result(data: dict[str, Any]) -> dict[str, Any] | None:
    """Return a completed result only when local ledger caches cover all circuits exactly."""
    result_frame = data.get("qpu_results", pd.DataFrame())
    ledger = data.get("qpu_jobs", {})
    if result_frame.empty or not isinstance(ledger, dict):
        return None
    completed = result_frame[result_frame.get("status", pd.Series(dtype=str)).astype(str).str.lower().eq("completed")]
    if completed.empty:
        return None
    required_metrics = ("accuracy", "f1", "balanced_accuracy", "drone_recall", "non_drone_recall")
    result = completed.iloc[-1].to_dict()
    if any(metric not in result or not np.isfinite(float(result[metric])) for metric in required_metrics):
        return None
    if (
        result.get("backend_name") != "ibm_pittsburgh"
        or int(result.get("shots", 0)) != 1024
        or int(result.get("circuit_count", 0)) != 4548
        or int(result.get("job_count", 0)) != 4
    ):
        return None

    jobs = ledger.get("jobs", [])
    if not isinstance(jobs, list) or len(jobs) != 4:
        return None
    covered: list[int] = []
    job_ids: list[str] = []
    for batch_index, job in enumerate(jobs):
        expected_start = batch_index * 1137
        expected_end = min(expected_start + 1137, 4548)
        if (
            job.get("status") != "completed"
            or job.get("backend") != "ibm_pittsburgh"
            or job.get("physical_layout") != [87, 97, 107, 108]
            or int(job.get("shots", 0)) != 1024
            or int(job.get("circuit_index_start", -1)) != expected_start
            or int(job.get("circuit_index_end_exclusive", -1)) != expected_end
        ):
            return None
        cache_value = job.get("result_cache")
        if not cache_value:
            return None
        cache_path = Path(cache_value)
        if not cache_path.is_absolute():
            cache_path = PROJECT_ROOT / cache_path
        if not cache_path.is_file():
            cache_path = QUANTUM_DIR / "qpu_job_cache" / Path(cache_value).name
        if not cache_path.is_file():
            return None
        try:
            cache = _safe_read_json(cache_path)
        except (OSError, json.JSONDecodeError):
            return None
        expected_indices = list(range(expected_start, expected_end))
        counts = cache.get("counts", [])
        pairs = cache.get("pairs", [])
        pair_keys = [
            (pair.get("split"), pair.get("left_index"), pair.get("right_index"))
            for pair in pairs
            if isinstance(pair, dict)
        ]
        if (
            cache.get("job_id") != job.get("job_id")
            or cache.get("backend") != "ibm_pittsburgh"
            or cache.get("physical_layout") != [87, 97, 107, 108]
            or int(cache.get("shots", 0)) != 1024
            or cache.get("circuit_index_start") != expected_start
            or cache.get("circuit_index_end_exclusive") != expected_end
            or cache.get("circuit_indices") != expected_indices
            or len(pairs) != len(expected_indices)
            or len(pair_keys) != len(pairs)
            or len(set(pair_keys)) != len(pair_keys)
            or len(counts) != len(expected_indices)
            or any(not isinstance(count_map, dict) or sum(count_map.values()) != 1024 for count_map in counts)
        ):
            return None
        job_ids.append(str(job.get("job_id")))
        covered.extend(cache["circuit_indices"])
    try:
        result_job_ids = json.loads(str(result.get("job_ids", "[]")))
    except json.JSONDecodeError:
        return None
    if covered != list(range(4548)) or result_job_ids != job_ids:
        return None
    return result


def _final_comparison_rows(data: dict[str, Any], qpu_result: dict[str, Any] | None) -> pd.DataFrame:
    display_names = {
        "RBF SVM": "SVM",
        "Small MLP": "MLP",
        "Ideal quantum simulator": "Ideal trainable QSVC",
        "Finite-shot zero-noise simulator": "Finite-shot ideal simulator",
        "Stage 6 noisy simulator (2% synthetic)": "Stage 6 noisy simulator (2%)",
        "Backend-derived noisy simulator": "Backend-derived Aer",
    }
    final_rows = data.get("final", pd.DataFrame())
    rows: list[dict[str, Any]] = []
    if not final_rows.empty and {"model", "status", "f1"}.issubset(final_rows.columns):
        for record in final_rows.to_dict(orient="records"):
            model_name = display_names.get(str(record.get("model")))
            if model_name is None or str(record.get("status", "")).lower() != "measured":
                continue
            if not pd.notna(record.get("f1")):
                continue
            rows.append({"Model": model_name, **record})
    else:
        matched = _matched_model_results(data)
        names = {"rbf_svm": "SVM", "small_mlp": "MLP", "fixed_qsvc": "Fixed QSVC", "trainable_qsvc": "Ideal trainable QSVC"}
        for record in matched.to_dict(orient="records"):
            rows.append({"Model": names.get(str(record.get("model")), str(record.get("model"))), **record})

    if qpu_result is not None:
        rows.append({"Model": "Real QPU", **qpu_result})
    return pd.DataFrame(rows)


def page_research(data: dict[str, Any]) -> None:
    st.title("THE RESEARCH BEHIND QUBITSKY")
    st.write(
        "Can quantum-kernel machine learning remain competitive with classical models when labeled data are limited and acoustic conditions are noisy?"
    )

    matched = _matched_model_results(data)
    qpu_result = _validated_qpu_result(data)
    comparison_rows = _final_comparison_rows(data, qpu_result)

    st.subheader("F1 Score by Model")
    chart_names = {
        "SVM": "SVM",
        "MLP": "MLP",
        "Ideal trainable QSVC": "Ideal Quantum",
        "Stage 6 noisy simulator (2%)": "Noisy Quantum",
        "Real QPU": "Real QPU",
    }
    chart_rows = comparison_rows[comparison_rows["Model"].isin(chart_names)].copy() if not comparison_rows.empty else pd.DataFrame()
    if not chart_rows.empty:
        chart_rows["Model"] = chart_rows["Model"].map(chart_names)
        chart_rows = chart_rows.drop_duplicates("Model", keep="last")
    chart_rows = chart_rows[["Model", "f1"]].rename(columns={"f1": "F1"}) if not chart_rows.empty else pd.DataFrame()
    if chart_rows.empty:
        st.info("No matched comparison results are available yet.")
    else:
        fig = px.bar(
            chart_rows,
            x="Model",
            y="F1",
            color="Model",
            text_auto=".3f",
            color_discrete_map={
                "SVM": "#2877d4",
                "MLP": "#119b91",
                "Ideal Quantum": "#7657d6",
                "Noisy Quantum": "#a18af0",
                "Real QPU": "#38c9e8",
            },
        )
        fig.update_layout(height=360, margin=dict(l=8, r=8, t=16, b=8), showlegend=False, yaxis_range=[0, 1])
        st.plotly_chart(fig, width="stretch")

    if not comparison_rows.empty:
        with st.expander("All comparison metrics"):
            comparison = comparison_rows[["Model", "accuracy", "precision", "recall", "f1", "balanced_accuracy", "drone_recall", "non_drone_recall"]].rename(
                columns={
                    "accuracy": "Accuracy",
                    "precision": "Precision",
                    "recall": "Recall",
                    "f1": "F1",
                    "balanced_accuracy": "Balanced accuracy",
                    "drone_recall": "Drone recall",
                    "non_drone_recall": "Non-drone recall",
                }
            )
            st.dataframe(comparison, width="stretch", hide_index=True, column_config={
                "Accuracy": st.column_config.NumberColumn(format="%.3f"),
                "Precision": st.column_config.NumberColumn(format="%.3f"),
                "Recall": st.column_config.NumberColumn(format="%.3f"),
                "F1": st.column_config.NumberColumn(format="%.3f"),
                "Balanced accuracy": st.column_config.NumberColumn(format="%.3f"),
                "Drone recall": st.column_config.NumberColumn(format="%.3f"),
                "Non-drone recall": st.column_config.NumberColumn(format="%.3f"),
            })
        st.caption("Matched held-out test set: four selected features and 24 training recordings. Scores are experiment results, not a claim of quantum advantage.")

    ledger = data.get("qpu_jobs", {}) if isinstance(data.get("qpu_jobs"), dict) else {}
    jobs = ledger.get("jobs", []) if isinstance(ledger.get("jobs", []), list) else []
    completed = sum(str(job.get("status", "")).lower() == "completed" for job in jobs)
    planned = 4
    batch4 = next((job for job in jobs if int(job.get("circuit_index_start", -1)) == 3411), None)
    batch4_state = str(batch4.get("status", "not submitted")).replace("_", " ").title() if batch4 else "Not submitted"
    if batch4_state == "Submitted":
        batch4_state = "Queued / Running"
    st.markdown("### REAL IBM QPU EXPERIMENT")
    if qpu_result is not None:
        st.markdown("**REAL IBM QPU EXPERIMENT COMPLETED**")
        st.caption(f"Backend: {qpu_result['backend_name']} · 4 qubits · {int(qpu_result['shots'])} shots")
        q1, q2 = st.columns(2)
        q1.metric("F1", f"{float(qpu_result['f1']):.3f}")
        q2.metric("Accuracy", f"{float(qpu_result['accuracy']):.3f}")
        q3, q4 = st.columns(2)
        q3.metric("Balanced accuracy", f"{float(qpu_result['balanced_accuracy']):.3f}")
        q4.metric("Drone recall", f"{float(qpu_result['drone_recall']):.3f}")
        st.caption(f"Non-drone recall: {float(qpu_result['non_drone_recall']):.3f}")
    else:
        st.markdown(
            f"<div class='qk-card'><strong>{completed} / {planned} batches completed</strong><br>Backend: {ledger.get('backend', 'ibm_pittsburgh')} · 4 qubits<br>Batch 4: {batch4_state}</div>",
            unsafe_allow_html=True,
        )

    with st.expander("Noise Robustness"):
        noisy = data.get("noisy", pd.DataFrame())
        if noisy.empty:
            st.info("No saved noise-sweep results are available.")
        else:
            fig = px.line(
                noisy.sort_values("error_level"),
                x="error_level",
                y="f1",
                color="noise_type",
                markers=True,
                labels={"error_level": "Simulated noise level", "f1": "F1 score", "noise_type": "Noise setting"},
                color_discrete_sequence=["#38c9e8", "#7657d6", "#119b91"],
            )
            fig.update_layout(height=330, margin=dict(l=8, r=8, t=12, b=8), yaxis_range=[0, 1])
            st.plotly_chart(fig, width="stretch")

    with st.expander("Training Size"):
        if matched.empty:
            st.info("No matched training-size results are available.")
        else:
            size_results = data["ideal"]
            size_results = size_results[
                size_results["evaluation_split"].eq("test")
                & size_results["experiment"].eq("matched_sample_clean_model_robustness")
                & size_results["evaluation_snr"].eq("clean")
                & size_results["feature_count"].eq(4)
                & size_results["training_snr"].eq("clean")
                & size_results["model"].isin(["rbf_svm", "small_mlp", "fixed_qsvc", "trainable_qsvc"])
            ].copy()
            size_results["Model"] = size_results["model"].map(
                {"rbf_svm": "SVM", "small_mlp": "MLP", "fixed_qsvc": "Fixed QSVC", "trainable_qsvc": "Trainable QSVC"}
            )
            fig = px.line(
                size_results.groupby(["training_size_requested", "Model"], as_index=False)["f1"].mean(),
                x="training_size_requested",
                y="f1",
                color="Model",
                markers=True,
                labels={"training_size_requested": "Training recordings", "f1": "F1 score"},
            )
            fig.update_layout(height=330, margin=dict(l=8, r=8, t=12, b=8), yaxis_range=[0, 1])
            st.plotly_chart(fig, width="stretch")

    with st.expander("View hardware details"):
        st.caption("Read-only local ledger snapshot. This page never contacts IBM or changes job state.")
        if jobs:
            jobs_df = pd.DataFrame(jobs)
            columns = [
                name
                for name in ["job_id", "status", "circuit_index_start", "circuit_index_end_exclusive", "circuit_count", "shots"]
                if name in jobs_df.columns
            ]
            st.dataframe(jobs_df[columns], width="stretch", hide_index=True)
        else:
            st.info("No local hardware jobs are recorded.")


def page_about(data: dict[str, Any]) -> None:
    st.title("ABOUT QUBITSKY")
    st.subheader("WHAT IS QUBITSKY?")
    st.write("QubitSky is an acoustic drone-detection research prototype.")
    st.subheader("HOW DOES IT WORK?")
    st.write("A microphone captures environmental audio. The system extracts acoustic features and predicts Drone or No Drone.")
    st.write("Birds, wind, helicopters, aircraft, engines and insects can sound similar or interfere with detection.")
    st.subheader("WHERE DOES QUANTUM FIT?")
    st.write("QubitSky compares conventional models with a four-qubit quantum-kernel model, including execution on a real IBM quantum processor, under limited-data and noisy conditions.")
    st.warning("Research prototype — not a safety-critical anti-drone system.")


def main() -> None:
    st.set_page_config(page_title="QubitSky", page_icon=":satellite:", layout="wide")
    _inject_style(PROJECT_ROOT / "background.jpg")
    data = load_results()
    ambience_path = ASSETS_DIR / "ambience.mp3"

    nav_col, sound_col = st.columns([5, 1], vertical_alignment="center")
    page = nav_col.radio(
        "Navigate",
        ["DETECT", "RESEARCH", "ABOUT"],
        index=0,
        horizontal=True,
        label_visibility="collapsed",
        key="main_navigation",
    )
    st.session_state.setdefault("ambient_on", False)
    ambient_label = "🔊" if st.session_state["ambient_on"] else "🔇"
    if sound_col.button(
        ambient_label,
        help="Ambient sound. Optional and off by default; turns off when an audio clip is uploaded.",
        disabled=not ambience_path.is_file(),
    ):
        st.session_state["ambient_on"] = not st.session_state["ambient_on"]

    if page == "DETECT":
        page_detect(data)
    elif page == "RESEARCH":
        page_research(data)
    else:
        page_about(data)

    if st.session_state.get("audio_uploaded", False):
        st.session_state["ambient_on"] = False
    if st.session_state["ambient_on"] and ambience_path.is_file():
        with ambience_path.open("rb") as audio_handle:
            st.audio(audio_handle.read(), format="audio/mp3", autoplay=True, loop=True)


if __name__ == "__main__":
    main()
