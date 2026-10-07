from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from app import (
    analysis_detail_data,
    demo_audio_samples,
    load_results,
    predict_audio,
    _final_comparison_rows,
    _validated_qpu_result,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class AudioInferenceSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.demos = demo_audio_samples()
        manifest = json.loads((PROJECT_ROOT / "assets" / "demo_audio" / "demo_samples.json").read_text())
        cls.demo_manifest = manifest["samples"]

    def _check_pipeline(self, demo_label: str, expected_folder: str) -> dict:
        self.assertIn(demo_label, self.demos, f"No split-verified {demo_label} demo audio was found")
        audio_path = self.demos[demo_label]
        self.assertTrue(audio_path.is_file())
        self.assertEqual(audio_path.suffix.lower(), ".flac")
        recording_id = audio_path.name.split("__", maxsplit=1)[0]
        self.assertEqual(self.demo_manifest[demo_label]["split"], "test")
        self.assertEqual(self.demo_manifest[demo_label]["recording_id"], recording_id)
        self.assertIn(expected_folder, audio_path.parts)

        result = predict_audio(audio_path)
        self.assertIn(result["prediction"], (0, 1))
        self.assertTrue(np.isfinite(result["clip"]).all())
        self.assertEqual(result["clip"].shape, (48_000,), "Preprocessing should resample/pad to mono 16 kHz x 3 s")
        self.assertTrue(all(np.isfinite(value) for value in result["features"].values()))
        self.assertTrue(result["feature_count"] > 0)
        self.assertEqual(len(result["feature_names"]), result["feature_count"])
        self.assertEqual(result["feature_names"], [
            "mfcc_2_mean",
            "spectral_flatness_mean",
            "spectral_bandwidth_mean",
            "mfcc_1_mean",
            "rms_energy_mean",
            "spectral_centroid_mean",
        ])
        self.assertTrue(Path(result["model_path"]).is_file())
        self.assertTrue(Path(result["model_path"]).name)
        waveform, spectrogram, feature_table = analysis_detail_data(result)
        self.assertFalse(waveform.empty)
        self.assertTrue(np.isfinite(spectrogram).all())
        self.assertEqual(feature_table["Feature"].tolist(), result["feature_names"])
        self.assertTrue(np.isfinite(feature_table["Value"].to_numpy()).all())
        print(
            f"{demo_label}: {audio_path.relative_to(PROJECT_ROOT)} -> "
            f"{'DRONE DETECTED' if result['prediction'] == 1 else 'NO DRONE DETECTED'} "
            f"({result['model_name']}; demo only)"
        )
        return result

    def test_held_out_drone_demo_runs_inference(self) -> None:
        self._check_pipeline("Drone Sample", "drone")

    def test_held_out_environment_demo_runs_inference(self) -> None:
        self._check_pipeline("Environmental / No-Drone Sample", "background")

    def test_wav_upload_and_analyze_button_use_shared_inference(self) -> None:
        from streamlit.testing.v1 import AppTest

        demo_path = self.demos.get("Environmental / No-Drone Sample")
        self.assertIsNotNone(demo_path, "No held-out environmental demo is available for WAV upload testing")
        audio, sample_rate = sf.read(demo_path, dtype="float32")
        with tempfile.TemporaryDirectory(prefix="qsky-upload-smoke-") as temp_dir:
            wav_path = Path(temp_dir) / "demo_upload.wav"
            sf.write(wav_path, audio, sample_rate, format="WAV")
            app = AppTest.from_file(str(PROJECT_ROOT / "app.py")).run()
            self.assertEqual(app.radio[0].value, "DETECT")
            app.get("file_uploader")[0].set_value(
                (wav_path.name, wav_path.read_bytes(), "audio/wav")
            ).run()
            self.assertEqual(len(app.exception), 0)
            self.assertTrue(app.get("audio"), "Uploaded WAV should show an audio player")
            analyze = next(button for button in app.button if button.label == "ANALYZE AUDIO")
            analyze.click().run()
            self.assertEqual(len(app.exception), 0)
            result_markup = [element.value for element in app.markdown if "DRONE DETECTED" in element.value]
            self.assertTrue(result_markup, "Analyze should render a binary prediction")
            print(f"WAV upload: {result_markup[-1]}")
            self.assertTrue(any(expander.label == "View analysis details" for expander in app.get("expander")))

    def test_demo_mode_analyze_and_details(self) -> None:
        from streamlit.testing.v1 import AppTest

        app = AppTest.from_file(str(PROJECT_ROOT / "app.py")).run(timeout=20)
        app.radio[1].set_value("Try a Demo").run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(
            app.selectbox[0].options,
            ["Drone Sample", "Environmental / No-Drone Sample"],
        )
        self.assertTrue(app.get("audio"), "Selecting a demo should show an audio player")
        next(button for button in app.button if button.label == "ANALYZE AUDIO").click().run(timeout=20)
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(any("Demo example; this output is not a new evaluation result." in item.value for item in app.caption))
        self.assertTrue(any(expander.label == "View analysis details" for expander in app.get("expander")))

    def test_research_page_shows_only_validated_completed_qpu_result(self) -> None:
        from streamlit.testing.v1 import AppTest

        app = AppTest.from_file(str(PROJECT_ROOT / "app.py")).run()
        app.radio[0].set_value("RESEARCH").run()
        self.assertEqual(len(app.exception), 0)
        rendered_titles = [title.value for title in app.title]
        self.assertIn("THE RESEARCH BEHIND QUBITSKY", rendered_titles)
        text_content = "\n".join(element.value for element in app.markdown)
        self.assertIn("REAL IBM QPU EXPERIMENT COMPLETED", text_content)
        captions = "\n".join(caption.value for caption in app.caption)
        self.assertIn("ibm_pittsburgh", captions)
        self.assertIn("1024 shots", captions)
        self.assertNotIn("3 / 4", text_content)
        self.assertNotIn("Experiment in progress", text_content)
        self.assertNotIn("Experiment paused", text_content)
        self.assertEqual(len(app.metric), 4)
        self.assertGreaterEqual(len(app.get("plotly_chart")), 1)
        self.assertTrue(any(expander.label == "All comparison metrics" for expander in app.get("expander")))
        qpu_result = _validated_qpu_result(load_results())
        self.assertIsNotNone(qpu_result)
        comparison = _final_comparison_rows(load_results(), qpu_result)
        chart_categories = {"SVM", "MLP", "Ideal trainable QSVC", "Stage 6 noisy simulator (2%)", "Real QPU"}
        self.assertTrue(chart_categories.issubset(set(comparison["Model"])))
        self.assertEqual(len(app.exception), 0)


if __name__ == "__main__":
    unittest.main()
