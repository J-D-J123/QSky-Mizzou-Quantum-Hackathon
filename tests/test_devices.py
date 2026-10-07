"""Device selection and inference checks; these tests never fit a model."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from qsky_classical import cnn, models
from qsky_classical.cli import parser, run


@pytest.fixture(autouse=True)
def prohibit_training(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Device checks must not start training")

    monkeypatch.setattr(cnn, "fit_cnn", forbidden)
    monkeypatch.setattr(models, "fit_tabular", forbidden)


def test_cpu_override_never_queries_cuda(monkeypatch):
    def forbidden():
        pytest.fail("Explicit CPU selection must not initialize CUDA")

    monkeypatch.setattr(torch.cuda, "is_available", forbidden)
    assert cnn.resolve_device("cpu") == torch.device("cpu")


def test_auto_without_cuda_uses_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert cnn.resolve_device() == torch.device("cpu")


def test_explicit_cuda_without_gpu_fails(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="CUDA was requested"):
        cnn.resolve_device("cuda")


def test_auto_probes_visible_gpu_and_configures_determinism(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    probed = []
    monkeypatch.setattr(cnn, "_probe_cuda", probed.append)
    assert cnn.resolve_device() == torch.device("cuda:0")
    assert probed == [torch.device("cuda:0")]
    assert cnn.os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"


def test_auto_skips_unusable_gpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    probed = []

    def probe(device):
        probed.append(device)
        if device.index == 0:
            raise RuntimeError("device unavailable")

    monkeypatch.setattr(cnn, "_probe_cuda", probe)
    assert cnn.resolve_device() == torch.device("cuda:1")
    assert probed == [torch.device("cuda:0"), torch.device("cuda:1")]


def test_broken_cuda_falls_back_only_in_auto_mode(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)

    def broken(device):
        raise RuntimeError("no kernel image is available")

    monkeypatch.setattr(cnn, "_probe_cuda", broken)
    with pytest.warns(RuntimeWarning, match="falling back to CPU"):
        assert cnn.resolve_device() == torch.device("cpu")
    with pytest.raises(ValueError, match="no kernel image"):
        cnn.resolve_device("cuda")


def test_cuda_discovery_failure_falls_back(monkeypatch):
    def broken():
        raise RuntimeError("driver initialization failed")

    monkeypatch.setattr(torch.cuda, "is_available", broken)
    with pytest.warns(RuntimeWarning, match="driver initialization failed"):
        assert cnn.resolve_device() == torch.device("cpu")


def test_invalid_device_rejected():
    with pytest.raises(ValueError, match="device must be"):
        cnn.resolve_device("typo")


def test_cli_device_defaults_and_override():
    assert parser().parse_args(["--manifest", "unused.csv"]).device == "auto"
    assert parser().parse_args(["--manifest", "unused.csv", "--device", "cpu"]).device == "cpu"


def test_cuda_requirement_fails_before_reading_audio_or_creating_output(monkeypatch, tmp_path):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    output = tmp_path / "run"
    args = parser().parse_args([
        "--manifest", "does-not-exist.csv", "--models", "D", "--device", "cuda",
        "--output", str(output),
    ])
    with pytest.raises(ValueError, match="CUDA was requested"):
        run(args)
    assert not output.exists()


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_inference_and_portable_checkpoint_without_training(device, tmp_path):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("Real CUDA inference requires a CUDA-enabled machine")
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        network = cnn.LogMelCNN().to(device)
        model = cnn.CNNModel(network, mean=0.5, std=2.0, batch_size=2)
        x = np.random.default_rng(9).normal(size=(3, 64, 101)).astype(np.float32)
        probabilities = model.predict_proba(x)
        assert probabilities.shape == (3, 2)
        assert np.isfinite(probabilities).all()
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)

        checkpoint = tmp_path / "model_D.pt"
        model.save(checkpoint)
        saved = torch.load(checkpoint, weights_only=True)
        assert all(value.device.type == "cpu" for value in saved["state_dict"].values())
        assert saved["training_device"].startswith(device)
        restored_network = cnn.LogMelCNN()
        restored_network.load_state_dict(saved["state_dict"])
        restored = cnn.CNNModel(restored_network, saved["mean"], saved["std"], saved["batch_size"])
        np.testing.assert_allclose(restored.predict_proba(x), probabilities, atol=1e-5, rtol=1e-5)
        assert model.device.type == device  # Saving does not move the live model.
    finally:
        torch.set_num_threads(previous_threads)
