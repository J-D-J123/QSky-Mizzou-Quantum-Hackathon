"""Optional PyTorch log-mel reference; no PCA or angle scaling."""

import copy
import os
import time
import warnings

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .models import clip_validation_score


def _probe_cuda(device):
    # Execute a kernel: availability alone does not guarantee a working driver/build.
    torch.ones(1, device=device).add_(1).cpu()
    torch.cuda.synchronize(device)


def resolve_device(requested="auto"):
    """Use the first usable visible CUDA GPU, or CPU in automatic mode."""
    if requested not in ("auto", "cpu", "cuda"):
        raise ValueError("device must be auto, cpu, or cuda")
    if requested == "cpu":
        return torch.device("cpu")
    failures = []
    try:
        count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    except (RuntimeError, AssertionError) as error:
        count = 0
        failures.append(str(error))
    if count:
        # Required by deterministic CUDA matrix multiplication; set before probing.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    for index in range(count):
        device = torch.device(f"cuda:{index}")
        try:
            _probe_cuda(device)
            return device
        except (RuntimeError, AssertionError) as error:
            failures.append(f"{device}: {error}")
    reason = "; ".join(failures) or "no CUDA GPU is available to this PyTorch installation"
    if requested == "cuda":
        raise ValueError(f"CUDA was requested but is unusable: {reason}. Use --device auto or cpu.")
    if failures:
        warnings.warn(f"CUDA is unusable; falling back to CPU: {reason}", RuntimeWarning, stacklevel=2)
    return torch.device("cpu")


def device_name(device):
    return torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"


class GlobalAveragePool(nn.Module):
    # Same output as AdaptiveAvgPool2d((1, 1)), whose CUDA gradient has no deterministic
    # implementation and raises under torch.use_deterministic_algorithms(True).
    def forward(self, x):
        return x.mean(dim=(2, 3), keepdim=True)


class LogMelCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            GlobalAveragePool(), nn.Flatten(),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.layers(x).squeeze(1)


class CNNModel:
    def __init__(self, model, mean, std, batch_size):
        self.model, self.mean, self.std, self.batch_size = model, mean, std, batch_size

    def tensor(self, x):
        return torch.from_numpy(((x - self.mean) / self.std).astype(np.float32))[:, None]

    @property
    def device(self):
        return next(self.model.parameters()).device

    def predict_proba(self, x):
        self.model.eval()
        scores = []
        with torch.no_grad():
            for batch in self.tensor(x).split(self.batch_size):
                scores.append(torch.sigmoid(self.model(batch.to(self.device))).cpu().numpy())
        p = np.concatenate(scores)
        return np.column_stack([1 - p, p])

    def save(self, path):
        torch.save({
            # Store CPU tensors so CUDA-trained checkpoints also load on laptops.
            "state_dict": {k: v.detach().cpu() for k, v in self.model.state_dict().items()},
            "mean": self.mean, "std": self.std,
            "batch_size": self.batch_size, "architecture": "LogMelCNN",
            "training_device": str(self.device), "device_name": device_name(self.device),
        }, path)


def fit_cnn(train_x, train_y, val_x, val_y, val_ids, *, seed, epochs, batch_size, device="auto"):
    start = time.perf_counter()
    device = device if isinstance(device, torch.device) else resolve_device(device)
    torch.manual_seed(seed)
    if device.type == "cpu":
        torch.set_num_threads(1)
    else:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        # DataLoader's pinned-memory allocation uses the current CUDA device.
        torch.cuda.set_device(device)
        torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    model = LogMelCNN().to(device)
    mean, std = float(train_x.mean()), max(float(train_x.std()), 1e-6)
    wrapper = CNNModel(model, mean, std, batch_size)
    dataset = TensorDataset(wrapper.tensor(train_x), torch.tensor(train_y, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(seed),
                        pin_memory=device.type == "cuda")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    positive_weight = float((train_y == 0).sum() / (train_y == 1).sum())
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(positive_weight, device=device))
    best_score, best_state, best_epoch = -np.inf, None, 0
    trials = []
    for epoch in range(1, epochs + 1):
        model.train()
        for x, y in loader:
            x = x.to(device, non_blocking=device.type == "cuda")
            y = y.to(device, non_blocking=device.type == "cuda")
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
        score = clip_validation_score(val_y, wrapper.predict_proba(val_x)[:, 1], val_ids, 0.5)
        trials.append({"epoch": epoch, "val_clip_balanced_accuracy": float(score)})
        if score > best_score:
            best_score, best_state, best_epoch = score, copy.deepcopy(model.state_dict()), epoch
    model.load_state_dict(best_state)
    return wrapper, {
        "params": {"epochs": epochs, "best_epoch": best_epoch, "batch_size": batch_size,
                   "learning_rate": 1e-3, "weight_decay": 1e-4,
                   "device": str(device), "device_name": device_name(device)},
        "val_clip_balanced_accuracy": float(best_score), "threshold": 0.5,
        "fit_seconds": time.perf_counter() - start,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "support_vectors": None, "trials": trials,
    }
