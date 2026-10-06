"""Optional PyTorch log-mel reference; no PCA or angle scaling."""

import copy
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .models import clip_validation_score


class LogMelCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.layers(x).squeeze(1)


class CNNModel:
    def __init__(self, model, mean, std, batch_size):
        self.model, self.mean, self.std, self.batch_size = model, mean, std, batch_size

    def tensor(self, x):
        return torch.from_numpy(((x - self.mean) / self.std).astype(np.float32))[:, None]

    def predict_proba(self, x):
        self.model.eval()
        scores = []
        with torch.no_grad():
            for batch in self.tensor(x).split(self.batch_size):
                scores.append(torch.sigmoid(self.model(batch)).numpy())
        p = np.concatenate(scores)
        return np.column_stack([1 - p, p])

    def save(self, path):
        torch.save({
            "state_dict": self.model.state_dict(), "mean": self.mean, "std": self.std,
            "batch_size": self.batch_size, "architecture": "LogMelCNN",
        }, path)


def fit_cnn(train_x, train_y, val_x, val_y, val_ids, *, seed, epochs, batch_size):
    start = time.perf_counter()
    torch.manual_seed(seed)
    # CPU execution keeps the default installation portable and deterministic.
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    model = LogMelCNN()
    mean, std = float(train_x.mean()), max(float(train_x.std()), 1e-6)
    wrapper = CNNModel(model, mean, std, batch_size)
    dataset = TensorDataset(wrapper.tensor(train_x), torch.tensor(train_y, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    positive_weight = float((train_y == 0).sum() / (train_y == 1).sum())
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(positive_weight))
    best_score, best_state, best_epoch = -np.inf, None, 0
    trials = []
    for epoch in range(1, epochs + 1):
        model.train()
        for x, y in loader:
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
                   "learning_rate": 1e-3, "weight_decay": 1e-4},
        "val_clip_balanced_accuracy": float(best_score), "threshold": 0.5,
        "fit_seconds": time.perf_counter() - start,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "support_vectors": None, "trials": trials,
    }
