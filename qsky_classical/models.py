"""Validation-selected classical models and shared binary metrics."""

import time

import numpy as np
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC


# The two recording-level procedures are different experiments, not interchangeable:
# averaging features yields one training/evaluation example per recording, while
# averaging scores classifies every window first.
AGGREGATIONS = {
    "score-mean": "classify each window, then average the window scores of a recording",
    "feature-mean": "average the window features of a recording, then classify it once",
}
SELECTION_METRICS = ("balanced_accuracy", "f1")


def aggregate_clips(labels, scores, clip_ids):
    ids, inverse = np.unique(clip_ids, return_inverse=True)
    sums = np.bincount(inverse, weights=scores)
    counts = np.bincount(inverse)
    clip_labels = np.array([labels[np.flatnonzero(inverse == i)[0]] for i in range(len(ids))])
    return clip_labels, sums / counts, ids


def aggregate_features(x, labels, ids):
    """Mean feature vector per ID, in sorted ID order; an ID must have one label."""
    unique, first, inverse = np.unique(ids, return_index=True, return_inverse=True)
    labels = np.asarray(labels)
    if (labels != labels[first][inverse]).any():
        raise ValueError("A recording has conflicting labels; cannot average its features.")
    sums = np.zeros((len(unique), x.shape[1]), dtype=np.float64)
    np.add.at(sums, inverse, x)
    return sums / np.bincount(inverse)[:, None], labels[first], unique


def mlp_parameter_count(inputs, hidden):
    sizes = [inputs, *hidden, 1]
    return sum((a + 1) * b for a, b in zip(sizes, sizes[1:]))


def metrics(labels, scores, threshold):
    predicted = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(labels, predicted),
        "balanced_accuracy": balanced_accuracy_score(labels, predicted),
        "precision": precision_score(labels, predicted, zero_division=0),
        "recall": recall_score(labels, predicted, zero_division=0),
        "f1": f1_score(labels, predicted, zero_division=0),
        "specificity": recall_score(labels, predicted, pos_label=0, zero_division=0),
        "roc_auc": roc_auc_score(labels, scores) if len(np.unique(labels)) == 2 else float("nan"),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp), "n_examples": len(labels),
    }


def scores_for(model, x):
    if isinstance(model, SVC):
        return model.decision_function(x)
    return model.predict_proba(x)[:, 1]


def clip_validation_score(labels, scores, ids, threshold):
    y, s, _ = aggregate_clips(labels, scores, ids)
    return balanced_accuracy_score(y, s >= threshold)


def selection_key(labels, scores, ids, threshold, metric="balanced_accuracy"):
    """Validation-only ranking key; the first element is the headline metric."""
    y, s, _ = aggregate_clips(labels, scores, ids)
    predicted = (s >= threshold).astype(int)
    balanced = balanced_accuracy_score(y, predicted)
    if metric == "balanced_accuracy":
        return (balanced,)
    if metric == "f1":
        return (f1_score(y, predicted, zero_division=0), balanced,
                recall_score(y, predicted, zero_division=0))
    raise ValueError(f"selection metric must be one of {SELECTION_METRICS}")


def fit_tabular(name, train_x, train_y, val_x, val_y, val_ids, *, seed, small_width, max_iter,
                selection="balanced_accuracy"):
    candidates = []
    if name == "A":
        for c in (0.1, 1.0, 10.0):
            for gamma in ("scale", 0.1, 1.0):
                params = dict(C=c, gamma=gamma, kernel="rbf", class_weight="balanced")
                candidates.append((SVC(**params, random_state=seed), params))
        threshold = 0.0
    else:
        widths = [(small_width,)] if name == "B" else [(64, 32), (128, 64)]
        for hidden in widths:
            for alpha in (1e-4, 1e-2):
                params = dict(hidden_layer_sizes=hidden, alpha=alpha, solver="lbfgs", max_iter=max_iter)
                candidates.append((MLPClassifier(**params, random_state=seed), params))
        threshold = 0.5
    best_model, best_params, best_score, best_key = None, None, -np.inf, None
    trials = []
    start = time.perf_counter()
    for model, params in candidates:
        model.fit(train_x, train_y)
        val_scores = scores_for(model, val_x)
        score = clip_validation_score(val_y, val_scores, val_ids, threshold)
        key = selection_key(val_y, val_scores, val_ids, threshold, selection)
        trials.append({"params": params, "val_clip_balanced_accuracy": float(score),
                       "val_selection_key": [float(v) for v in key]})
        # Strict improvement only: ties keep the first candidate.
        if best_key is None or key > best_key:
            best_model, best_params, best_score, best_key = model, params, score, key
    elapsed = time.perf_counter() - start
    parameter_count = (
        sum(w.size for w in best_model.coefs_) + sum(b.size for b in best_model.intercepts_)
        if name != "A" else None
    )
    return best_model, {
        "params": best_params, "val_clip_balanced_accuracy": float(best_score),
        "selection_metric": selection, "val_selection_key": [float(v) for v in best_key],
        "threshold": threshold, "fit_seconds": elapsed, "parameter_count": parameter_count,
        "support_vectors": int(best_model.n_support_.sum()) if name == "A" else None,
        "trials": trials,
    }
