"""Balanced training-recording subsets shared by every model in a run."""

import numpy as np


def group_labels(group_ids, labels):
    result = {}
    for group, label in zip(group_ids, labels):
        if result.setdefault(str(group), int(label)) != int(label):
            raise ValueError(f"Recording {group} has mixed labels; it cannot be subsampled by class.")
    return result


def max_balanced_size(labels_by_group):
    counts = np.bincount(list(labels_by_group.values()), minlength=2)
    return 2 * int(counts.min())


def balanced_subsets(labels_by_group, sizes, seed):
    """Nested balanced subsets: a seed's smaller subset is contained in its larger ones."""
    limit = max_balanced_size(labels_by_group)
    rng = np.random.default_rng(seed)
    order = {}
    for label in (0, 1):
        members = sorted(g for g, y in labels_by_group.items() if y == label)
        order[label] = [members[i] for i in rng.permutation(len(members))]
    result = {}
    for size in sizes:
        if size <= 0 or size % 2:
            raise ValueError(f"Training size {size} must be a positive even number of recordings.")
        if size > limit:
            # Never shrink silently: a smaller subset would be reported under the wrong size.
            raise ValueError(f"Training size {size} exceeds the largest balanced subset ({limit} "
                             "recordings); choose a smaller size.")
        result[int(size)] = sorted(order[0][:size // 2] + order[1][:size // 2])
    return result
